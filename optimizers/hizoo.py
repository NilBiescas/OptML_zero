import torch
from torch.optim.optimizer import Optimizer


class HiZOO(Optimizer):
    """
    Hessian-Informed Zeroth-Order Optimizer (HiZOO).
    Zhao et al., NeurIPS 2024: "Second-Order Fine-Tuning without Pain for
    LLMs: A Hessian Informed Zeroth-Order Optimizer" (arXiv:2402.15173).

    Per step (Algorithm 1, three forward passes):
        L          = L(theta)
        L_plus     = L(theta + mu * sqrt(Sigma) * u)
        L_minus    = L(theta - mu * sqrt(Sigma) * u)
        diag(Sigma') = (L_plus + L_minus - 2*L) / (2 mu^2) * (u^2 * Sigma_inv)
        Sigma_inv  = (1 - alpha) * Sigma_inv + alpha * |diag(Sigma')|
        c          = (L_plus - L_minus) / (2 mu)
        theta      = theta - lr * c * sqrt(Sigma) * u

    Sigma is stored as its diagonal inverse (`sigma_inv`) for numerical
    stability. Memory cost is ~2x MeZO. Initialised to I, so the first
    step is exactly MeZO; preconditioning grows in as the EMA fills.

    The per-param seed combines a base seed with the optimizer step and a
    `param_id` attribute injected by the training loop — same convention as
    MeZO/LOZO/DiZO so multi-GPU runs stay in sync.

    Args:
        lr        (float): learning rate.
        eps       (float): perturbation scale (`mu` in the paper).
        alpha     (float): EMA rate for Sigma_inv. Larger = faster
            adaptation but more variance. Paper defaults around 1e-4 to 1e-3.
        sigma_eps (float): floor on Sigma_inv to avoid sqrt blow-up.
        seed      (int): base seed for synchronized RNG.
    """

    def __init__(self, params, lr=1e-6, eps=1e-3, alpha=1e-4,
                 sigma_eps=1e-8, seed=42):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]: {alpha}")
        defaults = dict(lr=lr, eps=eps, alpha=alpha,
                        sigma_eps=sigma_eps, seed=seed)
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise RuntimeError("HiZOO requires a closure that returns the loss.")

        eps       = self.defaults['eps']
        alpha     = self.defaults['alpha']
        sigma_eps = self.defaults['sigma_eps']
        seed      = self.defaults['seed']

        # ---- Pass 1: L(theta) at unperturbed weights ----
        L = closure()
        if isinstance(L, torch.Tensor):
            L = L.item()

        # ---- Sample u and apply forward perturbation: theta + mu*sqrt(Sigma)*u ----
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step']      = 0
                    state['sigma_inv'] = torch.ones_like(p.data)  # Sigma = I

                param_id   = getattr(p, 'param_id', 0)
                param_seed = seed + state['step'] + param_id * 1000003

                if 'generator' not in state:
                    state['generator'] = torch.Generator(device=p.device)
                state['generator'].manual_seed(param_seed)

                state['z'] = torch.randn(p.shape, dtype=p.dtype,
                                         device=p.device,
                                         generator=state['generator'])

                # sqrt(Sigma) = 1 / sqrt(Sigma_inv); clamp Sigma_inv from below
                sqrt_sigma = state['sigma_inv'].clamp(min=sigma_eps).rsqrt()
                state['sqrt_sigma_z'] = sqrt_sigma * state['z']

                p.add_(state['sqrt_sigma_z'], alpha=eps)

        L_plus = closure()
        if isinstance(L_plus, torch.Tensor):
            L_plus = L_plus.item()

        # ---- Backward perturbation: theta - mu*sqrt(Sigma)*u (net of forward) ----
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                p.add_(self.state[p]['sqrt_sigma_z'], alpha=-2.0 * eps)

        L_minus = closure()
        if isinstance(L_minus, torch.Tensor):
            L_minus = L_minus.item()

        # ---- Update Sigma_inv (EMA of |diag(Sigma')|) and theta ----
        # Loss-difference scalars are computed once and reused across params.
        c           = (L_plus - L_minus) / (2.0 * eps)
        h_scalar    = (L_plus + L_minus - 2.0 * L) / (2.0 * eps * eps)
        self._step_count += 1

        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]

                # diag(Sigma') = h_scalar * (u^2 * Sigma_inv)  — paper's form
                sigma_prime_diag = (h_scalar * state['z'].mul(state['z'])).mul(
                    state['sigma_inv']
                )
                state['sigma_inv'].mul_(1.0 - alpha).add_(
                    sigma_prime_diag.abs(), alpha=alpha
                )

                # Restore + update in one shot. Currently
                #   p = p_orig - eps * sqrt(Sigma)*u
                # We want
                #   p_new = p_orig - lr * c * sqrt(Sigma)*u
                # So add (eps - lr*c) * sqrt(Sigma)*u.
                p.add_(state['sqrt_sigma_z'], alpha=eps - lr * c)

                del state['z']
                del state['sqrt_sigma_z']
                state['step'] += 1

        return L
