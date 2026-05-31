import math
import torch
from torch.optim.optimizer import Optimizer


class ConMeZO(Optimizer):
    """
    Constrained-cone MeZO (ConMeZO).
    AISTATS 2026: "ConMeZO: A Constrained-Cone Zeroth-Order Optimizer for
    Memory-Efficient Fine-Tuning of LLMs" (arXiv:2511.02757).

    Per step:
        z         ~ N(0, I)
        if step==0 or ||mu||<tol:
            z'    = z                                   # plain MeZO warm-up
        else:
            mu_hat = mu / ||mu||
            alpha  = sin(cone_theta)                    # cone-mixing weight
            z'     = sqrt(1 - alpha^2) * z + alpha * ||z|| * mu_hat
        c          = (L(theta + eps*z') - L(theta - eps*z')) / (2 eps)
        theta      = theta - lr * c * z'
        mu         = beta * mu + (1 - beta) * c * z'

    The cone biasing preserves E[||z'||^2] (in expectation over Gaussian z)
    while concentrating the search direction toward the momentum-EMA mu, which
    halves the number of iterations needed to reach MeZO's final accuracy on
    most SuperGLUE tasks (paper Table 2).

    Args:
        lr         (float): learning rate.
        eps        (float): perturbation scale.
        cone_theta (float): cone half-angle in radians. The official OPT script
            (opt/examples/cone.sh) and RoBERTa script (roberta/examples/cone.sh)
            both default to 1.35. The paper text quotes 1.4 for OPT. We default
            to 1.4 to match the paper; the configs override per-stack.
            Larger = wider cone, closer to plain MeZO; alpha = sin(cone_theta).
        cone_beta  (float): FINAL EMA rate for the momentum direction mu. OPT
            script default 0.99, RoBERTa script default 0.95. The actual beta
            used at each step ramps up via the paper's warm-up schedule (see
            `cone_warmup_total` / Section 3.4 of arXiv:2511.02757).
        cone_warmup_total (int): total training steps used to size the warm-up
            schedule. Paper Section 3.4 specifies the schedule for 20K-step
            runs (phases 0-200, 200-2000) and notes "for 10K runs, halve the
            interval lengths". This kwarg lets us scale accordingly: phases
            are 0..cone_warmup_total/100 and that..cone_warmup_total/10.
        seed       (int): base seed for synchronized RNG.
    """

    def __init__(self, params, lr=1e-7, eps=1e-3,
                 cone_theta=1.4, cone_beta=0.99,
                 cone_warmup_total=20000, seed=42):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= cone_beta < 1.0:
            raise ValueError(f"cone_beta must be in [0, 1): {cone_beta}")
        if not 0.0 <= cone_theta <= math.pi / 2.0:
            raise ValueError(f"cone_theta must be in [0, pi/2]: {cone_theta}")
        if cone_warmup_total < 1:
            raise ValueError(f"cone_warmup_total must be >= 1: {cone_warmup_total}")
        defaults = dict(lr=lr, eps=eps,
                        cone_theta=cone_theta, cone_beta=cone_beta,
                        cone_warmup_total=cone_warmup_total, seed=seed)
        super().__init__(params, defaults)
        self._step_count = 0

    @staticmethod
    def _beta_warmup(t: int, beta_final: float, total: int) -> float:
        """Paper §3.4 warm-up schedule. For a 20K-step run:
            beta = 0.1                                       0 <= t <= 200
            beta = beta_final - (beta_final - 0.1)
                   / (1 + 8*((t-200)/1800)**1.8)**3          200 < t <= 2000
            beta = beta_final                                t > 2000
        For other run lengths we proportionally scale the interval endpoints
        (200 -> total/100, 2000 -> total/10).
        """
        warmup_short = max(1, total // 100)    # 200 for 20K
        warmup_long  = max(warmup_short + 1, total // 10)  # 2000 for 20K
        if t <= warmup_short:
            return 0.1
        if t <= warmup_long:
            denom_span = warmup_long - warmup_short
            x = (t - warmup_short) / denom_span
            return beta_final - (beta_final - 0.1) / (1.0 + 8.0 * (x ** 1.8)) ** 3
        return beta_final

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise RuntimeError("ConMeZO requires a closure that returns the loss.")

        eps        = self.defaults['eps']
        theta      = self.defaults['cone_theta']
        beta_final = self.defaults['cone_beta']
        seed       = self.defaults['seed']
        warmup_T   = self.defaults['cone_warmup_total']
        # Effective beta at this optimizer step, per the paper's warm-up.
        beta       = self._beta_warmup(self._step_count, beta_final, warmup_T)
        alpha = math.sin(theta)
        scale = math.sqrt(max(0.0, 1.0 - alpha * alpha))

        # Phase 0: lazy-init state and accumulate the GLOBAL momentum norm.
        # Paper Algorithm 1 builds z_t = sqrt(d) * (cos(theta) * mu_global_hat
        # + sin(theta) * u_global) over the *concatenated* parameter vector --
        # a single shared cone direction across the whole network. Computing
        # ||mu||_per_tensor and unit-normalizing per tensor (an earlier bug)
        # makes each tensor explore its own cone, dispersing exploration and
        # leaving ~2-3pp on the table vs paper Table 2.
        mu_norm_sq_global = 0.0
        total_d           = 0
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    # Keep momentum in fp32 regardless of param dtype: in fp16
                    # the cumulative EMA loses precision after ~100 steps and
                    # the cone direction degrades. The buffer cost is small.
                    state['mu']   = torch.zeros(p.shape, dtype=torch.float32,
                                                device=p.device)
                mu_norm_sq_global += state['mu'].pow(2).sum().item()
                total_d           += p.numel()
        mu_norm_global = math.sqrt(mu_norm_sq_global)
        sqrt_d         = math.sqrt(total_d)
        # Scalar that, multiplied by mu_local, yields the local view of
        # sqrt(d) * cos(theta) * mu_global / ||mu_global||.
        fac_global = (scale * sqrt_d / mu_norm_global) if mu_norm_global > 0.0 else 0.0
        cone_active = (self._step_count > 0) and (mu_norm_global > 1e-12)

        # Forward perturbation: theta <- theta + eps * z'
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]

                param_id   = getattr(p, 'param_id', 0)
                param_seed = seed + state['step'] + param_id * 1000003

                if 'generator' not in state:
                    state['generator'] = torch.Generator(device=p.device)
                state['generator'].manual_seed(param_seed)

                z = torch.randn(p.shape, dtype=p.dtype, device=p.device,
                                generator=state['generator'])

                # ConMeZO cone bias. Paper Algorithm 1 / Eq. 6 on the GLOBAL
                # flat parameter vector. Per-tensor decomposition:
                #   z'_local = sin(theta) * z_local + fac_global * mu_local
                # where fac_global = cos(theta) * sqrt(d_global) / ||mu_global||
                # and z_local ~ N(0, I) is the local slice of a global z that
                # approximates a unit-sphere sample after global normalization.
                # The cos+sin split has cos on the (parallel) mu-direction and
                # sin on the (orthogonal) random direction; at theta=1.35 this
                # is a WIDE cone (~0.98 random, ~0.22 mu) -- intentional soft
                # bias toward momentum (NOT mode-collapse into mu).
                if not cone_active:
                    # Plain-MeZO warm-up: cone hasn't formed yet.
                    z_prime = z
                else:
                    z_fp32 = z.float()
                    # z'_fp32 = sin(theta) * z + fac_global * mu_local
                    z_prime_fp32 = z_fp32.mul(alpha).add_(state['mu'],
                                                          alpha=fac_global)
                    z_prime = z_prime_fp32.to(p.dtype)

                state['z'] = z_prime
                p.add_(z_prime, alpha=eps)

        loss_plus = closure()
        if isinstance(loss_plus, torch.Tensor):
            loss_plus = loss_plus.item()

        # Backward perturbation: theta <- theta - eps * z' (net of forward)
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                p.add_(self.state[p]['z'], alpha=-2.0 * eps)

        loss_minus = closure()
        if isinstance(loss_minus, torch.Tensor):
            loss_minus = loss_minus.item()

        c = (loss_plus - loss_minus) / (2.0 * eps)
        self._step_count += 1

        # Restore + fused update + EMA momentum.
        # Currently p = p_orig - eps * z'. Add (eps - lr*c) * z' to land on
        # p_orig - lr*c*z'. Then update mu <- beta*mu + (1-beta) * c * z'.
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                z_prime = state['z']
                p.add_(z_prime, alpha=eps - lr * c)
                # mu update in fp32 — accumulate the param-dtype z' upcast.
                state['mu'].mul_(beta).add_(z_prime.float(),
                                            alpha=(1.0 - beta) * c)
                del state['z']
                state['step'] += 1

        return loss_plus
