import torch
from torch.optim.optimizer import Optimizer


class FZOO(Optimizer):
    """
    Fast Zeroth-Order Optimizer (FZOO).
    arXiv:2506.09034 — "FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning LLMs
    with Adaptive Batched Forward Passes".

    Three innovations vs MeZO:
        1. Rademacher (+/-1) perturbations instead of Gaussian.
        2. One-sided estimator with a reference loss L_0:
               c_k = (L_k - L_0) / eps
           where L_k = L(theta + eps * z_k) and L_0 = L(theta).
        3. Std-adaptive learning rate:
               sigma_hat = std({c_k}_{k=1..Nq})
               lr_eff    = lr / max(sigma_hat, sigma_floor)

    The paper's "batched forward" runs Nq perturbations through one batched
    pass via functorch.vmap. This implementation runs them sequentially
    (semantics-identical) because the existing closure pattern in train.py
    returns a single scalar loss per call. To match the paper's wall-clock
    gains a future revision can vmap the closure.

    Per-step structure (Nq + 1 forwards):
        L_0 = closure()                          # at unperturbed theta
        acc[p] = 0;  c_vals = []
        for k in 1..Nq:
            re-seed each param's generator with (param_seed + k)
            sample z_k Rademacher in {-1, +1}
            theta <- theta + eps * z_k
            L_k = closure()
            theta <- theta - eps * z_k           # restore
            c_k = (L_k - L_0) / eps
            c_vals.append(c_k)
            acc[p] += c_k * z_k
        sigma_hat = unbiased_std(c_vals)
        lr_eff    = lr / max(sigma_hat, sigma_floor)
        theta <- theta - lr_eff * acc / Nq

    DDP note: c_k is a scalar that comes out of closure(), which itself does
    accelerator.reduce(loss, mean). So sigma_hat is identical across ranks
    automatically — no extra all-reduce needed. The Rademacher draws are
    seeded with param_seed + k so they are also identical across ranks.

    Args:
        lr           (float): base learning rate.
        eps          (float): perturbation scale.
        Nq           (int):   number of probe queries per step (Nq >= 1).
        sigma_floor  (float): floor on sigma_hat to keep lr_eff finite.
        seed         (int):   base seed for synchronized RNG.
    """

    def __init__(self, params, lr=2e-7, eps=1e-3, Nq=8,
                 sigma_floor=1e-6, seed=42):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if Nq < 1:
            raise ValueError(f"Nq must be >= 1: {Nq}")
        if sigma_floor <= 0.0:
            raise ValueError(f"sigma_floor must be > 0: {sigma_floor}")
        defaults = dict(lr=lr, eps=eps, Nq=Nq,
                        sigma_floor=sigma_floor, seed=seed)
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise RuntimeError("FZOO requires a closure that returns the loss.")

        eps         = self.defaults['eps']
        Nq          = int(self.defaults['Nq'])
        sigma_floor = self.defaults['sigma_floor']
        seed        = self.defaults['seed']

        # 0. Reference loss at unperturbed theta.
        L_0 = closure()
        if isinstance(L_0, torch.Tensor):
            L_0 = L_0.item()

        # Lazy-init per-param state.
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                if 'generator' not in state:
                    state['generator'] = torch.Generator(device=p.device)
                state['acc'] = torch.zeros_like(p.data)

        c_vals = []

        # 1. Nq one-sided probes.
        for k in range(Nq):
            # 1a. Sample z_k for every param and perturb.
            for group in self.param_groups:
                for p in group['params']:
                    if not p.requires_grad:
                        continue
                    state = self.state[p]

                    param_id   = getattr(p, 'param_id', 0)
                    param_seed = seed + state['step'] + param_id * 1000003 + k
                    state['generator'].manual_seed(param_seed)

                    # Rademacher z_k in {-1, +1}, shape == p.shape.
                    z_int = torch.randint(0, 2, p.shape, device=p.device,
                                          generator=state['generator'],
                                          dtype=torch.int8)
                    z_k = z_int.to(p.dtype).mul_(2.0).sub_(1.0)
                    state['z_k'] = z_k
                    p.add_(z_k, alpha=eps)

            L_k = closure()
            if isinstance(L_k, torch.Tensor):
                L_k = L_k.item()

            # 1b. Restore theta and accumulate c_k * z_k into 'acc'.
            c_k = (L_k - L_0) / eps
            c_vals.append(c_k)

            for group in self.param_groups:
                for p in group['params']:
                    if not p.requires_grad:
                        continue
                    state = self.state[p]
                    z_k = state['z_k']
                    p.add_(z_k, alpha=-eps)              # restore
                    state['acc'].add_(z_k, alpha=c_k)    # acc += c_k * z_k
                    del state['z_k']

        # 2. Std-adaptive LR.
        if Nq >= 2:
            c_tensor  = torch.tensor(c_vals, dtype=torch.float64)
            sigma_hat = c_tensor.std(unbiased=True).item()
        else:
            sigma_hat = abs(c_vals[0])
        scale = 1.0 / max(sigma_hat, sigma_floor)

        # 3. Apply the averaged update with the std-adaptive lr.
        self._step_count += 1
        for group in self.param_groups:
            lr_eff = group['lr'] * scale
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                # update = (1/Nq) * acc; theta <- theta - lr_eff * update
                p.add_(state['acc'], alpha=-lr_eff / Nq)
                del state['acc']
                state['step'] += 1

        return L_0
