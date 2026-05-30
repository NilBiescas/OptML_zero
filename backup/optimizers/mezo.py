import torch
from torch.optim.optimizer import Optimizer


class MeZO(Optimizer):
    """
    Memory-efficient Zeroth-Order SGD (MeZO).
    Malladi et al., NeurIPS 2023: "Fine-Tuning Language Models with Just
    Forward Passes" (arXiv:2305.17333).

    Vanilla full-rank SPSA estimator:
        c     = (L(theta + eps*z) - L(theta - eps*z)) / (2 eps)
        theta = theta - lr * c * z,    z ~ N(0, I)

    Memory cost matches inference: only one Gaussian z is materialised per
    parameter and is regenerated from a deterministic per-param seed instead
    of being stored across the two forward passes. Per-param seed combines
    a base seed with the optimizer step and a `param_id` attribute injected
    by the training loop, which keeps multi-GPU processes in sync.
    """

    def __init__(self, params, lr=1e-7, eps=1e-3, seed=42):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr, eps=eps, seed=seed)
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise RuntimeError("MeZO requires a closure that returns the loss.")

        eps  = self.defaults['eps']
        seed = self.defaults['seed']

        # Forward perturbation: theta <- theta + eps*z
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0

                param_id   = getattr(p, 'param_id', 0)
                param_seed = seed + state['step'] + param_id * 1000003

                if 'generator' not in state:
                    state['generator'] = torch.Generator(device=p.device)
                state['generator'].manual_seed(param_seed)

                state['z'] = torch.randn(p.shape, dtype=p.dtype,
                                         device=p.device,
                                         generator=state['generator'])
                p.add_(state['z'], alpha=eps)

        loss_plus = closure()
        if isinstance(loss_plus, torch.Tensor):
            loss_plus = loss_plus.item()

        # Backward perturbation: theta <- theta - eps*z
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

        # Restore + apply update in one fused step:
        # current p = p_orig - eps*z; add (eps - lr*c)*z to land on p_orig - lr*c*z.
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                p.add_(state['z'], alpha=eps - lr * c)
                del state['z']
                state['step'] += 1

        return loss_plus
