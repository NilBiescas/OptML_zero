import torch
from torch.optim.optimizer import Optimizer


class SparseMeZO(Optimizer):
    """
    Sparse-MeZO: zeroth-order SGD that perturbs only a sparse subset of each
    parameter tensor's entries per step. Selecting the smallest-magnitude
    entries (default) follows Liu et al., "Sparse MeZO: Less Parameters for
    Better Performance in Zeroth-Order LLM Fine-Tuning" (2024); a 'random'
    mode is also provided as a baseline.

    Update rule (per param tensor X with mask M and perturbation Z):
        X+ = X + eps * (Z ⊙ M)
        X- = X - eps * (Z ⊙ M)
        c  = (L(X+) - L(X-)) / (2 * eps)
        X  <- X - lr * c * (Z ⊙ M)

    Random seeds are synchronized per-parameter across GPU processes via the
    `param_id` attribute injected by the training loop (same pattern as LOZO).

    Args:
        lr (float): learning rate.
        eps (float): perturbation scale.
        sparsity (float): fraction of entries perturbed each step, in (0, 1].
            sparsity=1.0 recovers vanilla MeZO; sparsity=0.5 perturbs half.
        mask_mode (str): 'small_magnitude' | 'large_magnitude' | 'random'.
        mask_refresh (int): recompute the magnitude-based mask every N steps.
            Ignored for 'random' (mask is resampled every step).
        seed (int): base seed for the synchronized RNG.
    """

    def __init__(self, params, lr=1e-6, eps=1e-3, sparsity=0.5,
                 mask_mode="small_magnitude", mask_refresh=50, seed=42):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < sparsity <= 1.0:
            raise ValueError(f"sparsity must be in (0, 1]: {sparsity}")
        if mask_mode not in ("small_magnitude", "large_magnitude", "random"):
            raise ValueError(f"Invalid mask_mode: {mask_mode}")
        defaults = dict(lr=lr, eps=eps, sparsity=sparsity,
                        mask_mode=mask_mode, mask_refresh=mask_refresh,
                        seed=seed)
        super().__init__(params, defaults)

    @staticmethod
    def _build_mask(p, sparsity, mode, generator):
        numel = p.numel()
        k = max(1, int(round(sparsity * numel)))
        if mode == "random":
            # Bernoulli mask with expected density = sparsity; cheap and unbiased.
            rand = torch.rand(p.shape, device=p.device, generator=generator)
            return (rand < sparsity).to(p.dtype)
        # magnitude-based: pick the k smallest or largest |w|
        flat_abs = p.detach().abs().view(-1)
        largest = (mode == "large_magnitude")
        _, idx = torch.topk(flat_abs, k, largest=largest, sorted=False)
        mask_flat = torch.zeros(numel, device=p.device, dtype=p.dtype)
        mask_flat[idx] = 1.0
        return mask_flat.view_as(p)

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise RuntimeError("SparseMeZO requires a closure that computes the loss")

        eps = self.defaults['eps']
        seed = self.defaults['seed']

        # 1. Perturb X -> X + eps * (Z ⊙ M)
        for group in self.param_groups:
            sparsity = group['sparsity']
            mask_mode = group['mask_mode']
            mask_refresh = group['mask_refresh']
            for p in group['params']:
                if not p.requires_grad:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0

                param_id = getattr(p, 'param_id', 0)
                param_seed = seed + state['step'] + param_id * 1000003
                if 'generator' not in state:
                    state['generator'] = torch.Generator(device=p.device)
                generator = state['generator']
                generator.manual_seed(param_seed)

                # Refresh mask on schedule (random mode refreshes every step).
                need_refresh = (
                    'mask' not in state
                    or mask_mode == 'random'
                    or state['step'] % mask_refresh == 0
                )
                if need_refresh:
                    state['mask'] = self._build_mask(p, sparsity, mask_mode, generator)

                # Sample Z with the synchronized generator and mask it.
                state['Z'] = torch.randn(p.shape, dtype=p.dtype, device=p.device,
                                         generator=generator) * state['mask']

                p.add_(state['Z'], alpha=eps)

        loss_plus = closure()
        if isinstance(loss_plus, torch.Tensor):
            loss_plus = loss_plus.item()

        # 2. Perturb X -> X - eps * (Z ⊙ M)  (subtract 2*eps from current X+)
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                p.add_(state['Z'], alpha=-2 * eps)

        loss_minus = closure()
        if isinstance(loss_minus, torch.Tensor):
            loss_minus = loss_minus.item()

        c = (loss_plus - loss_minus) / (2 * eps)
        self._step_count = getattr(self, '_step_count', 0) + 1

        # 3. Reset to original X and apply update in one fused step:
        #    X- + eps*Z - lr*c*Z  ==  X + (eps - lr*c)*Z (with Z already masked).
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                net_alpha = eps - lr * c
                p.add_(state['Z'], alpha=net_alpha)
                state['step'] += 1

        return loss_plus
