"""Sparse-MeZO: magnitude-sparse zeroth-order SGD.

Liu et al., 2024 — "Sparse MeZO: Less Parameters for Better Performance
in Zeroth-Order LLM Fine-Tuning" (NUS-HPC-AI-Lab/SparseMeZO).

Update rule (per param tensor X, mask M, perturbation Z ~ N(0,I)):
    X+ = X + eps * (Z ⊙ M)
    X- = X - eps * (Z ⊙ M)
    c  = (L(X+) - L(X-)) / (2 * eps)        # projected gradient
    X  <- X - lr * c * (Z ⊙ M)

Mask M selects the `sparsity` fraction of entries with the SMALLEST
absolute values (paper default: sparsity=0.20 means 20% perturbed,
corresponding to the paper's "sparsity=0.80" convention).
mask_refresh=1 recomputes the mask every step (paper default).

Per-param RNG seeding via p.param_id (injected by train.py) keeps
multi-GPU processes in lock-step (same pattern as LOZO/HiZOO).
"""

import torch
from torch.optim import Optimizer


class SparseMeZO(Optimizer):
    """Sparse-MeZO optimizer.

    Args:
        params:        model.parameters()
        lr:            learning rate (paper Table 7: 1e-6)
        eps:           perturbation scale (paper: 1e-3)
        sparsity:      fraction of entries PERTURBED per step, in (0, 1].
                       sparsity=1.0 → vanilla MeZO. Paper "sparsity=0.80"
                       means 80% masked, 20% perturbed → use sparsity=0.20.
        mask_mode:     'small_magnitude' (paper default) | 'random'
        mask_refresh:  recompute magnitude mask every N steps.
                       1 = every step (paper default, dynamic mask).
        weight_decay:  L2 regularisation (paper default: 0)
        seed:          base RNG seed
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        eps: float = 1e-3,
        sparsity: float = 0.20,
        mask_mode: str = "small_magnitude",
        mask_refresh: int = 1,
        weight_decay: float = 0.0,
        seed: int = 42,
    ):
        if not 0.0 < sparsity <= 1.0:
            raise ValueError(f"sparsity must be in (0, 1]: {sparsity}")
        if mask_mode not in ("small_magnitude", "random"):
            raise ValueError(f"mask_mode must be 'small_magnitude' or 'random'")
        defaults = dict(
            lr=float(lr), eps=float(eps), sparsity=float(sparsity),
            mask_mode=mask_mode, mask_refresh=int(mask_refresh),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)
        self._base_seed = int(seed)
        self._step = 0
        self.last_metrics: dict = {}

    @staticmethod
    def _build_mask(p: torch.Tensor, sparsity: float, mode: str) -> torch.Tensor:
        """Return a binary mask (same dtype/device as p) selecting entries to perturb."""
        numel = p.numel()
        k = max(1, int(round(sparsity * numel)))
        if mode == "random":
            mask = torch.zeros(numel, device=p.device, dtype=p.dtype)
            idx = torch.randperm(numel, device=p.device)[:k]
            mask[idx] = 1.0
            return mask.view_as(p)
        # small_magnitude: perturb the k entries with smallest |w|
        # Cast to float32 first — torch.topk on bf16 raises RuntimeError on some CUDA versions
        flat_abs = p.detach().float().abs().view(-1)
        _, idx = torch.topk(flat_abs, k, largest=False, sorted=False)
        mask = torch.zeros(numel, device=p.device, dtype=p.dtype)
        mask[idx] = 1.0
        return mask.view_as(p)

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise RuntimeError("SparseMeZO requires a closure that returns the loss.")

        eps = self.defaults["eps"]
        sparsity = self.defaults["sparsity"]
        mask_mode = self.defaults["mask_mode"]
        mask_refresh = self.defaults["mask_refresh"]

        # ---- 1. Build / refresh mask + sample Z, then perturb X → X+ ----
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0

                param_id = getattr(p, "param_id", 0)
                param_seed = self._base_seed + state["step"] * 1000003 + param_id

                if "generator" not in state:
                    state["generator"] = torch.Generator(device=p.device)
                gen = state["generator"]
                gen.manual_seed(param_seed)

                # Refresh mask on schedule (always refresh on random mode or step 0)
                need_refresh = (
                    "mask" not in state
                    or mask_mode == "random"
                    or state["step"] % mask_refresh == 0
                )
                if need_refresh:
                    state["mask"] = self._build_mask(p, sparsity, mask_mode)

                # Sample Z ~ N(0,I), apply mask
                z = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
                state["mz"] = z * state["mask"]  # masked perturbation

                # Perturb: X → X+
                p.add_(state["mz"], alpha=eps)

        # ---- 2. First forward: L+ ----
        loss_plus = closure()
        if isinstance(loss_plus, torch.Tensor):
            loss_plus = loss_plus.item()

        # ---- 3. Perturb: X+ → X- (subtract 2eps) ----
        for group in self.param_groups:
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                p.add_(self.state[p]["mz"], alpha=-2.0 * eps)

        # ---- 4. Second forward: L- ----
        loss_minus = closure()
        if isinstance(loss_minus, torch.Tensor):
            loss_minus = loss_minus.item()

        # ---- 5. Projected gradient + restore + update in one fused step ----
        # Current state: X-  = X_orig - eps*mz
        # Target:        X'  = X_orig - lr*c*mz  = X- + (eps - lr*c)*mz
        c = (loss_plus - loss_minus) / (2.0 * eps)
        self._step += 1

        total_perturbed = 0
        total_params = 0
        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = self.state[p]
                mz = state["mz"]

                net_alpha = eps - lr * c
                if wd != 0.0:
                    # Weight decay applied only to perturbed entries (matches NUS ref)
                    p.mul_(1.0 - lr * wd * state["mask"])
                p.add_(mz, alpha=net_alpha)

                total_perturbed += state["mask"].sum().item()
                total_params += p.numel()

                del state["mz"]
                state["step"] += 1

        # ---- 6. Diagnostics ----
        self.last_metrics = {
            "projected_grad": c,
            "loss_plus": loss_plus,
            "loss_minus": loss_minus,
            "sparsity_eff": total_perturbed / max(1, total_params),
        }

        return loss_plus
