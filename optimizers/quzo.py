"""QuZO: Quantized Zeroth-Order optimizer.

Zhou et al., EMNLP 2025 — https://arxiv.org/abs/2502.12346
Reference: https://github.com/lloo099/QuZO

Core idea: perturb parameters with a *quantized* random direction z̃ (to
perturb_bits precision) instead of a full-precision z.  Both the forward
perturbation and the parameter update use the SAME z̃ — this is guaranteed by
(a) using the same torch seed for z_fp generation and (b) using a *local*
Generator for quantisation rounding so that the global RNG is never polluted
inside the param loop.

Without fix (a) + (b): torch.manual_seed() inside the quant call resets the
global RNG, making z_fp for params 2, 3, … inconsistent between perturb and
update → biased gradient estimate → model degradation.
"""

import math
import numpy as np
import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Quantisation helpers — deterministic, using a local Generator so the global
# RNG is never touched inside a per-param loop.
# ---------------------------------------------------------------------------

def _zo_quant_sym(
    x: torch.Tensor,
    nbits: int,
    seed: int | None = None,
    stochastic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric uniform quantisation of x to nbits bits.

    Uses block-exponent scaling (power-of-2 scale) from the paper.
    A local Generator is used for stochastic rounding so the caller's global
    RNG state is not disturbed.

    Returns (x_quant, scale) where dequant = x_quant / scale.
    """
    if not x.is_floating_point():
        x = x.float()

    n = 2 ** (nbits - 1)

    x1 = x.abs().max()
    if x1.item() == 0:
        return x.clone(), torch.ones(1, device=x.device, dtype=x.dtype)

    scale_raw = n / x1
    scale = 2.0 ** torch.floor(torch.log2(scale_raw))

    if stochastic:
        gen = torch.Generator(device=x.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        x_scaled = x * scale
        x_floor = torch.floor(x_scaled)
        rest = torch.clamp(x_scaled - x_floor, 0.0, 1.0)
        x_int = x_floor + torch.bernoulli(rest, generator=gen)
    else:
        x_int = torch.round(x * scale)

    x_quant = x_int.clamp(-n, n - 1)
    return x_quant, scale


def _zo_dequant(x_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x_quant / scale


# ---------------------------------------------------------------------------

class QuZO(Optimizer):
    """QuZO — quantised zeroth-order optimiser.

    Args:
        params:              model.parameters()
        lr:                  learning rate
        eps:                 perturbation ε (paper: 1e-3)
        perturb_bits:        bits for quantising the perturbation z (paper: 8)
        wbit:                bits for re-quantising weights after update.
                             Set 32 to disable (recommended when model is
                             already loaded in 8-bit via bitsandbytes, or
                             when only float norms/biases are trainable).
        quantized_perturb:   True → QuZO (quantised z̃); False → plain MeZO
        num_pertub:          number of ZO gradient estimates to average per step.
                             Each adds 2 forward passes. num_pertub=2 → 4 fwds/step,
                             variance ÷√2. Paper default: 1.
        weight_decay:        L2 regularisation (paper: 0)
        lr_scheduler:        "constant" | "cosine" | "linear"
        lr_min_ratio:        final LR multiplier for cosine/linear decay
        total_steps:         needed for schedule computation
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        eps: float = 1e-3,
        perturb_bits: int = 8,
        wbit: int = 32,
        quantized_perturb: bool = True,
        num_pertub: int = 1,
        weight_decay: float = 0.0,
        lr_scheduler: str = "constant",
        lr_min_ratio: float = 0.1,
        total_steps: int = 12000,
    ):
        defaults = dict(
            lr=float(lr),
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)
        self.perturb_bits      = int(perturb_bits)
        self.wbit              = int(wbit)
        self.quantized_perturb = bool(quantized_perturb)
        self.num_pertub        = max(1, int(num_pertub))
        self.lr_scheduler      = str(lr_scheduler)
        self.lr_min_ratio      = float(lr_min_ratio)
        self.total_steps       = int(total_steps)
        self._step             = 0
        self.last_metrics: dict = {}

    # ------------------------------------------------------------------
    def _effective_lr(self, base_lr: float) -> float:
        t = self._step / max(1, self.total_steps)
        if self.lr_scheduler == "cosine":
            return base_lr * (self.lr_min_ratio +
                   (1 - self.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))
        elif self.lr_scheduler == "linear":
            return base_lr * (1 - (1 - self.lr_min_ratio) * t)
        return base_lr  # "constant"

    # ------------------------------------------------------------------
    def _quantise_z(self, z_fp: torch.Tensor, seed: int, param_id: int) -> torch.Tensor:
        """Quantise z_fp to perturb_bits using a local Generator (no global RNG pollution)."""
        if not self.quantized_perturb:
            return z_fp
        z_q, z_scale = _zo_quant_sym(
            z_fp, nbits=self.perturb_bits,
            seed=seed + param_id,
            stochastic=True,   # unbiased — same seed → same result each call
        )
        return _zo_dequant(z_q, z_scale)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _perturb(self, trainable: list, seed: int, scale: float) -> None:
        """θ ← θ + scale * z̃ * eps.  Uses deterministic seed so perturb/update match."""
        zo_eps = self.param_groups[0]["eps"]
        # Reset the *global* RNG once before the loop so z_fp is reproducible.
        # _quantise_z uses a *local* Generator → does NOT advance the global RNG.
        torch.manual_seed(seed)
        for p in trainable:
            pid = getattr(p, "param_id", id(p) % (2**31))
            z_fp = torch.normal(0.0, 1.0, size=p.data.size(),
                                device=p.data.device,
                                dtype=p.data.dtype if p.data.is_floating_point()
                                      else torch.float32)
            z_eff = self._quantise_z(z_fp, seed, pid)

            if p.data.is_floating_point():
                p.data.add_(scale * z_eff * zo_eps)
            else:
                p.data = (p.data.float() + scale * z_eff * zo_eps).to(p.data.dtype)

    # ------------------------------------------------------------------
    def step(self, closure):  # type: ignore[override]
        assert closure is not None, "QuZO requires a closure"

        group        = self.param_groups[0]
        zo_eps       = group["eps"]
        weight_decay = group["weight_decay"]
        base_lr      = group["lr"]
        zo_lr        = self._effective_lr(base_lr)

        trainable = [
            p for g in self.param_groups
            for p in g["params"]
            if p.requires_grad
        ]

        # Collect num_pertub independent gradient estimates and average them.
        # Each uses a fresh random seed, contributing 2 forward passes.
        seeds = [np.random.randint(1_000_000_000) for _ in range(self.num_pertub)]

        with torch.no_grad():
            loss_first = None
            projected_grads = []

            for seed in seeds:
                # F(θ + εz̃)
                self._perturb(trainable, seed, scale=+1.0)
                loss1 = closure().detach()
                if loss_first is None:
                    loss_first = loss1

                # F(θ − εz̃)
                self._perturb(trainable, seed, scale=-2.0)
                loss2 = closure().detach()

                # Restore θ
                self._perturb(trainable, seed, scale=+1.0)

                projected_grads.append(((loss1 - loss2) / (2.0 * zo_eps)).item())

            # Apply accumulated updates from all perturbations (averaged).
            for seed, pg in zip(seeds, projected_grads):
                torch.manual_seed(seed)
                for p in trainable:
                    pid = getattr(p, "param_id", id(p) % (2**31))
                    z_fp = torch.normal(0.0, 1.0, size=p.data.size(),
                                        device=p.data.device,
                                        dtype=p.data.dtype if p.data.is_floating_point()
                                              else torch.float32)
                    z_eff = self._quantise_z(z_fp, seed, pid)
                    grad = (pg / self.num_pertub) * z_eff

                    if p.data.is_floating_point():
                        if weight_decay != 0.0:
                            grad = grad + (weight_decay / self.num_pertub) * p.data
                        p.data.add_(-zo_lr * grad)
                    else:
                        p_f = p.data.float()
                        if weight_decay != 0.0:
                            grad = grad + (weight_decay / self.num_pertub) * p_f
                        p.data = (p_f - zo_lr * grad).to(p.data.dtype)

            # Weight re-quantisation after all updates applied.
            if self.wbit < 32:
                for p in trainable:
                    if p.data.is_floating_point():
                        q, scale_w = _zo_quant_sym(p.data, nbits=self.wbit, stochastic=False)
                        p.data = _zo_dequant(q, scale_w).to(p.data.dtype)

        self._step += 1

        avg_pg = sum(projected_grads) / len(projected_grads)
        self.last_metrics = {
            "loss_plus":       float(loss_first.item()),
            "projected_grad":  avg_pg,
            "lr_effective":    zo_lr,
            "num_pertub":      self.num_pertub,
        }

        return loss_first
