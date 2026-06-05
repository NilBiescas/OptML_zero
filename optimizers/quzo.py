"""QuZO: Quantized Zeroth-Order optimizer.

Zhou et al., EMNLP 2025 — https://arxiv.org/abs/2502.12346
Reference: https://github.com/lloo099/QuZO

Core idea: perturb parameters with a *quantized* random direction z̃ (to
perturb_bits precision) instead of a full-precision z.  After each parameter
update, the weights are re-quantized to wbit precision to stay in the
quantized-forward regime.  The combination yields significantly lower peak GPU
memory than MeZO at FP16, matching an 8-bit forward pass.

Harness adaptations:
  • zo_forward(model, inputs) → closure()         (called exactly 2× per step)
  • All quantisation helpers are inlined here;
    no dependency on the original QuZO repo's trainer files.
  • wbit < 32 triggers weight re-quantisation after each update;
    pass wbit=32 to disable (equivalent to plain MeZO with quantised z̃).
  • `quantized_perturb` flag toggles quantised z̃ (True = QuZO, False = MeZO).
"""

import math
import numpy as np
import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Quantisation helpers (ported + patched from lloo099/QuZO/large_models/)
# ---------------------------------------------------------------------------

def _zo_quant_sym(x: torch.Tensor, nbits: int, stochastic: bool = True,
                  seed: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric uniform quantisation of x to nbits bits.

    Returns (x_quant, scale) where dequant = x_quant / scale.
    Uses block-exponent scaling (power-of-2 scale) from the paper.
    """
    if not x.is_floating_point():
        x = x.float()

    n = 2 ** (nbits - 1)

    x1 = x.abs().max()
    if x1.item() == 0:
        # Constant-zero tensor — identity quantisation.
        return x.clone(), torch.ones(1, device=x.device, dtype=x.dtype)

    # Block-exponent scale (power of 2) — exact as in the paper.
    scale_raw = n / x1
    scale = 2.0 ** torch.floor(torch.log2(scale_raw))

    if seed is not None:
        torch.manual_seed(seed)

    if stochastic:
        x_scaled = x * scale
        x_floor = torch.floor(x_scaled)
        rest = torch.clamp(x_scaled - x_floor, 0.0, 1.0)
        x_int = x_floor + torch.bernoulli(rest)
    else:
        x_int = torch.round(x * scale)

    x_quant = x_int.clamp(-n, n - 1)
    return x_quant, scale


def _zo_dequant(x_quant: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x_quant / scale


def _zo_quant_data_sym(x: torch.Tensor, nbits: int,
                        stochastic: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric quantisation for weight re-quantisation after each update.

    Safe for Int8 tensors (bitsandbytes): casts to float32 first.
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
        x_scaled = x * scale
        x_floor = torch.floor(x_scaled)
        rest = torch.clamp(x_scaled - x_floor, 0.0, 1.0)
        x_int = x_floor + torch.bernoulli(rest)
    else:
        x_int = torch.round(x * scale)

    x_quant = x_int.clamp(-n, n - 1)
    return x_quant, scale


# ---------------------------------------------------------------------------

class QuZO(Optimizer):
    """QuZO — quantised zeroth-order optimiser.

    Args:
        params:              model.parameters()
        lr:                  learning rate (paper: 1e-6 for LLaMA-2-7B)
        eps:                 perturbation ε (paper: 1e-3)
        perturb_bits:        bits for quantising the perturbation z (paper: 8)
        wbit:                bits for re-quantising weights after update
                             (paper: 8; set 32 to disable weight re-quant)
        quantized_perturb:   True → QuZO (quantised z̃); False → plain MeZO
        weight_decay:        L2 regularisation (paper: 0)
        lr_scheduler:        "constant" | "cosine" | "linear"
        lr_min_ratio:        final LR multiplier for cosine/linear decay
        total_steps:         needed for schedule computation
    """

    def __init__(
        self,
        params,
        lr: float = 1e-7,
        eps: float = 1e-3,
        perturb_bits: int = 8,
        wbit: int = 8,
        quantized_perturb: bool = True,
        weight_decay: float = 0.0,
        lr_scheduler: str = "cosine",
        lr_min_ratio: float = 0.1,
        total_steps: int = 12000,
    ):
        # YAML may deliver numeric kwargs as strings.
        defaults = dict(
            lr=float(lr),
            eps=float(eps),
            weight_decay=float(weight_decay),
        )
        super().__init__(params, defaults)
        self.perturb_bits      = int(perturb_bits)
        self.wbit              = int(wbit)
        self.quantized_perturb = bool(quantized_perturb)
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
    @torch.no_grad()
    def _perturb(self, trainable, seed, scale):
        """Add scale * z̃ * eps to every trainable parameter (in-place)."""
        group = self.param_groups[0]
        zo_eps = group["eps"]
        torch.manual_seed(seed)
        for p in trainable:
            dtype = p.data.dtype if p.data.is_floating_point() else torch.float32
            z_fp = torch.normal(0, 1, size=p.data.size(),
                                device=p.data.device, dtype=dtype)
            if self.quantized_perturb:
                # Quantise the random direction z then dequantise — core QuZO step.
                z_q, z_scale = _zo_quant_sym(z_fp, nbits=self.perturb_bits,
                                              stochastic=True,
                                              seed=seed + getattr(p, 'param_id', 0))
                z_eff = _zo_dequant(z_q, z_scale)
            else:
                z_eff = z_fp

            if p.data.is_floating_point():
                p.data.add_(scale * z_eff * zo_eps)
            else:
                # Int8 bitsandbytes param — operate in float, write back.
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

        seed = np.random.randint(1_000_000_000)

        with torch.no_grad():
            # F(θ + εz̃)
            self._perturb(trainable, seed, scale=+1.0)
            loss1 = closure().detach()

            # F(θ − εz̃)  (net: θ − 2εz̃)
            self._perturb(trainable, seed, scale=-2.0)
            loss2 = closure().detach()

            # Restore θ
            self._perturb(trainable, seed, scale=+1.0)

            projected_grad = ((loss1 - loss2) / (2.0 * zo_eps)).item()

            # Parameter update using the same quantised z̃
            torch.manual_seed(seed)
            for p in trainable:
                dtype = p.data.dtype if p.data.is_floating_point() else torch.float32
                z_fp = torch.normal(0, 1, size=p.data.size(),
                                    device=p.data.device, dtype=dtype)
                if self.quantized_perturb:
                    z_q, z_scale = _zo_quant_sym(z_fp, nbits=self.perturb_bits,
                                                  stochastic=False,
                                                  seed=seed + getattr(p, 'param_id', 0))
                    z_eff = _zo_dequant(z_q, z_scale)
                else:
                    z_eff = z_fp

                grad = projected_grad * z_eff

                if p.data.is_floating_point():
                    if weight_decay != 0.0:
                        grad = grad + weight_decay * p.data
                    p.data.add_(-zo_lr * grad)
                else:
                    p_float = p.data.float()
                    if weight_decay != 0.0:
                        grad = grad + weight_decay * p_float
                    p.data = (p_float - zo_lr * grad).to(p.data.dtype)

                # Weight re-quantisation: simulate 8-bit quantisation-aware update.
                # Disabled when wbit=32 or for non-float params (already quantised
                # by bitsandbytes; we'd double-quantise).
                if self.wbit < 32 and p.data.is_floating_point():
                    q, scale = _zo_quant_data_sym(p.data, nbits=self.wbit,
                                                   stochastic=True)
                    p.data = _zo_dequant(q, scale).to(p.data.dtype)

        self._step += 1

        self.last_metrics = {
            "loss_plus":       float(loss1.item()),
            "loss_minus":      float(loss2.item()),
            "projected_grad":  projected_grad,
            "lr_effective":    zo_lr,
        }

        return loss1
