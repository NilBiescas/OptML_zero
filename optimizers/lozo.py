"""LOZO: Low-rank Zeroth-Order Optimizer.

Chen et al., ICLR 2025 — https://arxiv.org/abs/2402.xxxxx
Reference: LOZO/large_models/LOZOtrainer.py

This file wraps the exact mathematical operations from zo_helpers.py
(LOZOTrainerHelper) into the `optimizer.step(closure)` interface.

Adaptations required by the harness interface:
  • zo_forward(model, inputs) → closure()  (called 2× per step)
  • self.v[name] is stored keyed by param_name (injected by train.py)
  • Learning rate comes from self.param_groups[0]["lr"]
  • Weight decay logic is preserved verbatim from the reference
"""

import math
import numpy as np
import torch
from torch.optim import Optimizer


class LOZO(Optimizer):
    """LOZO optimizer — low-rank perturbation ZO with lazy V resample.

    Faithful translation of LOZOTrainerHelper (zo_helpers.py) /
    LowRankTrainer (LOZO/large_models/LOZOtrainer.py) into the
    standard PyTorch optimizer.step(closure) interface.

    Args:
        params:          model.parameters()
        lr:              learning rate (paper: 1e-6)
        eps:             perturbation ε  (paper: 1e-3)
        weight_decay:    L2 regularisation (paper: 0)
        rank_r:          rank of the low-rank perturbation U V^T (paper: 8)
        step_interval:   how often to resample V (paper: 10)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        eps: float = 1e-3,
        weight_decay: float = 0.0,
        rank_r: int = 8,
        step_interval: int = 10,
    ):
        defaults = dict(lr=float(lr), eps=float(eps), weight_decay=float(weight_decay))
        super().__init__(params, defaults)
        self.rank_r = int(rank_r)
        self.step_interval = int(step_interval)
        self._step = 0
        self._v = {}             # param_name → V matrix (resampled every step_interval)
        self.last_metrics: dict = {}

    # ------------------------------------------------------------------
    # Verbatim from LOZOTrainerHelper.random_gaussian_matrix
    # ------------------------------------------------------------------
    def _random_gaussian_matrix(self, m, n, device, dtype):
        return torch.randn(m, n, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Verbatim from LOZOTrainerHelper.lowrank_zo_perturb_parameters
    # ------------------------------------------------------------------
    def _perturb(self, trainable, scaling_factor, zo_eps):
        for p in trainable:
            name = p.param_name
            if p.data.ndim >= 2:
                if self._step % self.step_interval == 0 or name not in self._v:
                    v = torch.randn(
                        p.data.size(1), self.rank_r,
                        device=p.data.device, dtype=p.data.dtype,
                    )
                    self._v[name] = v
                else:
                    v = self._v[name]
                u = self._random_gaussian_matrix(
                    m=p.data.size(0), n=self.rank_r,
                    device=p.data.device, dtype=p.data.dtype,
                )
                p.data = p.data + scaling_factor * (u @ v.t()) * zo_eps
            else:
                z = torch.normal(
                    mean=0, std=1, size=p.data.size(),
                    device=p.data.device, dtype=p.data.dtype,
                )
                p.data = p.data + scaling_factor * z * zo_eps

    # ------------------------------------------------------------------
    # step() — combines lowrank_zo_step + lowrank_zo_update from reference
    # ------------------------------------------------------------------
    def step(self, closure):  # type: ignore[override]
        assert closure is not None, "LOZO requires a closure"

        group = self.param_groups[0]
        zo_lr = group["lr"]
        zo_eps = group["eps"]
        weight_decay = group["weight_decay"]

        trainable = [
            p for g in self.param_groups
            for p in g["params"]
            if p.requires_grad
        ]

        # --- lowrank_zo_step ---
        zo_random_seed = np.random.randint(1000000000)

        with torch.no_grad():
            # First function evaluation: perturb +1
            torch.manual_seed(zo_random_seed)
            self._perturb(trainable, scaling_factor=1, zo_eps=zo_eps)
            loss1 = closure().detach()

            # Second function evaluation: perturb -2 (net = -1 from original)
            torch.manual_seed(zo_random_seed)
            self._perturb(trainable, scaling_factor=-2, zo_eps=zo_eps)
            loss2 = closure().detach()

            projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()

            # Reset model back: perturb +1 (net = 0)
            torch.manual_seed(zo_random_seed)
            self._perturb(trainable, scaling_factor=1, zo_eps=zo_eps)

        # --- lowrank_zo_update ---
        with torch.no_grad():
            torch.manual_seed(zo_random_seed)
            for p in trainable:
                name = p.param_name
                if p.data.ndim >= 2:
                    v = self._v[name]
                    u = self._random_gaussian_matrix(
                        m=p.data.size(0), n=self.rank_r,
                        device=p.data.device, dtype=p.data.dtype,
                    )
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        p.data = p.data - zo_lr * (projected_grad * (u @ v.t()) + weight_decay * p.data)
                    else:
                        p.data = p.data - zo_lr * (projected_grad * (u @ v.t()))
                else:
                    z = torch.normal(
                        mean=0, std=1, size=p.data.size(),
                        device=p.data.device, dtype=p.data.dtype,
                    )
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        p.data = p.data - zo_lr * (projected_grad * z + weight_decay * p.data)
                    else:
                        p.data = p.data - zo_lr * (projected_grad * z)

        self._step += 1

        self.last_metrics = {
            "projected_grad": projected_grad,
            "loss1": float(loss1.item()),
            "loss2": float(loss2.item()),
        }

        return loss1
