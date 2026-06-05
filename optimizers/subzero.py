"""SubZero: Subspace Zeroth-Order Optimizer.

Yu et al., ICCV 2025 — per-layer orthogonal U, V perturbation
in q-dim subspace; variance O(q) not O(d).
Reference: SubZero/large_models/subzero_helper.py

This file wraps the exact mathematical operations from zo_helpers.py
(SubZeroTrainerHelper) into the `optimizer.step(closure)` interface.

Adaptations required by the harness interface:
  • zo_forward(model, inputs) → closure()  (called 2× per step)
  • self.p_state[name] → self._p_state[param_name]
  • Learning rate comes from self.param_groups[0]["lr"]
  • Weight decay is handled via the underlying SGD-style update
    (reference uses self.optimizer.step() with param.grad, but here
     we inline the update to avoid needing a nested optimizer)

NOTE on SubZero's original update:
  The reference code sets `param.grad = projected_grad * z` and then
  calls `self.optimizer.step()` (an AdamW / SGD). This means SubZero's
  update is NOT a simple SGD step on the ZO gradient — it uses whatever
  optimizer was configured (typically SGD with momentum or AdamW).
  In our harness, train.py creates only ONE optimizer (this class), so
  we inline a plain SGD update (lr * grad + weight_decay * param) which
  matches the reference when the inner optimizer is SGD without momentum.
"""

import math
import numpy as np
import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Verbatim from SubZero/large_models/subzero_helper.py
# ---------------------------------------------------------------------------
def _fast_svd_method_v2(w_shape, device, dtype, rank=8):
    U, _ = torch.linalg.qr(torch.randn((w_shape[0], rank), device=device))
    U = U.to(dtype).contiguous()
    V, _ = torch.linalg.qr(torch.randn((w_shape[1], rank), device=device))
    Vt = V.to(dtype).T.contiguous()
    return U, Vt


def _reshape_matrix(integer):
    factor1, factor2 = 1, integer
    for i in range(1, int(math.sqrt(integer)) + 1):
        if int(integer / i) == integer / i:
            if integer / i - i < factor2 - factor1:
                factor1, factor2 = i, integer / i
    return factor1, int(factor2)


class SubZero(Optimizer):
    """SubZero optimizer — subspace ZO with orthogonal U, V perturbation.

    Faithful translation of SubZeroTrainerHelper (zo_helpers.py) /
    SubZero/large_models/subzero_helper.py into the standard PyTorch
    optimizer.step(closure) interface.

    Args:
        params:             model.parameters()
        lr:                 learning rate (paper: 1e-6)
        eps:                perturbation ε  (paper: 1e-3)
        weight_decay:       L2 regularisation (paper: 0)
        gauss_rank:         rank of the QR subspace (paper: 8)
        update_interval:    how often to re-compute U, V (paper: 10)
        perturbation_mode:  "two_side" (default) or "one_side"
        mode:               "full" (default) or "lora"/"prefix"/"prompt"
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        eps: float = 1e-3,
        weight_decay: float = 0.0,
        gauss_rank: int = 8,
        update_interval: int = 10,
        perturbation_mode: str = "two_side",
        mode: str = "full",
    ):
        defaults = dict(lr=float(lr), eps=float(eps), weight_decay=float(weight_decay))
        super().__init__(params, defaults)
        self.gauss_rank = int(gauss_rank)
        self.update_interval = int(update_interval)
        self.perturbation_mode = str(perturbation_mode)
        self.mode = str(mode)
        self._step = 0
        self._p_state = {}       # param_name → {'U': ..., 'V': ...}
        self.last_metrics: dict = {}

    # ------------------------------------------------------------------
    # Build the list [(param, U, V), ...] — verbatim from
    # SubZeroTrainerHelper.zo_subspace_step parameter-init block
    # ------------------------------------------------------------------
    def _build_trainable(self):
        result = []
        for g in self.param_groups:
            for p in g["params"]:
                if not p.requires_grad:
                    continue
                name = p.param_name
                if len(p.data.shape) == 2:
                    if self._step == 0 or name not in self._p_state:
                        self._p_state[name] = {
                            'U': torch.zeros(p.data.size(0), self.gauss_rank,
                                             device=p.device, dtype=p.dtype),
                            'V': torch.zeros(self.gauss_rank, p.data.size(1),
                                             device=p.device, dtype=p.dtype),
                        }
                    ps = self._p_state[name]
                    if self._step % self.update_interval == 0:
                        if self.mode in ('lora', 'prefix', 'prompt'):
                            w_shape = _reshape_matrix(p.data.numel())
                            U, V = _fast_svd_method_v2(
                                w_shape=w_shape, device=p.device,
                                dtype=p.data.dtype, rank=self.gauss_rank,
                            )
                        else:
                            U, V = _fast_svd_method_v2(
                                w_shape=p.data.shape, device=p.device,
                                dtype=p.data.dtype, rank=self.gauss_rank,
                            )
                        ps['U'] = U
                        ps['V'] = V
                    result.append((p, ps['U'], ps['V']))
                else:
                    result.append((p, torch.tensor([1.0]), torch.tensor([1.0])))
                p.grad = None
        return result

    # ------------------------------------------------------------------
    # Verbatim from SubZeroTrainerHelper.zo_subspace_perturb_parameters
    # ------------------------------------------------------------------
    def _perturb(self, trainable, scaling_factor, zo_eps):
        for p, U, V in trainable:
            if len(U.shape) == 1:
                z = torch.normal(
                    mean=0, std=1, size=p.data.size(),
                    device=p.data.device, dtype=p.data.dtype,
                )
                p.data = p.data + scaling_factor * z * zo_eps
            elif len(U.shape) == 2:
                z = torch.normal(
                    mean=0, std=1, size=(U.shape[1], V.shape[0]),
                    device=p.data.device, dtype=p.data.dtype,
                )
                z = (U @ z @ V * math.sqrt(p.data.numel() / z.numel())).view(p.data.shape)
                p.data = p.data + scaling_factor * z * zo_eps

    # ------------------------------------------------------------------
    # step() — combines zo_subspace_step + zo_subspace_update
    # ------------------------------------------------------------------
    def step(self, closure):  # type: ignore[override]
        assert closure is not None, "SubZero requires a closure"

        group = self.param_groups[0]
        zo_lr = group["lr"]
        zo_eps = group["eps"]
        weight_decay = group["weight_decay"]

        trainable = self._build_trainable()

        zo_random_seed = np.random.randint(1000000000)

        with torch.no_grad():
            # First function evaluation
            torch.manual_seed(zo_random_seed)
            self._perturb(trainable, scaling_factor=1, zo_eps=zo_eps)
            loss1 = closure().detach()

            # Second function evaluation
            if self.perturbation_mode == "one_side":
                torch.manual_seed(zo_random_seed)
                self._perturb(trainable, scaling_factor=-1, zo_eps=zo_eps)
                loss2 = closure().detach()
                projected_grad = ((loss1 - loss2) / zo_eps).item()
            else:  # two_side (default)
                torch.manual_seed(zo_random_seed)
                self._perturb(trainable, scaling_factor=-2, zo_eps=zo_eps)
                loss2 = closure().detach()
                projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()
                # Reset model back to original
                torch.manual_seed(zo_random_seed)
                self._perturb(trainable, scaling_factor=1, zo_eps=zo_eps)

        # --- zo_subspace_update ---
        with torch.no_grad():
            torch.manual_seed(zo_random_seed)
            for p, U, V in trainable:
                if len(p.data.shape) == 2:
                    z0 = torch.normal(
                        mean=0, std=1,
                        size=(self.gauss_rank, self.gauss_rank),
                        device=p.data.device, dtype=p.data.dtype,
                    )
                    z = (U @ z0 @ V * math.sqrt(p.data.numel() / z0.numel())).view(p.data.shape).to(p.data.dtype)
                else:
                    z = torch.normal(
                        mean=0, std=1, size=p.data.size(),
                        device=p.data.device, dtype=p.data.dtype,
                    )

                # Inline SGD update: param -= lr * (projected_grad * z + wd * param)
                grad = projected_grad * z
                if weight_decay > 0:
                    p.data = p.data - zo_lr * (grad + weight_decay * p.data)
                else:
                    p.data = p.data - zo_lr * grad

        self._step += 1

        self.last_metrics = {
            "projected_grad": projected_grad,
            "loss1": float(loss1.item()),
            "loss2": float(loss2.item()),
        }

        return loss1
