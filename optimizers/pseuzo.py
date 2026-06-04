"""PseuZO: Pseudo Zeroth-Order Optimizer.

NeurIPS 2025 — Jacobian estimation via model output differentiation +
EMA variance reduction with a sliding window.
Reference: PseuZO/pzo_helper.py, PseuZO/pzo_trainer.py

This file wraps the exact mathematical operations from zo_helpers.py
(PZOTrainerHelper) into the `optimizer.step(closure)` interface.

Adaptations required by the harness interface:
  • pzo_forward(model, inputs, need_grad=True)
      → closure(need_output=True) returns (loss, last_hidden, ∂L/∂h)
    pzo_forward(model, inputs, need_grad=False)
      → closure()            returns loss (but PseuZO also needs logits)
    IMPORTANT: the harness closure(need_output=True) returns ∂L/∂last_hidden
    (gradient w.r.t. the last hidden state before lm_head), whereas the
    original PseuZO computes ∂L/∂logits. To stay faithful to the reference
    math, we use the harness-provided (loss, last_hidden, grad_h) directly:
      - last_hidden plays the role of "o" (the output snapshot)
      - grad_h plays the role of "grad_last" (the Jacobian direction)
    The dot product <do, grad_last> is the same chain-rule quantity in both
    representations (change in output times output-space gradient of loss).

  • pzo_perturb_parameters → self._perturb (loops over self.param_groups)
  • self.named_parameters_to_optim → built from param_groups each step
  • Learning rate comes from self.param_groups[0]["lr"]
  • Weight decay logic is preserved verbatim from the reference
"""

import math
import numpy as np
import torch
from collections import deque
from torch.optim import Optimizer


class PseuZO(Optimizer):
    """PseuZO optimizer — pseudo ZO with sliding-window EMA.

    Faithful translation of PZOTrainerHelper (zo_helpers.py) /
    PseuZO/pzo_helper.py into the standard PyTorch optimizer.step(closure)
    interface.

    Args:
        params:                 model.parameters()
        lr:                     learning rate (paper: 1e-6)
        eps:                    perturbation ε  (paper: 1e-3)
        weight_decay:           L2 regularisation (paper: 0)
        sliding_window_length:  size of the history window (paper: 14)
        momentum_fb:            EMA momentum (paper: 0.9)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        eps: float = 1e-3,
        weight_decay: float = 0.0,
        sliding_window_length: int = 14,
        momentum_fb: float = 0.9,
    ):
        defaults = dict(lr=float(lr), eps=float(eps), weight_decay=float(weight_decay))
        super().__init__(params, defaults)
        self.sliding_window_length = int(sliding_window_length)
        self.momentum_fb = float(momentum_fb)
        self._sliding_window = deque(maxlen=self.sliding_window_length)
        self._grad_last = None
        self._step = 0
        self.last_metrics: dict = {}

        # Pre-compute the EMA coefficients (verbatim from reference)
        self._coefficients = self._build_coefficients()

    # ------------------------------------------------------------------
    # Verbatim from PZOTrainerHelper.reset_momentum_fb
    # ------------------------------------------------------------------
    def _build_coefficients(self):
        coefficients = []
        for i in range(self.sliding_window_length):
            if i == 0:
                coefficients.append(1.0)
            else:
                coefficients = [co * self.momentum_fb for co in coefficients]
                coefficients.append(1.0)
        return coefficients

    # ------------------------------------------------------------------
    # Dynamic EMA Momentum Schedule (from pzo_trainer.py)
    # ------------------------------------------------------------------
    def on_epoch_start(self, epoch: int, total_epochs: int):
        import os
        momentum_env = os.environ.get("MOMENTUM_FB", None)
        if momentum_env is not None:
            self.momentum_fb = float(momentum_env)
        else:
            # cyclic_hyperbola(t, T, k=2)
            T = total_epochs
            k = 2
            cyc = max(1, T // k)
            if epoch >= cyc * k:
                self.momentum_fb = 0.0
            else:
                t = epoch % cyc
                self.momentum_fb = 1.0 / (1 + 10 * t) if t <= 10 else 0.0
        
        self._coefficients = self._build_coefficients()
        self.last_metrics["momentum_fb"] = self.momentum_fb

    # ------------------------------------------------------------------
    # Verbatim from PZOTrainerHelper.Random_noise
    # ------------------------------------------------------------------
    @staticmethod
    def _random_noise(size, device, dtype, noise_type='Gaussian'):
        if noise_type == 'Gaussian':
            return torch.normal(mean=0, std=1, size=size, device=device, dtype=dtype)
        elif noise_type == 'Rademacher':
            return torch.randint(0, 2, size=size, device=device, dtype=dtype) - 1
        else:
            raise NotImplementedError

    # ------------------------------------------------------------------
    # Verbatim from PZOTrainerHelper.pzo_perturb_parameters
    # ------------------------------------------------------------------
    def _perturb(self, trainable, random_seed, scaling_factor, zo_eps):
        torch.manual_seed(random_seed)
        for p in trainable:
            z = self._random_noise(
                size=p.data.size(), device=p.data.device,
                dtype=p.data.dtype, noise_type='Gaussian',
            )
            p.data = p.data + scaling_factor * z * zo_eps

    # ------------------------------------------------------------------
    # step() — combines pzo_step + pzo_update from reference
    #
    # PseuZO requires the enriched closure form:
    #   closure(need_output=True) → (loss, last_hidden, grad_h)
    # It calls the closure 2× per step:
    #   1) need_output=True  → unperturbed forward (gets o0, grad_last)
    #   2) standard closure  → perturbed forward   (gets loss1, and we
    #      separately get o1 via need_output=True)
    #
    # However, to reduce forward passes while preserving the math, we
    # call need_output=True twice (the harness counts each as 1 forward):
    #   call 1: unperturbed → (loss0, o0, grad_last)
    #   perturb +eps
    #   call 2: perturbed   → (loss1, o1, _)
    #   perturb -eps (restore)
    #   do = o1 - o0; sliding_window.append((seed, do))
    #   Then update using the sliding window.
    # ------------------------------------------------------------------
    def step(self, closure):  # type: ignore[override]
        assert closure is not None, "PseuZO requires a closure"

        group = self.param_groups[0]
        zo_lr = group["lr"]
        zo_eps = group["eps"]
        weight_decay = group["weight_decay"]

        trainable = [
            p for g in self.param_groups
            for p in g["params"]
            if p.requires_grad
        ]

        # --- pzo_step ---
        random_seed = np.random.randint(1000000000)

        # Call 1: unperturbed forward with output (need_grad equivalent)
        loss0, o0, grad_last = closure(need_output=True)
        loss0 = loss0.detach()
        o0 = o0.detach()
        grad_last = grad_last.detach()

        # Perturb parameters
        self._perturb(trainable, random_seed, scaling_factor=1, zo_eps=zo_eps)

        # Call 2: perturbed forward with output
        loss1, o1, _ = closure(need_output=True)
        loss1 = loss1.detach()
        o1 = o1.detach()

        # Store grad_last for the update
        # Original pzo_trainer.py:947 uses grad_last[0] — only first batch element
        self._grad_last = grad_last[0:1]

        # Restore parameters
        self._perturb(trainable, random_seed, scaling_factor=-1, zo_eps=zo_eps)

        # Compute output difference and store in sliding window
        do = o1 - o0
        self._sliding_window.append((random_seed, do))

        # --- pzo_update ---
        dot_products = []
        random_seeds = []
        for seed, do_hist in self._sliding_window:
            # Handle shape mismatches (verbatim from reference)
            if do_hist.dim() == 3 and self._grad_last.dim() == 3:
                b1, seq1, _ = do_hist.shape
                b2, seq2, _ = self._grad_last.shape
                min_b = min(b1, b2)
                min_seq = min(seq1, seq2)
                do_sliced = do_hist[:min_b, -min_seq:, :]
                grad_sliced = self._grad_last[:min_b, -min_seq:, :]
            elif do_hist.dim() == 2 and self._grad_last.dim() == 2:
                s1, _ = do_hist.shape
                s2, _ = self._grad_last.shape
                min_s = min(s1, s2)
                do_sliced = do_hist[-min_s:, :]
                grad_sliced = self._grad_last[-min_s:, :]
            else:
                # Fallback: flatten and dot
                min_n = min(do_hist.numel(), self._grad_last.numel())
                do_sliced = do_hist.reshape(-1)[:min_n]
                grad_sliced = self._grad_last.reshape(-1)[:min_n]

            dot_product = torch.sum(do_sliced * grad_sliced)
            dot_products.append(dot_product)
            random_seeds.append(seed)

        n = len(dot_products)
        coefficients = [
            co * dot.item() / zo_eps
            for co, dot in zip(self._coefficients[-n:], dot_products)
        ]

        with torch.no_grad():
            for project_value, seed in zip(coefficients, random_seeds):
                torch.manual_seed(seed)
                for p in trainable:
                    name = p.param_name
                    z = self._random_noise(
                        size=p.data.size(), device=p.data.device,
                        dtype=p.data.dtype, noise_type='Gaussian',
                    )
                    if "bias" not in name and "layer_norm" not in name and "layernorm" not in name:
                        p.data = p.data - zo_lr * (project_value * z + weight_decay * p.data)
                    else:
                        p.data = p.data - zo_lr * (project_value * z)

        self._step += 1

        self.last_metrics = {
            "loss_unperturbed": float(loss0.item()),
            "loss_perturbed": float(loss1.item()),
            "window_size": len(self._sliding_window),
            "n_update_dirs": n,
        }

        return loss1
