"""HiZOO: Hessian Informed Zeroth-Order Optimizer.

Zhao et al., ICLR 2025 — https://arxiv.org/abs/2402.15173
Reference: https://github.com/Yanjun-Zhao/HiZOO

This file is a faithful translation of two reference functions into the
`optimizer.step(closure)` interface of the harness:

  trainer.py::efficient_Hessian_perturb_parameters
  trainer.py::zo_Hessian_step_update

The Hessian_smooth_scheduler (TYPE_TO_SCHEDULER_FUNCTION + Hessian_smooth_scheduler)
is copied verbatim from Hessian_smooth_scheduler.py.

Adaptations required by the harness interface:
  • zo_forward(model, inputs) → closure()  (called 3× per step)
  • self.Hessian_matrix[name] → self.state[p]["hessian"]  (param_id keyed,
    survives state_dict() / load_state_dict() automatically)
  • Hessian_smooth is re-evaluated each step (reference re-evaluates each epoch;
    for constant schedules the result is identical)
  • The 4th forward pass (loss_out after the update) is omitted; loss_original
    is returned instead and diagnostics go to self.last_metrics
"""

import math
import numpy as np
import torch
from torch.optim import Optimizer


# ---------------------------------------------------------------------------
# Hessian_smooth_scheduler — copied verbatim from
# HiZOO/Hessian_smooth_scheduler.py (TYPE_TO_SCHEDULER_FUNCTION + wrapper).
# Only the constant-* and annealed schedules used in the paper are kept;
# the full dict from the reference is preserved so any schedule from the paper
# can be selected via the YAML.
# ---------------------------------------------------------------------------

def get_constant0_schedule(current_step, num_training_steps):
    return 0

def get_constant2_schedule(current_step, num_training_steps):
    return 1e-2

def get_constant4_schedule(current_step, num_training_steps):
    return 1e-4

def get_constant6_schedule(current_step, num_training_steps):
    return 1e-6

def get_constant8_schedule(current_step, num_training_steps):
    return 1e-8

def get_constant10_schedule(current_step, num_training_steps):
    return 1e-10

def get_constant12_schedule(current_step, num_training_steps):
    return 1e-12

def get_constant_decay1_schedule(current_step, num_training_steps):
    if current_step < 9800:
        return 1e-6
    else:
        return 1e-8

def get_linear_schedule_with_warmup(current_step, num_training_steps,
                                    num_warmup_steps=0, **kw):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0,
               float(num_training_steps - current_step) /
               float(max(1, num_training_steps - num_warmup_steps)))

def get_cosine_schedule_with_warmup(current_step, num_training_steps,
                                    num_warmup_steps=0, num_cycles=2, **kw):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps))
    return max(0.0,
               0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

def get_polynomial_decay_schedule(current_step, num_training_steps,
                                  lr_end=1e-10, power=3, **kw):
    if current_step > num_training_steps:
        return lr_end
    pct_remaining = 1 - current_step / num_training_steps
    return (1.0 - lr_end) * pct_remaining ** power + lr_end

TYPE_TO_SCHEDULER_FUNCTION = {
    # constant schedules (paper defaults)
    'constant0':                       get_constant0_schedule,
    'constant1e-2':                    get_constant2_schedule,
    'constant1e-4':                    get_constant4_schedule,
    'constant1e-6':                    get_constant6_schedule,
    'constant1e-8':                    get_constant8_schedule,   # paper best result
    'constant1e-10':                   get_constant10_schedule,
    'constant1e-12':                   get_constant12_schedule,
    'constant_decay1':                 get_constant_decay1_schedule,
    # annealed schedules
    'linear_with_warmup':              get_linear_schedule_with_warmup,
    'cosine_with_warmup':              get_cosine_schedule_with_warmup,
    'polynomial_decay':                get_polynomial_decay_schedule,
}


def Hessian_smooth_scheduler(Hessian_smooth_type, current_step, num_training_steps):
    """Verbatim copy of reference Hessian_smooth_scheduler()."""
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[Hessian_smooth_type]
    return schedule_func(current_step, num_training_steps)


# ---------------------------------------------------------------------------
class HiZOO(Optimizer):
    """HiZOO optimizer — diagonal Hessian preconditioned zeroth-order.

    Faithful translation of trainer.py::zo_Hessian_step_update into the
    standard PyTorch optimizer.step(closure) interface.

    Args:
        params:               model.parameters()
        lr:                   learning rate (paper: 1e-6 for OPT-13B + RTE)
        eps:                  perturbation ε  (paper: 1e-3)
        hessian_smooth_type:  key into TYPE_TO_SCHEDULER_FUNCTION
                              (paper best result: "constant1e-8")
        weight_decay:         L2 regularisation (paper: 0)
        total_steps:          passed as num_training_steps to the scheduler
    """

    def __init__(
        self,
        params,
        lr: float = 1e-6,
        eps: float = 1e-3,
        hessian_smooth_type: str = "constant1e-8",
        weight_decay: float = 0.0,
        total_steps: int = 20000,
        lr_scheduler: str = "cosine",   # "constant" | "cosine" | "linear"
        lr_min_ratio: float = 0.1,      # final LR = lr * lr_min_ratio
    ):
        # YAML may deliver numeric kwargs as strings (e.g. lr: 1e-6 → '1e-6')
        defaults = dict(lr=float(lr), eps=float(eps), weight_decay=float(weight_decay))
        super().__init__(params, defaults)
        self.hessian_smooth_type = str(hessian_smooth_type)
        self.total_steps         = int(total_steps)
        self.lr_scheduler        = str(lr_scheduler)
        self.lr_min_ratio        = float(lr_min_ratio)
        self._step               = 0
        self.last_metrics: dict  = {}

    # ------------------------------------------------------------------
    # Hessian matrix — lives in self.state[p]["hessian"] so it is
    # automatically included in state_dict() / load_state_dict().
    # Initialised on first access to match reference:
    #   self.Hessian_matrix[name] = torch.ones(size=param.data.size(),
    #                                device=param.data.device,
    #                                dtype=param.data.dtype)
    # ------------------------------------------------------------------
    def _get_hessian(self, p: torch.Tensor) -> torch.Tensor:
        state = self.state[p]
        if "hessian" not in state:
            state["hessian"] = torch.ones(
                size=p.data.size(),
                device=p.data.device,
                dtype=p.data.dtype,
            )
        return state["hessian"]

    # ------------------------------------------------------------------
    # Verbatim translation of efficient_Hessian_perturb_parameters():
    #
    #   def efficient_Hessian_perturb_parameters(self, model, random_seed,
    #                                            Hessian_matrix, scaling_factor):
    #       torch.manual_seed(random_seed)
    #       for name, param in self.named_parameters_to_optim:
    #           z = torch.normal(mean=0, std=1, size=param.data.size(),
    #                            device=param.data.device, dtype=param.data.dtype)
    #           param.data = param.data + scaling_factor
    #                        / torch.sqrt(Hessian_matrix[name]) * z * self.args.zo_eps
    #
    # `named_parameters_to_optim` → `trainable` (list[Tensor] in param order).
    # `Hessian_matrix[name]`      → self.state[p]["hessian"].
    # ------------------------------------------------------------------
    def _efficient_hessian_perturb(
        self,
        trainable: list,
        random_seed: int,
        scaling_factor: float,
        zo_eps: float,
    ) -> None:
        torch.manual_seed(random_seed)
        for param in trainable:
            z = torch.normal(
                mean=0, std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            Hessian_matrix = self._get_hessian(param)
            param.data = param.data + scaling_factor / torch.sqrt(Hessian_matrix) * z * zo_eps

    # ------------------------------------------------------------------
    # step() — verbatim translation of zo_Hessian_step_update():
    #
    #   random_seed   = np.random.randint(1000000000)
    #   loss_original = zo_forward()              # closure call 1
    #   perturb(scaling=+1) ; loss1 = zo_forward()  # closure call 2
    #   perturb(scaling=-2) ; loss2 = zo_forward()  # closure call 3
    #   restore(scaling=+1)
    #   for name, param in named_parameters_to_optim:   [same seed]
    #       z             = torch.normal(...)
    #       Hessian_temp  = Hessian_matrix[name] * z * z
    #       H_est         = |loss1+loss2-2*loss_original| * Hessian_temp
    #                         * Hessian_smooth / (2*eps²)
    #       Hessian_matrix[name] = (1-Hessian_smooth)*Hessian_matrix[name] + H_est
    #       grad          = (loss1-loss2)/(2*eps) * z / sqrt(Hessian_matrix[name])
    #       param.data   -= zo_lr * (grad + weight_decay * param.data)
    # ------------------------------------------------------------------
    def step(self, closure):  # type: ignore[override]
        assert closure is not None, "HiZOO requires a closure"

        group        = self.param_groups[0]
        base_lr      = group["lr"]
        zo_eps       = group["eps"]
        weight_decay = group["weight_decay"]

        # LR schedule (applied on top of base_lr, entirely inside the optimizer)
        t = min(1.0, self._step / max(1, self.total_steps))  # clamp to [0,1]
        if self.lr_scheduler == "cosine":
            zo_lr = base_lr * (self.lr_min_ratio +
                    (1 - self.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * t)))
        elif self.lr_scheduler == "linear":
            zo_lr = base_lr * (1 - (1 - self.lr_min_ratio) * t)
        else:  # "constant"
            zo_lr = base_lr

        # Reference evaluates Hessian_smooth once per epoch at epoch start;
        # for constant schedules (paper default) this is identical to per-step.
        Hessian_smooth = Hessian_smooth_scheduler(
            self.hessian_smooth_type, self._step, self.total_steps
        )

        # named_parameters_to_optim equivalent (params in forward-pass order)
        trainable = [
            p for g in self.param_groups
            for p in g["params"]
            if p.requires_grad
        ]

        # random_seed = np.random.randint(1000000000)  — reference uses numpy
        random_seed = np.random.randint(1000000000)

        with torch.no_grad():
            # loss_original = zo_forward(model, inputs)
            loss_original = closure().detach()

            # first function evaluation
            self._efficient_hessian_perturb(trainable, random_seed, scaling_factor=1, zo_eps=zo_eps)
            loss1 = closure().detach()

            # second function evaluation
            self._efficient_hessian_perturb(trainable, random_seed, scaling_factor=-2, zo_eps=zo_eps)
            loss2 = closure().detach()

            # restore
            self._efficient_hessian_perturb(trainable, random_seed, scaling_factor=1, zo_eps=zo_eps)

            # Hessian + parameter update — verbatim loop from reference
            torch.manual_seed(random_seed)
            for param in trainable:
                z = torch.normal(
                    mean=0, std=1,
                    size=param.data.size(),
                    device=param.data.device,
                    dtype=param.data.dtype,
                )

                Hessian_temp = self._get_hessian(param) * z * z
                Hessian_estimator = (
                    torch.abs(loss1 + loss2 - 2 * loss_original)
                    * Hessian_temp
                    * Hessian_smooth
                    / (2 * zo_eps * zo_eps)
                )

                # in-place update of the stored tensor, then immediately read for grad
                self.state[param]["hessian"] = (
                    (1 - Hessian_smooth) * self.state[param]["hessian"]
                    + Hessian_estimator
                ).clamp(min=1e-8)  # floor prevents 1/sqrt(H)→inf in fp16

                grad = (
                    (loss1 - loss2) / (2 * zo_eps)
                    * z
                    / torch.sqrt(self.state[param]["hessian"])
                )
                param.data = param.data - zo_lr * (grad + weight_decay * param.data)

        self._step += 1

        # Diagnostics exposed for train.py → WandB opt/* keys
        self.last_metrics = {
            "loss_original":  float(loss_original.item()),
            "loss1":          float(loss1.item()),
            "loss2":          float(loss2.item()),
            "hessian_smooth": Hessian_smooth,
            "curvature_est":  float((loss1 + loss2 - 2 * loss_original).abs().item()),
            "lr_effective":   zo_lr,
        }

        return loss_original
