"""Optimizer package.

Each ZO method lives in its own module here, e.g. `optimizers/mezo.py` exports
a class `MeZO(torch.optim.Optimizer)` with a `.step(closure)` method. The
training script imports them lazily by name (see train.py:OPTIMIZER_MODULES),
so only the module for the optimizer you're running needs to exist.

To add a new optimizer:
  1. Drop `optimizers/<lowercase_name>.py` with a class `<ClassName>`.
  2. Add `"<ClassName>": "<lowercase_name>"` to OPTIMIZER_MODULES in train.py.
  3. Add a YAML in configs/.

Implementations should follow the original paper's reference code as closely
as possible. Keep this folder empty of anything else.
"""
