# Qwen × SuperGLUE — ZO comparison harness

Unified training script for comparing zeroth-order LLM fine-tuning methods
on a fixed setup (Qwen-0.5B + SuperGLUE MultiRC/COPA). Built so each team
member runs the **same** harness with a different optimizer, yielding
apples-to-apples cross-method numbers in one WandB project.

## What's pinned

| Item        | Value |
|---|---|
| Model       | `Qwen/Qwen3.5-0.8B` (fp16 — ZO does no backprop) |
| Tasks       | SuperGLUE **MultiRC** and **COPA** (more in `tasks.py:TASKS`) |
| Train ex.   | 1000 (MeZO paper convention) |
| Steps       | 20 000 |
| Eval period | every 500 steps |
| GPU         | 1× H100 (works on A100-40G too) |
| WandB proj  | `Zero-Order-Opt` |
| Run name    | `{owner}-{method}-{mm_dd_hh_mm_ss}` (UTC) |

Owner must be one of **`maria`**, **`nil`**, **`cheng`** — set via
`--owner`, `RUN_OWNER` env var, or the `owner:` field in the YAML.

## Metrics logged to WandB

- `train/loss`, `train/step_time_sec`, `train/avg_step_time_sec`, `train/step`
- `mem/current_MB`, `mem/peak_MB`
- `eval/logit_accuracy` — log-likelihood ranking over candidate verbalizers (the "right" SuperGLUE metric)
- `eval/token_accuracy` — argmax over the gold-completion tokens (MeZO-style)
- `final/total_steps`, `final/avg_step_time_sec`, `final/total_time_sec`, `final/peak_mem_MB`

## Run

```bash
pip install -r requirements.txt
export WANDB_API_KEY=...
export HF_TOKEN=...

# Pick optimizer × task:
python train.py --config configs/mezo.yaml         --task multirc --owner maria
python train.py --config configs/sparse_mezo.yaml  --task copa    --owner maria
python train.py --config configs/lozo.yaml         --task multirc --owner cheng
python train.py --config configs/dizo.yaml         --task multirc --owner nil
python train.py --config configs/hizoo.yaml        --task copa    --owner maria
```

10 combinations total (5 optimizers × 2 tasks) → 10 WandB runs.

## Adding a new SuperGLUE task

Edit `tasks.py`: define `format_train` and `format_eval` for the task and
register it in `TASKS`. No other code changes. Then `--task <new_name>`.

## Layout

```
train.py        # main script
tasks.py        # SuperGLUE task templates + LL-eval helpers
optimizers/     # EMPTY — implement each method from the original paper here
configs/        # one YAML per optimizer
requirements.txt
```

## Implementing an optimizer

The `optimizers/` folder is intentionally empty — each team member implements
their assigned method from scratch following the paper's reference code.

Convention:
- File: `optimizers/<lowercase_name>.py` (e.g. `optimizers/mezo.py`)
- Class: `<CamelCaseName>(torch.optim.Optimizer)` (e.g. `MeZO`)
- Must implement `.step(closure)` where `closure()` returns the loss tensor
  (ZO methods call the closure twice — once for `loss(x + ε·z)` and once for
  `loss(x − ε·z)` — to estimate the directional derivative).

The class name in the YAML's `optimizer.name` field must match the class name
in the module. `train.py:OPTIMIZER_MODULES` maps class name → module file —
add an entry there when you add a new method.
