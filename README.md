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

# 11 methods × 2 tasks = 22 runs total. Owner is read from each YAML by default;
# override with --owner or RUN_OWNER if needed.
python train.py --config configs/mezo.yaml        --task multirc
python train.py --config configs/sparse_mezo.yaml --task copa
python train.py --config configs/hizoo.yaml       --task multirc
python train.py --config configs/quzo.yaml        --task copa
python train.py --config configs/lozo.yaml        --task multirc
python train.py --config configs/dizo.yaml        --task copa
python train.py --config configs/zo_muon.yaml     --task multirc
python train.py --config configs/conmezo.yaml     --task copa
python train.py --config configs/fzoo.yaml        --task multirc
python train.py --config configs/pseuzo.yaml      --task copa
python train.py --config configs/subzero.yaml     --task multirc
```

| Method      | Owner | Class        | Module                       |
|---|---|---|---|
| MeZO        | cheng | `MeZO`       | `optimizers/mezo.py`         |
| Sparse-MeZO | maria | `SparseMeZO` | `optimizers/sparse_mezo.py`  |
| HiZOO       | maria | `HiZOO`      | `optimizers/hizoo.py`        |
| QuZO        | maria | `QuZO`       | `optimizers/quzo.py`         |
| LOZO        | nil   | `LOZO`       | `optimizers/lozo.py`         |
| DiZO        | cheng | `DiZO`       | `optimizers/dizo.py`         |
| ZO-Muon     | cheng | `ZOMuon`     | `optimizers/zo_muon.py`      |
| ConMeZO     | cheng | `ConMeZO`    | `optimizers/conmezo.py`      |
| FZOO        | cheng | `FZOO`       | `optimizers/fzoo.py`         |
| PseuZO      | nil   | `PseuZO`     | `optimizers/pseuzo.py`       |
| SubZero     | nil   | `SubZero`    | `optimizers/subzero.py`      |

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
- Must implement `.step(closure)` where `closure` follows the protocol below.

### Closure protocol

The harness gives each optimizer a single callable `closure`. Two forms:

```python
loss = closure()                           # standard ZO path (used by everyone)
loss, last_hidden, grad_h = closure(need_output=True)   # enriched path (PseuZO)
```

- **Standard form** — `closure()` returns the loss tensor. Cheap forward,
  no autograd. Used by MeZO, Sparse-MeZO, HiZOO, QuZO, LOZO, DiZO, SubZero,
  ZO-Muon, ConMeZO, FZOO.
- **Enriched form** — `closure(need_output=True)` runs the model up to the
  last hidden state with no grad, then runs `lm_head + CE loss` with grad
  enabled on JUST the hidden state. Returns `(loss, last_hidden, ∂L/∂h)`.
  Used only by PseuZO (needs the Jacobian-via-output for its pseudo-ZO
  gradient estimate). The main model parameters never see autograd.

Each call to `closure` (either form) counts as **one forward pass** for the
fairness metric `train/total_forwards` logged to WandB. Methods that do more
than 2 closures per step (FZOO ~9, ZO-Muon ~8, HiZOO 3) will show higher
`total_forwards` at the same step count — plot accuracy vs `total_forwards`
for an apples-to-apples comparison.

The class name in the YAML's `optimizer.name` field must match the class name
in the module. `train.py:OPTIMIZER_MODULES` maps class name → module file —
add an entry there when you add a new method.
