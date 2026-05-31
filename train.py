"""Unified ZO comparison harness for Qwen-0.8B on SuperGLUE.

Goal: run any of MeZO / Sparse-MeZO / LOZO / LOZO-M / DiZO / HiZOO under the
SAME setup (model, dataset, prompts, eval protocol) so the team's head-to-head
numbers are apples-to-apples. The optimizer is selected via the YAML config;
the task (multirc | copa) is selected via the --task CLI flag.

Metrics logged to WandB (project `Zero-Order-Opt`):
- accuracy (logit-level — log-likelihood ranking over candidate verbalizers)
- accuracy (token-level — argmax over the gold completion tokens, MeZO-style)
- time (sec/step, avg sec/step, total steps)
- memory (current GPU allocated MB, peak GPU allocated MB)

WandB run naming convention: {owner}-{method}-{mm_dd_hh_mm_ss}
  owner is read from $RUN_OWNER (or `owner:` in the YAML). Use "maria",
  "nil", or "cheng" so the team can filter runs by author.

Usage:
  python train.py --config configs/mezo.yaml         --task multirc
  python train.py --config configs/sparse_mezo.yaml  --task copa
  RUN_OWNER=nil python train.py --config configs/hizoo.yaml --task multirc
"""
import argparse
import importlib
import math
import os
import time
from datetime import datetime, timezone

import torch
import wandb
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from tasks import load_task

# Map optimizer class name -> module filename under optimizers/.
# Add new methods here (or just edit the file at the path) — train.py will
# lazily import only the one selected by the YAML, so missing modules for
# methods you're not running don't break anything.
OPTIMIZER_MODULES = {
    "MeZO":       "mezo",
    "SparseMeZO": "sparse_mezo",
    "LOZO":       "lozo",
    "LOZOM":      "lozo",
    "DiZO":       "dizo",
    "HiZOO":      "hizoo",
}


def load_optimizer_cls(name: str):
    if name not in OPTIMIZER_MODULES:
        raise ValueError(f"Unknown optimizer {name!r}; registered: {list(OPTIMIZER_MODULES)}")
    mod = importlib.import_module(f"optimizers.{OPTIMIZER_MODULES[name]}")
    if not hasattr(mod, name):
        raise ImportError(
            f"optimizers/{OPTIMIZER_MODULES[name]}.py is missing class `{name}`. "
            "Implement it first (see optimizers/__init__.py for the convention)."
        )
    return getattr(mod, name)


# --------------------------------------------------------------------------
# CLI + config
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to optimizer YAML")
    p.add_argument("--task", required=True, choices=["multirc", "copa"],
                   help="SuperGLUE task to train / evaluate on")
    p.add_argument("--owner", default=None,
                   help="Run owner: maria | nil | cheng. Falls back to $RUN_OWNER or yaml owner.")
    return p.parse_args()


# --------------------------------------------------------------------------
# Data collation (right-pad to the longest sequence in the batch)
# --------------------------------------------------------------------------
def pad_collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(   b["labels"]    + [-100]   * n)
        attn.append(     b["attention_mask"] + [0] * n)
    return {
        "input_ids":      torch.tensor(input_ids,  dtype=torch.long),
        "labels":         torch.tensor(labels,     dtype=torch.long),
        "attention_mask": torch.tensor(attn,       dtype=torch.long),
    }


# --------------------------------------------------------------------------
# Evaluation: both token-level and logit-level (candidate LL ranking)
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, tokenizer, val_examples, spec, device):
    model.eval()
    # ---- token-level accuracy (argmax over gold completion tokens) ----
    token_correct = token_total = 0
    # ---- logit-level accuracy (rank candidate completions by total LL) ----
    logit_correct = 0

    for ex in val_examples:
        # Token-level: format like training, score gold completion only.
        train_pack = spec.format_train(ex, tokenizer)
        ids    = torch.tensor([train_pack["input_ids"]],   device=device)
        labels = torch.tensor([train_pack["labels"]],      device=device)
        out = model(input_ids=ids, labels=labels)
        # Shift so logit_t predicts token_{t+1}
        shift_logits = out.logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        if mask.any():
            preds = shift_logits.argmax(dim=-1)
            token_correct += (preds[mask] == shift_labels[mask]).sum().item()
            token_total   += mask.sum().item()

        # Logit-level: score each candidate's completion, pick argmax.
        eval_pack = spec.format_eval(ex, tokenizer)
        prompt_ids = eval_pack["prompt_ids"]
        best_ll, best_idx = -float("inf"), -1
        for cand_idx, cand_ids in enumerate(eval_pack["candidates"]):
            full = prompt_ids + cand_ids
            inp  = torch.tensor([full], device=device)
            logits = model(input_ids=inp).logits[0]  # [seq, vocab]
            # LL of cand token c_i is log_softmax(logits[len(prompt) + i - 1])[c_i]
            log_probs = torch.log_softmax(logits, dim=-1)
            ll = 0.0
            for k, tok_id in enumerate(cand_ids):
                ll += log_probs[len(prompt_ids) + k - 1, tok_id].item()
            ll /= max(1, len(cand_ids))  # length-normalized — fairer for unequal-length candidates (COPA)
            if ll > best_ll:
                best_ll, best_idx = ll, cand_idx
        if best_idx == eval_pack["gold_idx"]:
            logit_correct += 1

    n_eval = len(val_examples)
    return {
        "eval/token_accuracy": (token_correct / token_total) if token_total else 0.0,
        "eval/logit_accuracy": logit_correct / n_eval,
        "eval/num_examples":   n_eval,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Run owner — CLI > env > yaml
    owner = args.owner or os.environ.get("RUN_OWNER") or cfg.get("owner")
    if owner not in {"maria", "nil", "cheng"}:
        raise ValueError(f"--owner / RUN_OWNER / config 'owner' must be one of maria|nil|cheng (got {owner!r})")

    opt_name   = cfg["optimizer"]["name"]
    opt_kwargs = cfg["optimizer"].get("kwargs", {}) or {}
    opt_cls    = load_optimizer_cls(opt_name)

    seed       = cfg.get("training", {}).get("seed", 42)
    batch_size = cfg.get("training", {}).get("batch_size", 16)
    max_steps  = cfg.get("training", {}).get("max_steps", 20000)
    eval_steps = cfg.get("training", {}).get("eval_steps", 500)
    num_train  = cfg.get("training", {}).get("num_train", 1000)
    num_eval   = cfg.get("training", {}).get("num_eval", 1000)
    model_name = cfg.get("model", {}).get("name", "Qwen/Qwen3.5-0.8B")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- WandB run name: {owner}-{method}-{mm_dd_hh_mm_ss} (UTC) ----
    stamp = datetime.now(timezone.utc).strftime("%m_%d_%H_%M_%S")
    run_name = f"{owner}-{opt_name}-{stamp}"
    wandb.init(
        project="Zero-Order-Opt",
        name=run_name,
        config={**cfg, "task": args.task, "owner": owner, "_resolved_seed": seed},
    )

    # ---- Load task data ----
    spec, ds = load_task(args.task, num_train=num_train, seed=seed)
    val_split = "validation" if "validation" in ds else "test"
    val_examples = list(ds[val_split])
    if num_eval and len(val_examples) > num_eval:
        val_examples = val_examples[:num_eval]
    print(f"[Data] task={args.task}  train={len(ds['train'])}  eval={len(val_examples)}")

    # ---- Load tokenizer + model (fp16 for H100 throughput; ZO does no backprop) ----
    print(f"[Model] loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float16,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    # MeZO-family expects eval-mode forward (no dropout noise in the gradient estimate)
    model.eval()

    # Param-id injection so distributed ZO RNG stays in lock-step (LOZO/HiZOO need this)
    for i, (name, p) in enumerate(model.named_parameters()):
        if p.requires_grad:
            p.param_id = i

    # ---- Format train split lazily into tokenized examples ----
    train_packs = [spec.format_train(ex, tokenizer) for ex in ds["train"]]
    train_loader = DataLoader(
        train_packs, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: pad_collate(b, tokenizer.pad_token_id),
    )

    # ---- Optimizer ----
    optimizer = opt_cls(model.parameters(), **opt_kwargs)
    print(f"[Opt] {opt_name}({opt_kwargs})")

    # ---- Training loop ----
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    step_times = []
    global_step = 0
    train_iter = iter(train_loader)

    while global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        step_loss = {"v": None}
        def closure():
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            step_loss["v"] = out.loss.detach()
            return out.loss

        t0 = time.perf_counter()
        optimizer.step(closure)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        global_step += 1

        log = {
            "train/loss":          float(step_loss["v"].item()) if step_loss["v"] is not None else 0.0,
            "train/step_time_sec": dt,
            "train/avg_step_time_sec": sum(step_times) / len(step_times),
            "train/step":          global_step,
        }
        if torch.cuda.is_available():
            log["mem/current_MB"] = torch.cuda.memory_allocated() / 1024**2
            log["mem/peak_MB"]    = torch.cuda.max_memory_allocated() / 1024**2
        wandb.log(log, step=global_step)

        if global_step % 50 == 0:
            print(f"  step {global_step:5d}  loss={log['train/loss']:.4f}  "
                  f"sec/step={dt:.3f}  peak_mem={log.get('mem/peak_MB', 0):.0f}MB")

        # ---- Periodic eval ----
        if global_step % eval_steps == 0 or global_step == max_steps:
            ev = evaluate(model, tokenizer, val_examples, spec, device)
            ev["train/step"] = global_step
            wandb.log(ev, step=global_step)
            print(f"  [eval @ step {global_step}] "
                  f"logit_acc={ev['eval/logit_accuracy']:.4f}  "
                  f"token_acc={ev['eval/token_accuracy']:.4f}")
            model.eval()  # paranoia — keep eval mode for the next ZO step

    # ---- Final summary ----
    summary = {
        "final/total_steps":       global_step,
        "final/avg_step_time_sec": sum(step_times) / len(step_times),
        "final/total_time_sec":    sum(step_times),
    }
    if torch.cuda.is_available():
        summary["final/peak_mem_MB"] = torch.cuda.max_memory_allocated() / 1024**2
    wandb.log(summary, step=global_step)
    for k, v in summary.items():
        wandb.run.summary[k] = v
    print(f"[Done] {summary}")
    wandb.finish()


if __name__ == "__main__":
    main()
