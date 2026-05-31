"""Unified ZO comparison harness for Qwen-0.8B on SuperGLUE.

Goal: run any of MeZO / Sparse-MeZO / LOZO / DiZO / HiZOO / QuZO / ZO-Muon /
ConMeZO / FZOO / PseuZO / SubZero under the SAME setup (model, dataset,
prompts, eval protocol) so the team's head-to-head numbers are apples-to-apples.
The optimizer is selected via the YAML config; the task (multirc | copa) is
selected via the --task CLI flag.

Metrics logged to WandB (project `Zero-Order-Opt`):
  train/        loss, step_time_sec, avg_step_time_sec, step, epoch,
                datapoints_seen, forwards_this_step, total_forwards
  mem/          current_MB, peak_MB
  eval/         token_accuracy, logit_accuracy, num_examples
  best_eval/    logit_accuracy, token_accuracy, at_step, at_forwards, at_time_sec
  opt/<key>     any per-step diagnostics the optimizer exposes via
                `optimizer.last_metrics` (a dict). Optional.
  final/        total_steps, total_forwards, avg_step_time_sec, total_time_sec,
                peak_mem_MB, plus duplicates of the best_eval/* entries.

Each run is tagged with [owner, method, task] and grouped by task so the
WandB dashboard auto-organises by-task.

WandB run name: {owner}-{method}-{mm_dd_hh_mm_ss}  (UTC)
  owner ∈ {maria, nil, cheng}, read from --owner, $RUN_OWNER, or `owner:`
  in the YAML.

Usage:
  python train.py --config configs/mezo.yaml         --task multirc
  python train.py --config configs/sparse_mezo.yaml  --task copa
  RUN_OWNER=nil python train.py --config configs/lozo.yaml --task multirc
"""
import argparse
import importlib
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from tasks import load_task

# Map optimizer class name -> module filename under optimizers/.
# train.py lazily imports only the one selected by the YAML, so missing
# modules for methods you're not running don't break anything.
OPTIMIZER_MODULES = {
    "MeZO":       "mezo",         # Cheng
    "SparseMeZO": "sparse_mezo",  # Maria
    "HiZOO":      "hizoo",        # Maria
    "QuZO":       "quzo",         # Maria
    "LOZO":       "lozo",         # Nil
    "DiZO":       "dizo",         # Cheng
    "ZOMuon":     "zo_muon",      # Cheng
    "ConMeZO":    "conmezo",      # Cheng
    "FZOO":       "fzoo",         # Cheng
    "PseuZO":     "pseuzo",       # Nil
    "SubZero":    "subzero",      # Nil
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
# CLI + collator
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to optimizer YAML")
    p.add_argument("--task", required=True, choices=["multirc", "copa"],
                   help="SuperGLUE task to train / evaluate on")
    p.add_argument("--owner", default=None,
                   help="Run owner: maria | nil | cheng (falls back to $RUN_OWNER or yaml owner)")
    p.add_argument("--eval-batch-size", type=int, default=8,
                   help="Batch size for the eval forward (eval is bottleneck if =1)")
    p.add_argument("--ckpt-dir", default="checkpoints",
                   help="Root dir for best/last checkpoints (one subdir per run)")
    return p.parse_args()


def pad_collate(batch, pad_id):
    """Right-pad input_ids to the longest sequence in the batch."""
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(   b["labels"]    + [-100]   * n)
        attn.append(     b["attention_mask"] + [0] * n)
    return {
        "input_ids":      torch.tensor(input_ids, dtype=torch.long),
        "labels":         torch.tensor(labels,    dtype=torch.long),
        "attention_mask": torch.tensor(attn,      dtype=torch.long),
    }


# --------------------------------------------------------------------------
# Evaluation: both token-level (argmax) and logit-level (LL ranking),
# batched to keep eval overhead from dominating the run on H100.
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, tokenizer, val_examples, spec, device, eval_batch_size=8):
    model.eval()
    pad_id = tokenizer.pad_token_id

    # -------- token-level: standard cross-entropy over gold completion --------
    train_packs = [spec.format_train(ex, tokenizer) for ex in val_examples]
    token_correct = token_total = 0
    for start in range(0, len(train_packs), eval_batch_size):
        chunk = train_packs[start:start + eval_batch_size]
        b = pad_collate(chunk, pad_id)
        b = {k: v.to(device) for k, v in b.items()}
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                    labels=b["labels"])
        shift_logits = out.logits[:, :-1, :]
        shift_labels = b["labels"][:, 1:]
        mask = shift_labels != -100
        if mask.any():
            preds = shift_logits.argmax(dim=-1)
            token_correct += (preds[mask] == shift_labels[mask]).sum().item()
            token_total   += mask.sum().item()

    # -------- logit-level: rank candidates by length-normalized LL ----------
    eval_packs = [spec.format_eval(ex, tokenizer) for ex in val_examples]
    # Flatten to per-(example, candidate) rows for batching.
    rows = []  # (example_idx, cand_idx, full_ids, prompt_len, cand_len)
    for ex_i, ep in enumerate(eval_packs):
        for c_idx, cand_ids in enumerate(ep["candidates"]):
            full = ep["prompt_ids"] + cand_ids
            rows.append((ex_i, c_idx, full, len(ep["prompt_ids"]), len(cand_ids)))

    max_cands = max(len(ep["candidates"]) for ep in eval_packs)
    cand_lls  = [[None] * max_cands for _ in val_examples]

    for start in range(0, len(rows), eval_batch_size):
        chunk   = rows[start:start + eval_batch_size]
        max_len = max(len(r[2]) for r in chunk)
        ids_list, attn_list = [], []
        for _, _, full, _, _ in chunk:
            pad_n = max_len - len(full)
            ids_list.append(full + [pad_id] * pad_n)
            attn_list.append([1] * len(full) + [0] * pad_n)
        ids  = torch.tensor(ids_list,  dtype=torch.long, device=device)
        attn = torch.tensor(attn_list, dtype=torch.long, device=device)
        logits    = model(input_ids=ids, attention_mask=attn).logits  # [B, L, V]
        log_probs = torch.log_softmax(logits, dim=-1)
        for row_i, (ex_i, c_idx, full, prompt_len, cand_len) in enumerate(chunk):
            # LL of cand token at full[prompt_len + k] is log_probs[row, prompt_len + k - 1, tok]
            ll = 0.0
            for k in range(cand_len):
                tok_id = full[prompt_len + k]
                ll += log_probs[row_i, prompt_len + k - 1, tok_id].item()
            ll /= max(1, cand_len)   # length-normalize: COPA candidates have unequal lengths
            cand_lls[ex_i][c_idx] = ll

    logit_correct = 0
    for ex_i, ep in enumerate(eval_packs):
        n_c    = len(ep["candidates"])
        scores = [cand_lls[ex_i][c] for c in range(n_c)]
        pred   = max(range(n_c), key=lambda c: scores[c])
        if pred == ep["gold_idx"]:
            logit_correct += 1

    return {
        "eval/token_accuracy": (token_correct / token_total) if token_total else 0.0,
        "eval/logit_accuracy": logit_correct / len(val_examples),
        "eval/num_examples":   len(val_examples),
    }


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------
def save_checkpoint(model, tokenizer, dest_dir: Path, meta: dict):
    """Save model + tokenizer + a small training_meta.json under dest_dir."""
    import json
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(dest_dir)
        tokenizer.save_pretrained(dest_dir)
    except Exception as e:
        # 8-bit/quantized models can have surprises here; don't kill the run.
        print(f"[ckpt] save_pretrained failed at {dest_dir}: {e}")
    with open(dest_dir / "training_meta.json", "w") as fp:
        json.dump(meta, fp, indent=2, default=str)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    owner = args.owner or os.environ.get("RUN_OWNER") or cfg.get("owner")
    if owner not in {"maria", "nil", "cheng"}:
        raise ValueError(
            f"--owner / RUN_OWNER / config 'owner' must be one of maria|nil|cheng (got {owner!r})"
        )

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

    # ---- WandB: run name + grouping + tags --------------------------------
    stamp    = datetime.now(timezone.utc).strftime("%m_%d_%H_%M_%S")
    run_name = f"{owner}-{opt_name}-{stamp}"
    wandb.init(
        project="Zero-Order-Opt",
        name=run_name,
        group=args.task,                             # group all multirc / all copa together
        tags=[owner, opt_name, args.task],           # filter in the dashboard
        config={**cfg, "task": args.task, "owner": owner, "_resolved_seed": seed},
    )

    # ---- Load task data ---------------------------------------------------
    spec, ds = load_task(args.task, num_train=num_train, seed=seed)
    val_split = "validation" if "validation" in ds else "test"
    val_examples = list(ds[val_split])
    if num_eval and len(val_examples) > num_eval:
        val_examples = val_examples[:num_eval]
    print(f"[Data] task={args.task}  train={len(ds['train'])}  eval={len(val_examples)}")

    # ---- Load tokenizer + model ------------------------------------------
    print(f"[Model] loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.get("model", {}).get("load_in_8bit", False):
        from transformers import BitsAndBytesConfig
        print("[Model] using load_in_8bit (bitsandbytes)")
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch.float16,
        ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()  # ZO assumes no dropout noise in the gradient estimate

    # Param-id injection so distributed ZO RNG stays in lock-step
    for i, (name, p) in enumerate(model.named_parameters()):
        if p.requires_grad:
            p.param_id = i

    # ---- Tokenize train + build loader -----------------------------------
    train_packs = [spec.format_train(ex, tokenizer) for ex in ds["train"]]
    train_loader = DataLoader(
        train_packs, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: pad_collate(b, tokenizer.pad_token_id),
    )

    # ---- Optimizer --------------------------------------------------------
    optimizer = opt_cls(model.parameters(), **opt_kwargs)
    print(f"[Opt] {opt_name}({opt_kwargs})")

    # ---- Budget bookkeeping ----------------------------------------------
    steps_per_epoch = max(1, math.ceil(len(train_packs) / batch_size))
    total_epochs    = max_steps / steps_per_epoch
    print(f"[Budget] num_train={len(train_packs)}  batch_size={batch_size}  "
          f"steps_per_epoch={steps_per_epoch}  max_steps={max_steps}  "
          f"≈ {total_epochs:.1f} epochs (~{total_epochs:.0f} passes per datapoint)")
    wandb.run.summary["steps_per_epoch"]      = steps_per_epoch
    wandb.run.summary["planned_epochs"]       = total_epochs
    wandb.run.summary["passes_per_datapoint"] = total_epochs

    # ---- Checkpoint dirs --------------------------------------------------
    run_ckpt_dir = Path(args.ckpt_dir) / run_name
    best_dir     = run_ckpt_dir / "best"
    last_dir     = run_ckpt_dir / "last"
    print(f"[Ckpt] best -> {best_dir}   last -> {last_dir}")

    # ---- Training loop ----------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    step_times          = []
    global_step         = 0
    total_forwards      = 0  # cumulative closure calls across all completed steps
    train_iter          = iter(train_loader)
    nan_seen            = False
    run_start_time      = time.perf_counter()

    # Best-eval tracker: which step / forwards / wall-time gave the highest
    # logit_accuracy seen so far.
    best = {
        "logit_accuracy": -1.0,
        "token_accuracy": 0.0,
        "at_step":        0,
        "at_forwards":    0,
        "at_time_sec":    0.0,
    }

    last_eval = None  # final eval metrics (for the [Done] summary)

    while global_step < max_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = {k: v.to(device) for k, v in batch.items()}

        step_loss     = {"v": None}
        step_forwards = {"n": 0}

        # Closure protocol:
        #   closure()                       → returns loss tensor (cheap, no-grad)
        #   closure(need_output=True)       → returns (loss, last_hidden, ∂L/∂last_hidden)
        # The second form is what PseuZO needs (Jacobian-via-output). All other
        # methods just call the first form. Each call increments the forward
        # counter so we can plot apples-to-apples accuracy-vs-forwards.
        def closure(need_output: bool = False):
            step_forwards["n"] += 1
            if not need_output:
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                step_loss["v"] = out.loss.detach()
                return out.loss
            # Enriched path: cheap forward up to last hidden state, then a
            # tiny lm_head + CE with grad on JUST the hidden state. Main
            # model parameters never see autograd.
            with torch.no_grad():
                base_out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                )
            last_hidden = base_out.hidden_states[-1].detach().requires_grad_(True)
            lm_head = getattr(model, "lm_head", None) or model.get_output_embeddings()
            with torch.enable_grad():
                logits = lm_head(last_hidden)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                loss_t = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                loss_t.backward()
            step_loss["v"] = loss_t.detach()
            return loss_t.detach(), last_hidden.detach(), last_hidden.grad.detach()

        t0 = time.perf_counter()
        optimizer.step(closure)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        global_step    += 1
        total_forwards += step_forwards["n"]

        loss_val = float(step_loss["v"].item()) if step_loss["v"] is not None else 0.0

        # ---- NaN / inf guard: ZO often dies silently on bad LR ----
        if not math.isfinite(loss_val):
            nan_seen = True
            print(f"[NaN] non-finite loss at step {global_step} (loss={loss_val}). Aborting.")
            wandb.log({"train/nan_step": global_step, "train/loss_nonfinite": loss_val},
                      step=global_step)
            break

        log = {
            "train/loss":               loss_val,
            "train/step_time_sec":      dt,
            "train/avg_step_time_sec":  sum(step_times) / len(step_times),
            "train/step":               global_step,
            "train/epoch":              global_step / steps_per_epoch,
            "train/datapoints_seen":    global_step * batch_size,
            "train/forwards_this_step": step_forwards["n"],
            "train/total_forwards":     total_forwards,
        }
        if torch.cuda.is_available():
            log["mem/current_MB"] = torch.cuda.memory_allocated()     / 1024**2
            log["mem/peak_MB"]    = torch.cuda.max_memory_allocated() / 1024**2

        # ---- Drain optimizer diagnostics (opt-in) ----
        opt_metrics = getattr(optimizer, "last_metrics", None)
        if isinstance(opt_metrics, dict):
            for k, v in opt_metrics.items():
                # accept python scalars and 0-d tensors
                if hasattr(v, "item"):
                    v = v.item()
                log[f"opt/{k}"] = v

        wandb.log(log, step=global_step)

        if global_step % 50 == 0:
            print(f"  step {global_step:5d}  loss={loss_val:.4f}  "
                  f"sec/step={dt:.3f}  fwds/step={step_forwards['n']}  "
                  f"peak_mem={log.get('mem/peak_MB', 0):.0f}MB")

        # ---- Periodic eval ----
        if global_step % eval_steps == 0 or global_step == max_steps:
            ev = evaluate(model, tokenizer, val_examples, spec, device,
                          eval_batch_size=args.eval_batch_size)
            ev["train/step"] = global_step
            wandb.log(ev, step=global_step)
            last_eval = ev
            print(f"  [eval @ step {global_step}] "
                  f"logit_acc={ev['eval/logit_accuracy']:.4f}  "
                  f"token_acc={ev['eval/token_accuracy']:.4f}")

            # Best tracker + best ckpt save on improvement
            if ev["eval/logit_accuracy"] > best["logit_accuracy"]:
                best["logit_accuracy"] = ev["eval/logit_accuracy"]
                best["token_accuracy"] = ev["eval/token_accuracy"]
                best["at_step"]        = global_step
                best["at_forwards"]    = total_forwards
                best["at_time_sec"]    = time.perf_counter() - run_start_time
                meta = {**best, "run_name": run_name, "task": args.task,
                        "opt_name": opt_name, "owner": owner}
                save_checkpoint(model, tokenizer, best_dir, meta)
                wandb.log({
                    "best_eval/logit_accuracy": best["logit_accuracy"],
                    "best_eval/token_accuracy": best["token_accuracy"],
                    "best_eval/at_step":        best["at_step"],
                    "best_eval/at_forwards":    best["at_forwards"],
                    "best_eval/at_time_sec":    best["at_time_sec"],
                }, step=global_step)
                print(f"  [best] new best logit_acc={best['logit_accuracy']:.4f} "
                      f"(saved -> {best_dir})")

            model.eval()   # keep eval mode for the next ZO step

    # ---- Save LAST checkpoint -----------------------------------------------
    final_meta = {
        "total_steps":    global_step,
        "total_forwards": total_forwards,
        "wall_time_sec":  time.perf_counter() - run_start_time,
        "nan_aborted":    nan_seen,
        "run_name":       run_name,
        "task":           args.task,
        "opt_name":       opt_name,
        "owner":          owner,
    }
    save_checkpoint(model, tokenizer, last_dir, final_meta)

    # ---- Final summary ------------------------------------------------------
    total_time = time.perf_counter() - run_start_time
    summary = {
        "final/total_steps":       global_step,
        "final/total_forwards":    total_forwards,
        "final/avg_step_time_sec": (sum(step_times) / len(step_times)) if step_times else 0.0,
        "final/total_time_sec":    total_time,
        "final/nan_aborted":       nan_seen,
    }
    if torch.cuda.is_available():
        summary["final/peak_mem_MB"] = torch.cuda.max_memory_allocated() / 1024**2
    if last_eval is not None:
        summary["final/eval_logit_accuracy"] = last_eval["eval/logit_accuracy"]
        summary["final/eval_token_accuracy"] = last_eval["eval/token_accuracy"]
    # Mirror best/* into final summary so it's visible in the WandB run table
    summary["final/best_logit_accuracy"] = best["logit_accuracy"]
    summary["final/best_token_accuracy"] = best["token_accuracy"]
    summary["final/best_at_step"]        = best["at_step"]
    summary["final/best_at_forwards"]    = best["at_forwards"]
    summary["final/best_at_time_sec"]    = best["at_time_sec"]

    wandb.log(summary, step=global_step)
    for k, v in summary.items():
        wandb.run.summary[k] = v

    print()
    print("=" * 72)
    print(f"[Done] {run_name}")
    print(f"  BEST   logit_acc={best['logit_accuracy']:.4f}  "
          f"token_acc={best['token_accuracy']:.4f}")
    print(f"         reached at step {best['at_step']} / "
          f"forwards {best['at_forwards']} / wall {best['at_time_sec']:.1f}s")
    if last_eval is not None:
        print(f"  FINAL  logit_acc={last_eval['eval/logit_accuracy']:.4f}  "
              f"token_acc={last_eval['eval/token_accuracy']:.4f}")
    print(f"  TOTAL  {global_step} steps / {total_forwards} forwards / "
          f"{total_time:.1f}s wall / peak {summary.get('final/peak_mem_MB', 0):.0f}MB")
    if nan_seen:
        print("  WARN   run aborted on non-finite loss.")
    print(f"  CKPTS  best -> {best_dir}")
    print(f"         last -> {last_dir}")
    print("=" * 72)

    wandb.finish()


if __name__ == "__main__":
    main()
