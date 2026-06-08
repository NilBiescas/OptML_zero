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

WandB run name: {owner}-{method}-{task}-{mm_dd_hh_mm_ss}  (UTC)
  owner ∈ {maria, nil, cheng}, read from --owner, $RUN_OWNER, or `owner:`
  in the YAML.

Usage:
  python train.py --config configs/mezo.yaml         --task multirc
  python train.py --config configs/sparse_mezo.yaml  --task copa
  RUN_OWNER=nil python train.py --config configs/lozo.yaml --task multirc
"""
import argparse
import importlib
import json
import math
import os
import shutil
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

DTYPE_ALIASES = {
    "float16":  torch.float16,
    "fp16":     torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16":     torch.bfloat16,
    "float32":  torch.float32,
    "fp32":     torch.float32,
}

# HF Hub username per team-owner. Used to derive a default `hub.repo_id` when
# push_to_hub is on but the YAML doesn't specify one. Each person pushes to
# their own repo so no cross-account collaborator setup is required.
OWNER_HF_HANDLE = {
    "maria": "mpilligua",
    "nil":   "NilBiescas",
    "cheng": "chenghengli",
}

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
    p.add_argument("--model", default=None,
                   help="Override model.name from the YAML (e.g. Qwen/Qwen2.5-0.5B). "
                        "A short model tag is threaded into the run name + checkpoint key "
                        "so multiple models never collide.")
    p.add_argument("--eval-batch-size", type=int, default=8,
                   help="Batch size for the eval forward (eval is bottleneck if =1)")
    p.add_argument("--ckpt-dir", default="checkpoints",
                   help="Root dir for best/last checkpoints (one subdir per run)")
    p.add_argument("--resume-from", default=None,
                   help="Path to a prior run's `last/` dir to resume from "
                        "(reuses model weights + optimizer state + step/forward/best counters)")
    p.add_argument("--lr", type=float, default=None,
                   help="Override optimizer.kwargs.lr from the YAML (e.g. a smaller "
                        "task-specific lr for COPA). None = use the YAML value.")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Override training.max_steps from the YAML (e.g. fewer steps "
                        "for the tiny 400-example COPA set). None = use the YAML value.")
    p.add_argument("--eval-steps", type=int, default=None,
                   help="Override training.eval_steps (e.g. 50 for a short 500-step "
                        "COPA run, to capture the early peak finely). None = YAML value.")
    p.add_argument("--run-suffix", default=None,
                   help="Suffix appended to the run key (e.g. 'short') so this run is a "
                        "DISTINCT wandb run + checkpoint dir, coexisting with the default "
                        "long run instead of resuming it. Also added as a tag.")
    p.add_argument("--set", action="append", default=None, metavar="KEY=VALUE",
                   help="Override an optimizer.kwargs entry (repeatable), e.g. "
                        "--set cone_warmup_total=4000 --set refresh_T=100. Values are "
                        "parsed as bool/int/float when possible, else kept as strings.")
    p.add_argument("--base-eval", action="store_true",
                   help="Benchmark the UNTRAINED base model: load it, run a single "
                        "eval on the task, log to WandB as '<owner>-base-<task>', exit. "
                        "No training, no optimizer. Gives the zero-shot baseline.")
    p.add_argument("--lr-scheduler", default=None,
                   choices=["constant", "warmup_constant", "cosine", "linear"],
                   help="LR schedule (overrides training.lr_scheduler). The ZO papers "
                        "use constant LR; warmup_constant/cosine/linear add a linear "
                        "warmup + optional decay. Default constant.")
    p.add_argument("--warmup-ratio", type=float, default=None,
                   help="Fraction of max_steps for linear LR warmup (overrides "
                        "training.warmup_ratio; default 0).")
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
def save_checkpoint(model, tokenizer, optimizer, dest_dir: Path,
                    meta: dict, source_cfg_path: str | None = None):
    """Persist model + tokenizer + optimizer state + meta + resolved YAML.

    The YAML copy (+ git SHA in meta) makes any saved checkpoint
    self-describing — combined with the optimizer state pickle, a run can be
    resumed from disk with `--resume-from <this_dir>`.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(dest_dir)
        tokenizer.save_pretrained(dest_dir)
    except Exception as e:
        # 8-bit / quantized models can have surprises here; don't kill the run.
        print(f"[ckpt] save_pretrained failed at {dest_dir}: {e}")
    try:
        # Atomic: write to a temp file then rename, so a preemption mid-write
        # never leaves a truncated optimizer.pt.
        _opt_tmp = dest_dir / "optimizer.pt.tmp"
        torch.save(optimizer.state_dict(), _opt_tmp)
        os.replace(_opt_tmp, dest_dir / "optimizer.pt")
    except Exception as e:
        # Some custom optimizer states (closures, non-tensor objects) don't
        # pickle cleanly — record the failure but keep the rest of the ckpt.
        print(f"[ckpt] optimizer.state_dict() save failed at {dest_dir}: {e}")
    if source_cfg_path:
        try:
            shutil.copy(source_cfg_path, dest_dir / "config.yaml")
        except Exception as e:
            print(f"[ckpt] config.yaml copy failed at {dest_dir}: {e}")
    # training_meta.json is the resume GATE: it is written LAST and ATOMICALLY
    # (temp file -> fsync -> os.replace). A preemption that interrupts any
    # earlier write leaves the *previous* valid meta untouched; a preemption
    # during this write leaves either the old file or the new one, never a
    # truncated 0-byte file. This is what prevents the JSONDecodeError
    # crash-loop on auto-resume.
    _meta_tmp = dest_dir / "training_meta.json.tmp"
    with open(_meta_tmp, "w") as fp:
        json.dump(meta, fp, indent=2, default=str)
        fp.flush()
        os.fsync(fp.fileno())
    os.replace(_meta_tmp, dest_dir / "training_meta.json")


def _read_meta(meta_path: Path):
    """Load a training_meta.json, returning None if it is missing, empty, or
    corrupt (e.g. a write truncated by a preemption or a full disk). NEVER
    raises — a bad checkpoint meta must not crash the relaunch; the caller
    falls back to another checkpoint or starts fresh."""
    try:
        if not meta_path.exists() or meta_path.stat().st_size == 0:
            return None
        with open(meta_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def _lr_factor(step: int, max_steps: int, sched: str, warmup_steps: int) -> float:
    """Multiplicative LR factor at `step` (0-indexed) for a warmup + decay
    schedule. Linear warmup to 1.0 over warmup_steps, then constant / cosine /
    linear decay. The ZO optimizers read group['lr'] each step, so scaling it
    here schedules every method uniformly."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if sched in ("constant", "warmup_constant"):
        return 1.0
    denom = max(1, max_steps - warmup_steps)
    prog  = min(1.0, max(0.0, (step - warmup_steps) / denom))
    if sched == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * prog))
    if sched == "linear":
        return max(0.0, 1.0 - prog)
    return 1.0


def _git_sha() -> str | None:
    """Best-effort git SHA of the repo train.py lives in (None if not a repo)."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return None


def _hub_push_folder(folder: Path, repo_id: str, sub_path: str, commit_message: str):
    """Best-effort push to HF Hub. Soft-fails on any error."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, exist_ok=True)
        api.upload_folder(
            folder_path=str(folder),
            repo_id=repo_id,
            path_in_repo=sub_path,
            commit_message=commit_message,
        )
        return True
    except Exception as e:
        print(f"[hub] push to {repo_id}/{sub_path} failed: {e}")
        return False


def _assert_hub_writable(repo_id: str) -> None:
    """Fail fast at startup if the HF token can't push to `repo_id`.

    `push_to_hub` jobs are typically multi-hour runs; we'd rather error in the
    first second than after 4h of compute. Tries `create_repo(exist_ok=True)`
    which exercises the same auth path the final push will use.
    """
    if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        raise RuntimeError(
            "hub.push_to_hub: true but no HF_TOKEN / HUGGINGFACE_HUB_TOKEN in env. "
            "Export a token with write scope on the target repo before launching."
        )
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        who = api.whoami()
        api.create_repo(repo_id=repo_id, exist_ok=True)
        print(f"[hub] verified write access to {repo_id} as user={who.get('name')!r}")
    except Exception as e:
        raise RuntimeError(
            f"hub.push_to_hub: true but cannot write to {repo_id!r}: {e}\n"
            "Check that HF_TOKEN has write scope and that you have access to the repo."
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    owner = args.owner or os.environ.get("RUN_OWNER") or cfg.get("owner")
    if owner not in {"maria", "nil", "cheng", "chengheng"}:
        raise ValueError(
            f"--owner / RUN_OWNER / config 'owner' must be one of "
            f"maria|nil|cheng|chengheng (got {owner!r})"
        )

    _extra_tags = [t for t in os.environ.get("RUN_TAGS", "").split(",") if t.strip()]
    opt_name   = cfg["optimizer"]["name"]
    opt_kwargs = cfg["optimizer"].get("kwargs", {}) or {}
    # Base-model benchmark: name the run "<owner>-base-<task>" and skip the
    # optimizer entirely (just eval the untrained model once).
    if args.base_eval:
        opt_name = "base"
    # CLI lr override (task-specific tuning, e.g. a smaller COPA lr) — applied
    # before the optimizer is built and logged into the WandB config below.
    if args.lr is not None:
        print(f"[lr-override] {opt_name} lr {opt_kwargs.get('lr')} -> {args.lr}")
        opt_kwargs["lr"] = args.lr
    # Generic per-run kwargs overrides (repeatable --set key=value), for
    # task-specific tuning of method kwargs (e.g. COPA: cone_warmup_total to
    # match a shorter max_steps, refresh_T for ZO-Muon).
    for _kv in (args.set or []):
        _k, _, _vs = _kv.partition("=")
        _k, _vs = _k.strip(), _vs.strip()
        if _vs.lower() in ("true", "false"):
            _val = _vs.lower() == "true"
        else:
            try:
                _val = int(_vs)
            except ValueError:
                try:
                    _val = float(_vs)
                except ValueError:
                    _val = _vs
        print(f"[set-override] {opt_name} {_k} {opt_kwargs.get(_k)} -> {_val!r}")
        opt_kwargs[_k] = _val
    # First-order baselines (AdamW/Adam/SGD) for reference: they use real
    # backprop, not the ZO closure. Everything else (data, eval, logging) is
    # identical, so the accuracy is directly comparable to the ZO methods.
    _FIRST_ORDER = {"AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam,
                    "SGD": torch.optim.SGD}
    is_first_order = opt_name in _FIRST_ORDER
    opt_cls = None if (is_first_order or args.base_eval) else load_optimizer_cls(opt_name)

    seed       = cfg.get("training", {}).get("seed", 42)
    batch_size = cfg.get("training", {}).get("batch_size", 16)
    max_steps  = cfg.get("training", {}).get("max_steps", 20000)
    if args.max_steps is not None:
        print(f"[max-steps-override] {max_steps} -> {args.max_steps}")
        max_steps = args.max_steps
    eval_steps = cfg.get("training", {}).get("eval_steps", 500)
    if args.eval_steps is not None:
        print(f"[eval-steps-override] {eval_steps} -> {args.eval_steps}")
        eval_steps = args.eval_steps
    # ---- LR scheduler (default constant = ZO-paper convention) ------------
    lr_sched = (args.lr_scheduler
                or cfg.get("training", {}).get("lr_scheduler") or "constant")
    warmup_ratio = (args.warmup_ratio if args.warmup_ratio is not None
                    else cfg.get("training", {}).get("warmup_ratio", 0.0))
    warmup_steps = int(warmup_ratio * max_steps)
    print(f"[lr-sched] {lr_sched}  warmup_ratio={warmup_ratio}  warmup_steps={warmup_steps}")
    num_train  = cfg.get("training", {}).get("num_train", 1000)
    num_eval   = cfg.get("training", {}).get("num_eval", 1000)
    model_name = args.model or cfg.get("model", {}).get("name", "Qwen/Qwen3.5-0.8B")
    # Short tag derived from the model id (org stripped, lowercased), e.g.
    # "Qwen/Qwen3.5-0.8B" -> "qwen3.5-0.8b". Threaded into the WandB run name and
    # the stable checkpoint key so two models never collide on either.
    model_tag = model_name.split("/")[-1].lower()
    # Default dtype is bfloat16: more stable than fp16 for long-horizon ZO
    # runs (RCP1 hit fp16 underflow during Sparse-MeZO replication on WiC).
    dtype_str  = cfg.get("model", {}).get("dtype", "bfloat16")
    if dtype_str not in DTYPE_ALIASES:
        raise ValueError(f"model.dtype must be one of {list(DTYPE_ALIASES)} (got {dtype_str!r})")
    model_dtype = DTYPE_ALIASES[dtype_str]

    # ---- HF Hub push (on by default) -------------------------------------
    # Best + last ckpts are pushed once, at end of training. Each owner pushes
    # to their own repo to avoid cross-account collaborator setup. Override
    # the derived repo_id with `hub.repo_id: <handle>/<repo>` in the YAML, or
    # disable entirely with `hub.push_to_hub: false`.
    hub_cfg     = cfg.get("hub", {}) or {}
    push_to_hub = bool(hub_cfg.get("push_to_hub", True))
    hub_repo_id = hub_cfg.get("repo_id")
    if push_to_hub and not hub_repo_id:
        handle = OWNER_HF_HANDLE.get(owner)
        if not handle:
            raise ValueError(
                f"hub.push_to_hub on but no `hub.repo_id` set and no default "
                f"HF handle known for owner={owner!r}. Set hub.repo_id in the YAML "
                "or disable with hub.push_to_hub: false."
            )
        hub_repo_id = f"{handle}/zo-comparison-qwen"
        print(f"[hub] no repo_id in YAML; defaulting to {hub_repo_id}")

    # ---- Auto-resume on preemption ---------------------------------------
    # Checkpoints live at a STABLE, timestamp-free path keyed by
    # (owner, method, task) on the durable PVC. If a `last/` checkpoint already
    # exists there and the caller didn't pass --resume-from, resume from it:
    # the job picks up from the last saved step AND continues the SAME WandB
    # run (the run id is stored in training_meta.json). This makes a preempted
    # + relaunched job seamless.
    ckpt_key = f"{owner}-{opt_name}-{args.task}-{model_tag}"
    if args.run_suffix:
        ckpt_key = f"{ckpt_key}-{args.run_suffix}"   # distinct id/name/ckpt + tag
    _run_ckpt = Path(args.ckpt_dir) / ckpt_key
    if not args.resume_from:
        # Prefer last/ (latest step + optimizer). If its meta was truncated by
        # a preemption / full disk, fall back to best/. Validate the meta
        # actually PARSES before committing — a corrupt meta must NOT crash the
        # relaunch (the old bare-json.load bug); we just start fresh instead.
        for _cand in (_run_ckpt / "last", _run_ckpt / "best"):
            if _read_meta(_cand / "training_meta.json") is not None:
                args.resume_from = str(_cand)
                print(f"[Auto-resume] found valid checkpoint at {_cand}; "
                      "resuming (same step + same WandB run).")
                break
        else:
            if (_run_ckpt / "last" / "training_meta.json").exists():
                print(f"[Auto-resume] checkpoint meta under {_run_ckpt} is "
                      "corrupt/empty (preemption mid-write?); starting fresh.")

    # ---- Resume meta (loaded early so global_step / best / total_forwards
    # ----   are seeded from disk before training starts) ------------------
    resume_meta = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        meta_path   = resume_path / "training_meta.json"
        resume_meta = _read_meta(meta_path)
        if resume_meta is not None:
            print(f"[Resume] meta from {meta_path}: "
                  f"step={resume_meta.get('total_steps')} "
                  f"forwards={resume_meta.get('total_forwards')} "
                  f"best={resume_meta.get('best')}")
        else:
            print(f"[Resume] WARNING: training_meta.json missing/corrupt at "
                  f"{meta_path}; weights will load but counters start from 0.")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Fail fast on HF auth before doing anything expensive ----
    if push_to_hub:
        _assert_hub_writable(hub_repo_id)

    # ---- WandB: ONE deterministic run per (owner, method, task, model) ----
    # The run id is a stable hash of ckpt_key, so EVERY relaunch / preemption
    # resumes the SAME wandb run (resume="allow") rather than forking a fresh
    # duplicate — one continuous curve, no clutter. On the transient
    # "run ID ... is in use" lock (a just-preempted sibling hasn't released yet)
    # we WAIT and retry the same id instead of starting fresh. To begin a truly
    # NEW run (e.g. a hyperparameter change), delete the wandb run first;
    # relaunching then recreates it under the same id, fresh.
    import hashlib
    # Persist the wandb run id in the durable ckpt dir. A preempted+relaunched
    # job (valid checkpoint present) RESUMES the SAME run by reading this file;
    # a genuinely FRESH start instead mints a SALTED id. This dodges the
    # "phantom lock" failure mode: when an init times out server-side it leaves
    # a half-created run that is un-findable via the API yet permanently blocks
    # the plain deterministic id with "run ID is in use". Salting per fresh
    # start means a new run never inherits another run's poisoned id.
    _id_file  = _run_ckpt / "wandb_id"
    if args.resume_from:
        # Resuming real progress: reuse the same run id. Prefer the persisted
        # file; fall back to the legacy plain-deterministic id for checkpoints
        # written before this fix (so they keep ONE continuous curve).
        if _id_file.exists():
            stable_id = _id_file.read_text().strip()
        else:
            stable_id = "zo" + hashlib.md5(ckpt_key.encode()).hexdigest()[:14]
        print(f"[wandb] resuming run id={stable_id}")
    else:
        # Fresh start: salted id so we never inherit another run's poisoned id.
        _salt = hashlib.md5(f"{ckpt_key}-{time.time()}".encode()).hexdigest()[:6]
        stable_id = "zo" + hashlib.md5(f"{ckpt_key}-{_salt}".encode()).hexdigest()[:14]
        print(f"[wandb] fresh run id={stable_id}")
    try:
        _id_file.parent.mkdir(parents=True, exist_ok=True)
        _id_file.write_text(stable_id)
    except Exception as _e:
        print(f"[wandb] WARN: could not persist run id ({_e})")
    run_name  = ckpt_key   # includes the model tag (+ run-suffix if any)
    _wandb_base = dict(
        project="Zero-Order-Opt",
        entity="pilligua",   # team workspace — overrides any WANDB_ENTITY env
        group=args.task,     # group all multirc / all copa together
        tags=[owner, opt_name, args.task, model_tag]
             + ([args.run_suffix] if args.run_suffix else []) + _extra_tags,
        config={**cfg, "task": args.task, "owner": owner,
                "model": model_name, "model_tag": model_tag,
                "_resolved_seed": seed,
                "_resolved_dtype": dtype_str,
                "_resumed_from":   args.resume_from,
                "_hub_push":       push_to_hub,
                "_hub_repo_id":    hub_repo_id},
        settings=wandb.Settings(init_timeout=180),
    )
    run = None
    # The first online init can TIME OUT under wandb server load; that timeout
    # still registers the id server-side, so every subsequent attempt fails with
    # "run ID ... is in use" until the dead-process lock heartbeat expires
    # (~5-10 min). Retry the SAME id long enough to outlast that lock (do NOT
    # fork a fresh duplicate run). 15 attempts x 30s ~= 7.5 min covers it.
    for _attempt in range(15):
        try:
            run = wandb.init(id=stable_id, resume="allow", name=run_name, **_wandb_base)
            break
        except Exception as e:
            print(f"[wandb] init attempt {_attempt} for id={stable_id} failed "
                  f"({type(e).__name__}: {e}); retrying in 30s")
            try:
                wandb.finish(exit_code=1)
            except Exception:
                pass
            time.sleep(30)
    if run is None:
        # Last resort: NEVER let a transient wandb hiccup kill a multi-hour GPU
        # run. Fall back to OFFLINE mode (same deterministic id) so training
        # proceeds and logs to disk; the run can be `wandb sync`ed later. The
        # ckpt dir still lets it auto-resume after preemption.
        print(f"[wandb] WARNING: online init failed after 15 attempts; "
              f"falling back to OFFLINE mode for id={stable_id}")
        try:
            wandb.finish(exit_code=1)
        except Exception:
            pass
        # The submit script exports WANDB_MODE=online, and that env var OVERRIDES
        # wandb.Settings(mode="offline") -> the "offline" init would still hit the
        # server and re-raise. Force the env to offline so the fallback truly runs
        # locally (syncable later) and training proceeds.
        os.environ["WANDB_MODE"] = "offline"
        _offline = dict(_wandb_base)
        _offline["settings"] = wandb.Settings(init_timeout=180, mode="offline")
        run = wandb.init(id=stable_id, resume="allow", name=run_name, **_offline)
    print(f"[WandB] run id={stable_id} name={run_name} (resume=allow, "
          f"resumed={getattr(run, 'resumed', '?')})")

    # ---- Load task data ---------------------------------------------------
    spec, ds = load_task(args.task, num_train=num_train, seed=seed)
    val_split = "validation" if "validation" in ds else "test"
    val_examples = list(ds[val_split])
    if num_eval and len(val_examples) > num_eval:
        val_examples = val_examples[:num_eval]
    print(f"[Data] task={args.task}  train={len(ds['train'])}  eval={len(val_examples)}")

    # ---- Load tokenizer + model ------------------------------------------
    # On resume, load weights from `args.resume_from`; otherwise from HF Hub.
    weights_src = args.resume_from if args.resume_from else model_name
    print(f"[Model] loading {weights_src}  dtype={dtype_str}")
    tokenizer = AutoTokenizer.from_pretrained(weights_src, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.get("model", {}).get("load_in_8bit", False):
        from transformers import BitsAndBytesConfig
        print("[Model] using load_in_8bit (bitsandbytes)")
        model = AutoModelForCausalLM.from_pretrained(
            weights_src, trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            weights_src, trust_remote_code=True, torch_dtype=model_dtype,
        ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()  # ZO assumes no dropout noise in the gradient estimate

    # Param-id + param-name injection.
    # - param_id keeps per-param RNG seeds in lock-step across ranks (LOZO,
    #   HiZOO, ZO-Muon, etc. seed each param's generator from this).
    # - param_name lets methods that key state by parameter name (DiZO's Q/V
    #   detection, SubZero's per-layer U/V dict, PseuZO's parameter list)
    #   work with the closure-only optimizer interface — they can read
    #   `p.param_name` instead of needing `named_parameters()`.
    for i, (name, p) in enumerate(model.named_parameters()):
        if p.requires_grad:
            p.param_id   = i
            p.param_name = name

    # ---- Base-model benchmark: eval the untrained model once and exit ------
    if args.base_eval:
        print("[base-eval] evaluating the UNTRAINED base model (no training)")
        ev = evaluate(model, tokenizer, val_examples, spec, device,
                      eval_batch_size=args.eval_batch_size)
        wandb.log({**ev, "global_step": 0, "train/total_forwards": 0,
                   "train/cumulative_train_time_sec": 0.0})
        wandb.run.summary["final/eval_logit_accuracy"]  = ev["eval/logit_accuracy"]
        wandb.run.summary["final/best_logit_accuracy"]  = ev["eval/logit_accuracy"]
        wandb.run.summary["final/eval_token_accuracy"]  = ev["eval/token_accuracy"]
        wandb.run.summary["final/train_only_time_sec"]  = 0.0
        print(f"[base-eval] {args.task}: logit_acc={ev['eval/logit_accuracy']:.4f}  "
              f"token_acc={ev['eval/token_accuracy']:.4f}  n={ev['eval/num_examples']}")
        wandb.finish()
        return

    # ---- Tokenize train + build loader -----------------------------------
    train_packs = [spec.format_train(ex, tokenizer) for ex in ds["train"]]
    train_loader = DataLoader(
        train_packs, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: pad_collate(b, tokenizer.pad_token_id),
    )

    # First-order (AdamW) does REAL backprop, so it needs the full autograd
    # graph + grads + optimizer moments — unlike the forward-only ZO methods.
    # fp32 full fine-tuning of the 0.8B model at bs=16 over MultiRC's long
    # sequences OOMs an 80GB H100 without this. Gradient checkpointing trades
    # recompute for a large activation-memory saving so bs=16 fits.
    if is_first_order and hasattr(model, "gradient_checkpointing_enable"):
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        print("[first-order] gradient checkpointing enabled (fit bs=16 fp32)")

    # ---- Optimizer --------------------------------------------------------
    if is_first_order:
        optimizer = _FIRST_ORDER[opt_name](model.parameters(), **opt_kwargs)
    else:
        optimizer = opt_cls(model.parameters(), **opt_kwargs)
    print(f"[Opt] {opt_name}({opt_kwargs})  first_order={is_first_order}")
    # Base LRs captured for the scheduler (the optimizer reads group['lr'] /
    # group['lr_1d'] each step, so we rescale them in the loop).
    base_lr    = opt_kwargs.get("lr")
    base_lr_1d = opt_kwargs.get("lr_1d")

    # ---- Resume optimizer state (best-effort) ----------------------------
    if args.resume_from:
        opt_state_path = Path(args.resume_from) / "optimizer.pt"
        if opt_state_path.exists():
            try:
                optimizer.load_state_dict(torch.load(opt_state_path, weights_only=False))
                print(f"[Resume] loaded optimizer state from {opt_state_path}")
            except Exception as e:
                print(f"[Resume] could not load optimizer state ({e}); continuing fresh")
        else:
            print(f"[Resume] no optimizer.pt at {opt_state_path}; "
                  "ZO RNG seeds will re-derive from global_step")

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
    # Stable key (no timestamp) so a relaunched pod writes to / resumes from
    # the same directory on the PVC.
    run_ckpt_dir = Path(args.ckpt_dir) / ckpt_key
    best_dir     = run_ckpt_dir / "best"
    last_dir     = run_ckpt_dir / "last"
    print(f"[Ckpt] best -> {best_dir}   last -> {last_dir}")

    # ---- Training loop ----------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    step_times          = []
    train_iter          = iter(train_loader)
    nan_seen            = False
    run_start_time      = time.perf_counter()

    # Resume counters from disk if we have them.
    global_step         = int(resume_meta["total_steps"])    if resume_meta else 0
    total_forwards      = int(resume_meta["total_forwards"]) if resume_meta else 0
    best = {
        "logit_accuracy": -1.0, "token_accuracy": 0.0,
        "at_step":  0, "at_forwards": 0, "at_time_sec": 0.0,
    }
    if resume_meta and isinstance(resume_meta.get("best"), dict):
        best.update(resume_meta["best"])
    if global_step >= max_steps:
        print(f"[Resume] global_step {global_step} already >= max_steps {max_steps}; "
              "nothing to do.")
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

        # ---- LR schedule: rescale the optimizer lr before this step. All ZO
        #      opts read group['lr'] dynamically (ZOMuon also group['lr_1d']).
        if lr_sched != "constant" or warmup_steps > 0:
            _f = _lr_factor(global_step, max_steps, lr_sched, warmup_steps)
            for _g in optimizer.param_groups:
                if base_lr is not None:
                    _g["lr"] = base_lr * _f
                if base_lr_1d is not None and "lr_1d" in _g:
                    _g["lr_1d"] = base_lr_1d * _f
            cur_lr = (base_lr * _f) if base_lr is not None else None
        else:
            cur_lr = base_lr

        t0 = time.perf_counter()
        if is_first_order:
            # Standard first-order step: forward + backward + update. Train
            # mode (dropout on) is the usual fine-tuning regime; eval still
            # runs in eval mode below for a fair comparison with the ZO runs.
            model.train()
            optimizer.zero_grad(set_to_none=True)
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
            out.loss.backward()
            optimizer.step()
            step_loss["v"]      = out.loss.detach()
            step_forwards["n"] += 1  # 1 forward (+1 backward, not counted here)
        else:
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
            "train/lr":                 cur_lr if cur_lr is not None else 0.0,
            "train/step_time_sec":      dt,
            "train/avg_step_time_sec":  sum(step_times) / len(step_times),
            # Cumulative TRAINING-ONLY wall time: sum of per-step optimizer.step
            # durations. Excludes eval, checkpointing, model download, and any
            # other artifacts — this is the fair "time spent training" metric.
            "train/cumulative_train_time_sec": sum(step_times),
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
                best_meta = {
                    "best":           dict(best),
                    "total_steps":    global_step,
                    "total_forwards": total_forwards,
                    "run_name":       run_name,
                    "wandb_run_id":   wandb.run.id,
                    "task":           args.task,
                    "opt_name":       opt_name,
                    "owner":          owner,
                    "model_name":     model_name,
                    "model_tag":      model_tag,
                    "git_sha":        _git_sha(),
                }
                save_checkpoint(model, tokenizer, optimizer, best_dir,
                                best_meta, source_cfg_path=args.config)
                wandb.log({
                    "best_eval/logit_accuracy": best["logit_accuracy"],
                    "best_eval/token_accuracy": best["token_accuracy"],
                    "best_eval/at_step":        best["at_step"],
                    "best_eval/at_forwards":    best["at_forwards"],
                    "best_eval/at_time_sec":    best["at_time_sec"],
                }, step=global_step)
                print(f"  [best] new best logit_acc={best['logit_accuracy']:.4f} "
                      f"(saved -> {best_dir})")

            # Rolling "last" checkpoint every eval (on the durable PVC) so a
            # preempted + relaunched pod resumes from here — at most eval_steps
            # of progress lost. This save is OUTSIDE the train-step timer, so it
            # does not pollute train/cumulative_train_time_sec.
            last_meta = {
                "total_steps":    global_step,
                "total_forwards": total_forwards,
                "best":           dict(best),
                "run_name":       run_name,
                "wandb_run_id":   wandb.run.id,
                "task":           args.task,
                "opt_name":       opt_name,
                "owner":          owner,
                "model_name":     model_name,
                "model_tag":      model_tag,
            }
            save_checkpoint(model, tokenizer, optimizer, last_dir,
                            last_meta, source_cfg_path=args.config)

            model.eval()   # keep eval mode for the next ZO step

    # ---- Save LAST checkpoint -----------------------------------------------
    final_meta = {
        "total_steps":    global_step,
        "total_forwards": total_forwards,
        "wall_time_sec":  time.perf_counter() - run_start_time,
        "nan_aborted":    nan_seen,
        "best":           dict(best),
        "run_name":       run_name,
        "wandb_run_id":   wandb.run.id,
        "task":           args.task,
        "opt_name":       opt_name,
        "owner":          owner,
        "git_sha":        _git_sha(),
    }
    save_checkpoint(model, tokenizer, optimizer, last_dir,
                    final_meta, source_cfg_path=args.config)

    # ---- HF Hub: push best + last ONCE, only at end of training ----
    # Doing this here (instead of on every new best) means the run isn't
    # blocked by Hub network latency mid-training and we only pay one push
    # round-trip per checkpoint. The startup assertion already proved the
    # token is writable, so this is unlikely to fail.
    if push_to_hub:
        if best_dir.exists():
            _hub_push_folder(
                folder=best_dir,
                repo_id=hub_repo_id,
                sub_path=f"{run_name}/best",
                commit_message=(f"best ckpt {run_name} step={best['at_step']} "
                                f"logit_acc={best['logit_accuracy']:.4f}"),
            )
        _hub_push_folder(
            folder=last_dir,
            repo_id=hub_repo_id,
            sub_path=f"{run_name}/last",
            commit_message=f"last ckpt {run_name} step={global_step}",
        )

    # ---- Final summary ------------------------------------------------------
    total_time = time.perf_counter() - run_start_time
    summary = {
        "final/total_steps":       global_step,
        "final/total_forwards":    total_forwards,
        "final/avg_step_time_sec": (sum(step_times) / len(step_times)) if step_times else 0.0,
        "final/total_time_sec":    total_time,
        # Training-only wall time (sum of optimizer.step durations) — excludes
        # eval/checkpoint/download. Use THIS for cross-method time comparison,
        # not total_time_sec (which includes eval + the one-time model load).
        "final/train_only_time_sec": sum(step_times),
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
          f"{sum(step_times):.1f}s train-only / {total_time:.1f}s wall / "
          f"peak {summary.get('final/peak_mem_MB', 0):.0f}MB")
    if nan_seen:
        print("  WARN   run aborted on non-finite loss.")
    print(f"  CKPTS  best -> {best_dir}")
    print(f"         last -> {last_dir}")
    print("=" * 72)

    wandb.finish()


if __name__ == "__main__":
    main()
