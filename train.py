"""Unified ZO comparison harness for Qwen-0.8B on SuperGLUE.

Goal: run any of MeZO / Sparse-MeZO / LOZO / DiZO / HiZOO / QuZO / ZO-Muon /
ConMeZO / FZOO / PseuZO / SubZero under the SAME setup (model, dataset,
prompts, eval protocol) so the team's head-to-head numbers are apples-to-apples.
The optimizer is selected via the YAML config; the task (multirc | copa | sst2) is
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
  RUN_OWNER=nil python train.py --config configs/lozo.yaml --task sst2
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
import torch.optim.lr_scheduler
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

# First-order optimizers — taken straight from torch.optim. Routed through a
# separate training loop (backprop instead of closure-based ZO). Useful as the
# upper-bound baseline for the comparison plot.
FIRST_ORDER_OPTIMIZERS = {"Adam", "AdamW", "SGD"}


def load_optimizer_cls(name: str):
    if name in FIRST_ORDER_OPTIMIZERS:
        return getattr(torch.optim, name)
    if name not in OPTIMIZER_MODULES:
        raise ValueError(
            f"Unknown optimizer {name!r}; ZO methods: {list(OPTIMIZER_MODULES)}, "
            f"FO methods: {sorted(FIRST_ORDER_OPTIMIZERS)}"
        )
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
    p.add_argument("--task", required=True, choices=["multirc", "copa", "sst2"],
                   help="SuperGLUE / GLUE task to train / evaluate on")
    p.add_argument("--owner", default=None,
                   help="Run owner: maria | nil | cheng (falls back to $RUN_OWNER or yaml owner)")
    p.add_argument("--eval-batch-size", type=int, default=8,
                   help="Batch size for the eval forward (eval is bottleneck if =1)")
    p.add_argument("--ckpt-dir", default="checkpoints",
                   help="Root dir for best/last checkpoints (one subdir per run)")
    p.add_argument("--resume-from", default=None,
                   help="Path to a prior run's `last/` dir to resume from "
                        "(reuses model weights + optimizer state + step/forward/best counters)")
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
                    meta: dict, source_cfg_path: str | None = None,
                    scheduler=None):
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
        torch.save(optimizer.state_dict(), dest_dir / "optimizer.pt")
    except Exception as e:
        # Some custom optimizer states (closures, non-tensor objects) don't
        # pickle cleanly — record the failure but keep the rest of the ckpt.
        print(f"[ckpt] optimizer.state_dict() save failed at {dest_dir}: {e}")
    if scheduler is not None:
        try:
            torch.save(scheduler.state_dict(), dest_dir / "scheduler.pt")
        except Exception as e:
            print(f"[ckpt] scheduler.state_dict() save failed at {dest_dir}: {e}")
    if source_cfg_path:
        try:
            shutil.copy(source_cfg_path, dest_dir / "config.yaml")
        except Exception as e:
            print(f"[ckpt] config.yaml copy failed at {dest_dir}: {e}")
    with open(dest_dir / "training_meta.json", "w") as fp:
        json.dump(meta, fp, indent=2, default=str)


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
    if owner not in {"maria", "nil", "cheng"}:
        raise ValueError(
            f"--owner / RUN_OWNER / config 'owner' must be one of maria|nil|cheng (got {owner!r})"
        )

    opt_name   = cfg["optimizer"]["name"]
    opt_kwargs = cfg["optimizer"].get("kwargs", {}) or {}
    opt_cls    = load_optimizer_cls(opt_name)
    is_first_order = opt_name in FIRST_ORDER_OPTIMIZERS

    seed       = cfg.get("training", {}).get("seed", 42)
    batch_size = cfg.get("training", {}).get("batch_size", 16)
    max_steps  = cfg.get("training", {}).get("max_steps", 20000)
    eval_steps = cfg.get("training", {}).get("eval_steps", 500)
    num_train  = cfg.get("training", {}).get("num_train", 1000)
    num_eval   = cfg.get("training", {}).get("num_eval", 1000)
    loss_type  = cfg.get("training", {}).get("loss_type", "cross_entropy")
    if loss_type not in {"cross_entropy", "token_accuracy", "logit_accuracy"}:
        raise ValueError(f"training.loss_type must be one of: cross_entropy, token_accuracy, logit_accuracy (got {loss_type})")
    model_name = cfg.get("model", {}).get("name", "Qwen/Qwen3.5-0.8B")
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

    # ---- Resume meta (loaded early so global_step / best / total_forwards
    # ----   are seeded from disk before training starts) ------------------
    resume_meta = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
        meta_path   = resume_path / "training_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                resume_meta = json.load(f)
            print(f"[Resume] meta from {meta_path}: "
                  f"step={resume_meta.get('total_steps')} "
                  f"forwards={resume_meta.get('total_forwards')} "
                  f"best={resume_meta.get('best')}")
        else:
            print(f"[Resume] WARNING: no training_meta.json at {meta_path}; "
                  "weights will load but counters start from 0.")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Fail fast on HF auth before doing anything expensive ----
    if push_to_hub:
        _assert_hub_writable(hub_repo_id)

    # ---- WandB: resume the SAME run if we're picking up a prior ckpt ------
    resumed_run_id   = (resume_meta or {}).get("wandb_run_id")
    resumed_run_name = (resume_meta or {}).get("run_name")
    if resumed_run_id and resumed_run_name:
        run_name = resumed_run_name
        wandb.init(
            project="Zero-Order-Opt",
            id=resumed_run_id,
            resume="allow",
            name=run_name,
            group=args.task,
            tags=[owner, opt_name, args.task],
            config={**cfg, "task": args.task, "owner": owner,
                    "_resolved_seed": seed,
                    "_resolved_dtype": dtype_str,
                    "_resumed_from":   args.resume_from,
                    "_hub_push":       push_to_hub,
                    "_hub_repo_id":    hub_repo_id},
        )
        print(f"[Resume] continuing WandB run id={resumed_run_id} name={run_name}")
    else:
        stamp    = datetime.now(timezone.utc).strftime("%m_%d_%H_%M_%S")
        run_name = f"{owner}-{opt_name}-{args.task}-{stamp}"
        wandb.init(
            project="Zero-Order-Opt",
            name=run_name,
            group=args.task,                          # group all multirc / all copa together
            tags=[owner, opt_name, args.task],        # filter in the dashboard
            config={**cfg, "task": args.task, "owner": owner,
                    "_resolved_seed": seed,
                    "_resolved_dtype": dtype_str,
                    "_resumed_from":   args.resume_from,
                    "_hub_push":       push_to_hub,
                    "_hub_repo_id":    hub_repo_id},
        )

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
    pad_id = tokenizer.pad_token_id
    # ZO assumes no dropout noise in the gradient estimate; FO trains with
    # dropout enabled like a normal first-order fine-tune.
    if is_first_order:
        model.train()
    else:
        model.eval()

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

    # ---- Tokenize train + build loader -----------------------------------
    if loss_type == "logit_accuracy":
        train_packs = [spec.format_eval(ex, tokenizer) for ex in ds["train"]]
        train_loader = DataLoader(
            train_packs, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: b,
        )
    else:
        train_packs = [spec.format_train(ex, tokenizer) for ex in ds["train"]]
        train_loader = DataLoader(
            train_packs, batch_size=batch_size, shuffle=True,
            collate_fn=lambda b: pad_collate(b, pad_id),
        )

    # ---- Optimizer --------------------------------------------------------
    optimizer = opt_cls(
        (p for p in model.parameters() if p.requires_grad),
        **opt_kwargs,
    )
    print(f"[Opt] {opt_name}({opt_kwargs})  "
          f"{'(first-order, backprop)' if is_first_order else '(zeroth-order, closure)'}")

    # ---- Scheduler --------------------------------------------------------
    use_cosine_scheduler = cfg.get("optimizer", {}).get("use_cosine_scheduler", False)
    if use_cosine_scheduler:
        initial_step = int(resume_meta["total_steps"]) if resume_meta else 0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps, last_epoch=initial_step - 1
        )
        print(f"[Scheduler] CosineAnnealingLR enabled (T_max={max_steps}, initial_step={initial_step})")
    else:
        scheduler = None

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
                  
        if scheduler is not None:
            sched_state_path = Path(args.resume_from) / "scheduler.pt"
            if sched_state_path.exists():
                try:
                    scheduler.load_state_dict(torch.load(sched_state_path, weights_only=False))
                    print(f"[Resume] loaded scheduler state from {sched_state_path}")
                except Exception as e:
                    print(f"[Resume] could not load scheduler state ({e}); continuing fresh")

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
        # ---- Epoch Hooks ----
        if global_step % steps_per_epoch == 0 and opt_name == "PseuZO":
            epoch_idx = global_step // steps_per_epoch
            if hasattr(optimizer, "on_epoch_start"):
                optimizer.on_epoch_start(epoch_idx, math.ceil(total_epochs))

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        if loss_type != "logit_accuracy":
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
                if loss_type == "logit_accuracy":
                    rows = []
                    for ex_i, ep in enumerate(batch):
                        for c_idx, cand_ids in enumerate(ep["candidates"]):
                            full = ep["prompt_ids"] + cand_ids
                            rows.append((ex_i, c_idx, full, len(ep["prompt_ids"]), len(cand_ids)))
                    max_len = max(len(r[2]) for r in rows)
                    ids_list, attn_list = [], []
                    for _, _, full, _, _ in rows:
                        pad_n = max_len - len(full)
                        ids_list.append(full + [pad_id] * pad_n)
                        attn_list.append([1] * len(full) + [0] * pad_n)
                    ids = torch.tensor(ids_list, dtype=torch.long, device=device)
                    attn = torch.tensor(attn_list, dtype=torch.long, device=device)
                    
                    with torch.no_grad():
                        logits = model(input_ids=ids, attention_mask=attn).logits
                    
                    log_probs = torch.log_softmax(logits, dim=-1)
                    cand_lls = [[None] * len(ep["candidates"]) for ep in batch]
                    for row_i, (ex_i, c_idx, full, prompt_len, cand_len) in enumerate(rows):
                        ll = 0.0
                        for k in range(cand_len):
                            tok_id = full[prompt_len + k]
                            ll += log_probs[row_i, prompt_len + k - 1, tok_id].item()
                        ll /= max(1, cand_len)
                        cand_lls[ex_i][c_idx] = ll
                        
                    correct = 0
                    for ex_i, ep in enumerate(batch):
                        scores = cand_lls[ex_i]
                        pred = max(range(len(scores)), key=lambda c: scores[c])
                        if pred == ep["gold_idx"]:
                            correct += 1
                    
                    loss_t = torch.tensor(1.0 - (correct / len(batch)), device=device)
                    step_loss["v"] = loss_t.detach()
                    return loss_t
                
                elif loss_type == "token_accuracy":
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    shift_logits = out.logits[:, :-1, :]
                    shift_labels = batch["labels"][:, 1:]
                    mask = shift_labels != -100
                    if mask.any():
                        preds = shift_logits.argmax(dim=-1)
                        correct = (preds[mask] == shift_labels[mask]).sum()
                        token_acc = correct.float() / mask.sum()
                    else:
                        token_acc = torch.tensor(0.0, device=device)
                    loss_t = 1.0 - token_acc
                    step_loss["v"] = loss_t.detach()
                    return loss_t
                
                else: # cross_entropy
                    out = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    step_loss["v"] = out.loss.detach()
                    return out.loss

            # Enriched path: cheap forward up to logits, then CE with grad on
            # JUST the logits. Main model parameters never see autograd.
            if loss_type == "logit_accuracy":
                rows = []
                for ex_i, ep in enumerate(batch):
                    for c_idx, cand_ids in enumerate(ep["candidates"]):
                        full = ep["prompt_ids"] + cand_ids
                        rows.append((ex_i, c_idx, full, len(ep["prompt_ids"]), len(cand_ids)))
                max_len = max(len(r[2]) for r in rows)
                ids_list, attn_list = [], []
                for _, _, full, _, _ in rows:
                    pad_n = max_len - len(full)
                    ids_list.append(full + [pad_id] * pad_n)
                    attn_list.append([1] * len(full) + [0] * pad_n)
                ids = torch.tensor(ids_list, dtype=torch.long, device=device)
                attn = torch.tensor(attn_list, dtype=torch.long, device=device)
                
                with torch.no_grad():
                    base_logits = model(input_ids=ids, attention_mask=attn).logits
                logits = base_logits.detach().requires_grad_(True)
                
                log_probs = torch.log_softmax(logits, dim=-1)
                cand_lls = [[None] * len(ep["candidates"]) for ep in batch]
                for row_i, (ex_i, c_idx, full, prompt_len, cand_len) in enumerate(rows):
                    ll = 0.0
                    for k in range(cand_len):
                        tok_id = full[prompt_len + k]
                        ll += log_probs[row_i, prompt_len + k - 1, tok_id].item()
                    ll /= max(1, cand_len)
                    cand_lls[ex_i][c_idx] = ll
                    
                correct = 0
                for ex_i, ep in enumerate(batch):
                    scores = cand_lls[ex_i]
                    pred = max(range(len(scores)), key=lambda c: scores[c])
                    if pred == ep["gold_idx"]:
                        correct += 1
                
                loss_t = torch.tensor(1.0 - (correct / len(batch)), device=device)
                step_loss["v"] = loss_t.detach()
                return loss_t.detach(), logits.detach(), torch.zeros_like(logits)

            # For cross_entropy and token_accuracy, we run the model on batch["input_ids"]
            with torch.no_grad():
                base_out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                logits = base_out.logits.detach().requires_grad_(True)
                del base_out

            with torch.enable_grad():
                if opt_name == "PseuZO":
                    # PseuZO calls closure twice per batch. The second call is for 
                    # perturbed outputs and does not need gradients. Skipping 
                    # the backward pass here saves ~3GB of memory.
                    if step_forwards["n"] == 2:
                        if loss_type == "token_accuracy":
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = batch["labels"][..., 1:].contiguous()
                            mask = shift_labels != -100
                            if mask.any():
                                preds = shift_logits.argmax(dim=-1)
                                correct = (preds[mask] == shift_labels[mask]).sum()
                                token_acc = correct.float() / mask.sum()
                            else:
                                token_acc = torch.tensor(0.0, device=logits.device)
                            loss_t = 1.0 - token_acc
                        else: # cross_entropy
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = batch["labels"][..., 1:].contiguous()
                            loss_t = F.cross_entropy(
                                shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1),
                                ignore_index=-100,
                            )
                        step_loss["v"] = loss_t.detach()
                        return loss_t.detach(), logits.detach(), None
                        
                    # Chunk gradient computation over the batch dimension to avoid 
                    # a massive peak memory spike during cross-entropy backprop.
                    grad_logits = torch.zeros_like(logits)
                    loss_sum = 0.0
                    valid_tokens = 0
                    
                    for i in range(logits.size(0)):
                        # Small local graph for just 1 sequence
                        l_i = logits[i:i+1].detach().requires_grad_(True)
                        shift_lbl = batch["labels"][i:i+1, 1:].contiguous()
                        
                        if loss_type == "token_accuracy":
                            shift_l = l_i[..., :-1, :].contiguous()
                            mask = shift_lbl != -100
                            if mask.any():
                                preds = shift_l.argmax(dim=-1)
                                correct = (preds[mask] == shift_lbl[mask]).sum()
                                token_acc = correct.float() / mask.sum()
                                loss_i = 1.0 - token_acc
                                valid_tokens += 1
                                loss_sum += loss_i.item()
                            else:
                                loss_i = torch.tensor(0.0, device=logits.device)
                            
                            if loss_i.requires_grad and loss_i.grad_fn is not None:
                                grad_logits[i:i+1] = torch.autograd.grad(loss_i, l_i)[0].detach()
                        else: # cross_entropy
                            shift_l = l_i[..., :-1, :].contiguous()
                            loss_i = F.cross_entropy(
                                shift_l.view(-1, shift_l.size(-1)), 
                                shift_lbl.view(-1), 
                                ignore_index=-100, 
                                reduction='sum'
                            )
                            tokens_i = (shift_lbl != -100).sum().item()
                            valid_tokens += tokens_i
                            if loss_i.item() > 0:
                                grad_logits[i:i+1] = torch.autograd.grad(loss_i, l_i)[0].detach()
                            loss_sum += loss_i.item()
                        del l_i, shift_lbl, loss_i
                        
                    loss_t = torch.tensor(loss_sum / max(1, valid_tokens), device=logits.device)
                    if loss_type != "token_accuracy":
                        grad_logits /= max(1, valid_tokens)
                else:
                    if loss_type == "token_accuracy":
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = batch["labels"][..., 1:].contiguous()
                        mask = shift_labels != -100
                        if mask.any():
                            preds = shift_logits.argmax(dim=-1)
                            correct = (preds[mask] == shift_labels[mask]).sum()
                            token_acc = correct.float() / mask.sum()
                        else:
                            token_acc = torch.tensor(0.0, device=logits.device)
                        loss_t = 1.0 - token_acc
                        grad_logits = torch.zeros_like(logits)
                    else: # cross_entropy
                        shift_logits = logits[..., :-1, :].contiguous()
                        shift_labels = batch["labels"][..., 1:].contiguous()
                        loss_t = F.cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        )
                        grad_logits = torch.autograd.grad(loss_t, logits)[0].detach()
                
            step_loss["v"] = loss_t.detach()
            return loss_t.detach(), logits.detach(), grad_logits

        t0 = time.perf_counter()
        if is_first_order:
            # Standard backprop training step.
            optimizer.zero_grad(set_to_none=True)
            if loss_type == "logit_accuracy":
                rows = []
                for ex_i, ep in enumerate(batch):
                    for c_idx, cand_ids in enumerate(ep["candidates"]):
                        full = ep["prompt_ids"] + cand_ids
                        rows.append((ex_i, c_idx, full, len(ep["prompt_ids"]), len(cand_ids)))
                max_len = max(len(r[2]) for r in rows)
                ids_list, attn_list = [], []
                for _, _, full, _, _ in rows:
                    pad_n = max_len - len(full)
                    ids_list.append(full + [pad_id] * pad_n)
                    attn_list.append([1] * len(full) + [0] * pad_n)
                ids = torch.tensor(ids_list, dtype=torch.long, device=device)
                attn = torch.tensor(attn_list, dtype=torch.long, device=device)
                
                out_logits = model(input_ids=ids, attention_mask=attn).logits
                log_probs = torch.log_softmax(out_logits, dim=-1)
                cand_lls = [[None] * len(ep["candidates"]) for ep in batch]
                for row_i, (ex_i, c_idx, full, prompt_len, cand_len) in enumerate(rows):
                    ll = 0.0
                    for k in range(cand_len):
                        tok_id = full[prompt_len + k]
                        ll += log_probs[row_i, prompt_len + k - 1, tok_id].item()
                    ll /= max(1, cand_len)
                    cand_lls[ex_i][c_idx] = ll
                    
                correct = 0
                for ex_i, ep in enumerate(batch):
                    scores = cand_lls[ex_i]
                    pred = max(range(len(scores)), key=lambda c: scores[c])
                    if pred == ep["gold_idx"]:
                        correct += 1
                loss_t = torch.tensor(1.0 - (correct / len(batch)), device=device, requires_grad=True)
            elif loss_type == "token_accuracy":
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                shift_logits = out.logits[:, :-1, :]
                shift_labels = batch["labels"][:, 1:]
                mask = shift_labels != -100
                if mask.any():
                    preds = shift_logits.argmax(dim=-1)
                    correct = (preds[mask] == shift_labels[mask]).sum()
                    token_acc = correct.float() / mask.sum()
                else:
                    token_acc = torch.tensor(0.0, device=device)
                loss_t = 1.0 - token_acc
            else: # cross_entropy
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss_t = out.loss
                
            if loss_t.requires_grad and loss_t.grad_fn is not None:
                loss_t.backward()
                optimizer.step()
            else:
                if global_step == 0 or global_step % 1000 == 0:
                    print(f"[Warning] Loss has no gradient function for backpropagation (loss_type={loss_type}). Zero gradients applied.")
            
            step_loss["v"]     = loss_t.detach()
            step_forwards["n"] = 1
        else:
            optimizer.step(closure)
            
        if scheduler is not None:
            scheduler.step()
            
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
            "train/lr":                 scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"],
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
                best_meta = {
                    "best":           dict(best),
                    "total_steps":    global_step,
                    "total_forwards": total_forwards,
                    "run_name":       run_name,
                    "wandb_run_id":   wandb.run.id,
                    "task":           args.task,
                    "opt_name":       opt_name,
                    "owner":          owner,
                    "git_sha":        _git_sha(),
                }
                save_checkpoint(model, tokenizer, optimizer, best_dir,
                                best_meta, source_cfg_path=args.config,
                                scheduler=scheduler)
                wandb.log({
                    "best_eval/logit_accuracy": best["logit_accuracy"],
                    "best_eval/token_accuracy": best["token_accuracy"],
                    "best_eval/at_step":        best["at_step"],
                    "best_eval/at_forwards":    best["at_forwards"],
                    "best_eval/at_time_sec":    best["at_time_sec"],
                }, step=global_step)
                print(f"  [best] new best logit_acc={best['logit_accuracy']:.4f} "
                      f"(saved -> {best_dir})")

            # Restore mode for the next training step
            if is_first_order:
                model.train()
            else:
                model.eval()

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
                    final_meta, source_cfg_path=args.config,
                    scheduler=scheduler)

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
