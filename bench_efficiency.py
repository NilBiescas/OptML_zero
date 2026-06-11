"""Efficiency micro-benchmark: mean s/step + peak GPU memory per optimizer.

For each METHOD x TASK, loads Qwen3.5-0.8B (fp32), builds the optimizer from
configs/<method>.yaml exactly like train.py, and runs `--steps` optimizer
steps at `--batch-size` (default 1) with the standard cross-entropy closure.
Timing/memory protocol: 1 warmup step (excluded), then reset peak-memory
stats and time the remaining steps; peak GB = torch.cuda.max_memory_allocated.

Results are appended to memory_computation.txt (one line per combo) and
printed to stdout.

Usage (on a GPU node):
    python bench_efficiency.py                       # all 5 methods x 2 tasks
    python bench_efficiency.py --methods mezo dizo --tasks copa
    python bench_efficiency.py --batch-size 16 --steps 10
"""
import argparse
import importlib
import json
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from tasks import load_task

MODEL_NAME = "Qwen/Qwen3.5-0.8B"
METHODS = ["mezo", "conmezo", "fzoo", "zo_muon", "dizo"]
TASKS = ["multirc", "copa"]


def pad_collate(batch, pad_id):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append(b["attention_mask"] + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def load_optimizer_cls(name):
    mod = importlib.import_module(f"optimizers.{name.lower()}")
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type) and issubclass(obj, torch.optim.Optimizer) \
                and obj is not torch.optim.Optimizer:
            return obj
    raise ValueError(f"no Optimizer subclass in optimizers/{name}")


def bench_one(method, task, batch_size, steps, device, out_path):
    cfg = yaml.safe_load(open(f"configs/{method}.yaml"))
    opt_name = cfg["optimizer"]["name"]
    opt_kwargs = cfg["optimizer"].get("kwargs", {}) or {}
    opt_cls = load_optimizer_cls(opt_name)

    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32).to(device)
    model.eval()  # dropout off: deterministic timing

    # Same param annotations train.py applies (DiZO/ZO-Muon read these).
    for i, (name, p) in enumerate(model.named_parameters()):
        if p.requires_grad:
            p.param_id = i
            p.param_name = name

    # Same 10+1 examples for every method (shuffle=False) -> comparable.
    spec, ds = load_task(task, num_train=batch_size * (steps + 1), seed=42)
    packs = [spec.format_train(ex, tokenizer) for ex in ds["train"]]
    batches = [pad_collate(packs[i * batch_size:(i + 1) * batch_size],
                           tokenizer.pad_token_id)
               for i in range(steps + 1)]
    batches = [{k: v.to(device) for k, v in b.items()} for b in batches]

    optimizer = opt_cls((p for p in model.parameters() if p.requires_grad),
                        **opt_kwargs)

    def make_closure(batch):
        def closure(need_output=False):
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
            return out.loss
        return closure

    # Warmup step (cuda init, lazy allocs) -- excluded from stats.
    optimizer.step(make_closure(batches[0]))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for b in batches[1:]:
        t0 = time.perf_counter()
        optimizer.step(make_closure(b))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    s_per_step = sum(times) / len(times)
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    rec = {"method": method, "task": task, "model": MODEL_NAME,
           "batch_size": batch_size, "timed_steps": len(times),
           "s_per_step": round(s_per_step, 4), "peak_GB": round(peak_gb, 2)}
    line = (f"{method:8} {task:8} bs={batch_size:<3} steps={len(times):<3} "
            f"s/step={s_per_step:.3f}  peak={peak_gb:.2f} GB")
    print("[bench] " + line, flush=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(rec) + "\n")

    del optimizer, model
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--out", default="memory_computation.txt")
    args = ap.parse_args()

    device = torch.device("cuda")
    print(f"[bench] {MODEL_NAME} fp32  bs={args.batch_size} steps={args.steps} "
          f"(1 warmup excluded)  gpu={torch.cuda.get_device_name(0)}", flush=True)
    for task in args.tasks:
        for method in args.methods:
            try:
                bench_one(method, task, args.batch_size, args.steps,
                          device, args.out)
            except Exception as e:
                print(f"[bench] FAIL {method}/{task}: {type(e).__name__}: {e}",
                      flush=True)
    print("[bench] DONE", flush=True)
    print(open(args.out).read(), flush=True)


if __name__ == "__main__":
    main()
