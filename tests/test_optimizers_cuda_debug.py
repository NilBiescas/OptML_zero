"""CUDA debug script for the three new zeroth-order optimizers.

Runs each optimizer (ConMeZO, FZOO, ZOMuon) on a tiny causal LM
(facebook/opt-125m, fp16) for 3 steps on GPU. Verifies:
    - the model loads in fp16 on CUDA
    - each new optimizer's step() completes without CUDA errors / OOM
    - peak VRAM usage is reasonable
    - losses are finite
    - wandb logs land in project `optml-zero-debug`

This is NOT real training — it's a smoke test that the implementations work
end-to-end on the GPU before we kick off paper-faithful runs. Each run logs
to wandb so you can see them in the dashboard.
"""

import math
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizers.conmezo import ConMeZO
from optimizers.fzoo import FZOO
from optimizers.zo_muon import ZOMuon

# wandb logging is best-effort: if auth fails we still want the debug to run.
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False


MODEL_NAME = "facebook/opt-125m"
SEQ_LEN    = 64
BATCH_SIZE = 2
N_STEPS    = 3


def _make_batch(tokenizer, device, dtype):
    """Two fixed prompts, labels = next-token targets (causal LM)."""
    texts = ["The capital of France is", "The quick brown fox jumps"]
    enc = tokenizer(texts, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=SEQ_LEN)
    input_ids = enc["input_ids"].to(device)
    attn      = enc["attention_mask"].to(device)
    # Causal LM: labels == input_ids with pad masked to -100.
    labels = input_ids.clone()
    labels[attn == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def _inject_param_ids(model):
    """Mirror train.py:220-223 — stable per-param IDs for DDP-safe RNG."""
    for i, (_, p) in enumerate(model.named_parameters()):
        if p.requires_grad:
            p.param_id = i


def _run_one(opt_name, opt_factory, model_cls, tokenizer_cls):
    """Run N_STEPS of the given optimizer on opt-125m, return summary dict."""
    print(f"\n========== {opt_name} ==========", flush=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda")
    dtype  = torch.float16

    print(f"  loading {MODEL_NAME} in {dtype} on {device}...", flush=True)
    t0 = time.time()
    tokenizer = tokenizer_cls.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # use_safetensors=True avoids the CVE-2025-32434 torch.load check that
    # transformers 5.x raises against torch < 2.6.
    model = model_cls.from_pretrained(MODEL_NAME, torch_dtype=dtype,
                                       use_safetensors=True).to(device)
    model.eval()  # ZO needs eval mode (no dropout) for deterministic F+/F-.
    _inject_param_ids(model)
    print(f"  model loaded in {time.time()-t0:.1f}s, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params",
          flush=True)

    batch = _make_batch(tokenizer, device, dtype)
    print(f"  batch input_ids: {tuple(batch['input_ids'].shape)}, dtype={batch['input_ids'].dtype}",
          flush=True)

    optimizer = opt_factory(model)
    print(f"  optimizer ready: {type(optimizer).__name__}", flush=True)

    def closure():
        with torch.no_grad():
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
            return out.loss.float()  # ZO scalar c needs full fp32 stability

    losses = []
    step_times = []
    for step in range(N_STEPS):
        t_step = time.time()
        loss = optimizer.step(closure)
        torch.cuda.synchronize()
        step_dt = time.time() - t_step
        step_times.append(step_dt)
        losses.append(float(loss))
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"  step {step}: loss={loss:.4f}  dt={step_dt:.2f}s  peak_VRAM={peak_mb:.0f} MB",
              flush=True)
        if _HAS_WANDB and wandb.run is not None:
            wandb.log({
                f"{opt_name}/loss":       loss,
                f"{opt_name}/step_time":  step_dt,
                f"{opt_name}/peak_VRAM_MB": peak_mb,
            }, step=step)

    final_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    finite = all(math.isfinite(L) for L in losses)
    print(f"  -> result: finite={finite}, peak_VRAM={final_peak_mb:.0f} MB, "
          f"mean_step_time={sum(step_times)/len(step_times):.2f}s", flush=True)

    # Free everything for the next optimizer.
    del optimizer, model, tokenizer, batch
    torch.cuda.empty_cache()

    return {
        "optimizer":      opt_name,
        "losses":         losses,
        "finite":         finite,
        "peak_VRAM_MB":   final_peak_mb,
        "mean_step_sec":  sum(step_times) / len(step_times),
    }


def main():
    print("=" * 60, flush=True)
    print(f"CUDA device:    {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM total:     {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB",
          flush=True)
    print(f"torch:          {torch.__version__}", flush=True)
    print(f"model:          {MODEL_NAME} (seq_len={SEQ_LEN}, bs={BATCH_SIZE}, "
          f"steps={N_STEPS} per optimizer)", flush=True)
    print("=" * 60, flush=True)

    # Import HF lazily so the import error message is precise.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Init a single wandb run that covers all three optimizers (one chart each).
    if _HAS_WANDB:
        try:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "optml-zero-debug"),
                name="cuda-debug-3opt-" + time.strftime("%Y%m%d-%H%M%S"),
                config={
                    "model": MODEL_NAME, "seq_len": SEQ_LEN,
                    "batch_size": BATCH_SIZE, "n_steps": N_STEPS,
                    "gpu": torch.cuda.get_device_name(0),
                },
            )
            print(f"wandb: logging to {wandb.run.url}", flush=True)
        except Exception as e:
            print(f"wandb init failed: {e} -- continuing without wandb", flush=True)

    factories = [
        ("ConMeZO", lambda m: ConMeZO(m.parameters(),
                                       lr=1e-7, eps=1e-3,
                                       cone_theta=1.4, cone_beta=0.99, seed=0)),
        ("FZOO",    lambda m: FZOO(m.parameters(),
                                    lr=1e-6, eps=1e-3, Nq=4,
                                    sigma_floor=1e-6, seed=0)),
        ("ZOMuon",  lambda m: ZOMuon(m.parameters(),
                                      lr=1e-3, eps=1e-3,
                                      r=16, Nq=2, ns_steps=5,
                                      refresh_T=100, momentum=0.95, seed=0)),
    ]

    summaries = []
    failed = []
    for name, factory in factories:
        try:
            summaries.append(_run_one(name, factory,
                                       AutoModelForCausalLM, AutoTokenizer))
        except Exception as e:
            print(f"\nFAIL {name}: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            failed.append((name, str(e)))

    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for s in summaries:
        status = "PASS" if s["finite"] else "FAIL(non-finite)"
        print(f"  {status:10s} {s['optimizer']:8s}  losses={[round(L,3) for L in s['losses']]}  "
              f"peak_VRAM={s['peak_VRAM_MB']:.0f}MB  mean_step={s['mean_step_sec']:.2f}s",
              flush=True)
    for name, msg in failed:
        print(f"  FAIL       {name:8s}  -> {msg}", flush=True)

    if _HAS_WANDB and wandb.run is not None:
        wandb.summary["all_pass"] = (not failed) and all(s["finite"] for s in summaries)
        wandb.finish()

    return 0 if (not failed) and all(s["finite"] for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
