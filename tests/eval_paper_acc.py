"""Paper-style per-candidate log-likelihood eval for MeZO-template tasks.

For each example in `validation` of a MeZO task, compute the per-candidate
log-likelihood and pick the argmax. This matches the eval used in:
  - MeZO (Malladi 2023, arXiv:2305.17333)
  - ConMeZO (arXiv:2511.02757)
  - FZOO (arXiv:2506.09034)
  - ZO-Muon (arXiv:2602.17155)

Run against a saved checkpoint:
    python tests/eval_paper_acc.py \
        --checkpoint best_checkpoint_causal \
        --task sst2 --num_eval 1000 --seed 29

Reports accuracy (per-candidate argmax) and majority-class accuracy as a
sanity floor.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.mezo_tasks import load_mezo_task
from data.mezo_templates import get_template


@torch.no_grad()
def score_candidate(model, tokenizer, prompt: str, candidate: str, device):
    """Return total per-token log-prob of `candidate` given `prompt` under
    the causal LM. We tokenize prompt+candidate together (so BPE merges with
    the prompt's last token), then sum log-probs over the candidate tokens.
    """
    full = prompt + candidate
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    full_ids   = tokenizer(full,   return_tensors="pt").input_ids.to(device)
    cand_len   = full_ids.size(1) - prompt_ids.size(1)
    if cand_len <= 0:
        return float("-inf")

    out = model(full_ids)
    # shift: logits[t] predicts token t+1. To score full_ids[prompt_len..end-1],
    # we read logits at positions [prompt_len-1 .. end-2].
    logits = out.logits[0]  # (T, V)
    log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    total = 0.0
    p = prompt_ids.size(1)
    for k in range(cand_len):
        total += log_probs[p - 1 + k, full_ids[0, p + k]].item()
    return total


def eval_task(checkpoint: str, task: str, num_eval: int, seed: int,
              dtype: torch.dtype = torch.float16) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {checkpoint}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, torch_dtype=dtype, use_safetensors=True
    ).to(device)
    model.eval()

    print(f"Loading task {task} (num_eval={num_eval}, seed={seed})", flush=True)
    ds = load_mezo_task(task, seed=seed, num_train=1, num_dev=1,
                        num_eval=num_eval)
    tpl = get_template(task)
    if tpl.type != "classification":
        raise NotImplementedError(
            f"per-candidate eval only for classification tasks; {task} is {tpl.type}")

    # Pull the raw examples back out so we can re-render prompts and choices.
    # We use ds["test"], which is the eval split (validation).
    eval_split = ds["test"]
    n = len(eval_split)
    print(f"Scoring {n} examples...", flush=True)

    correct = 0
    by_gold = {0: [0, 0], 1: [0, 0], 2: [0, 0]}  # gold_idx -> [n_correct, n_total]
    t0 = time.time()
    for i, ex in enumerate(eval_split):
        prompt   = ex["prompt"]
        choices  = ex["all_choices"]
        gold_idx = int(ex["answer_idx"])

        scores = [score_candidate(model, tokenizer, prompt, c, device)
                  for c in choices]
        pred = int(max(range(len(scores)), key=lambda k: scores[k]))
        if pred == gold_idx:
            correct += 1
        by_gold.setdefault(gold_idx, [0, 0])
        by_gold[gold_idx][1] += 1
        if pred == gold_idx:
            by_gold[gold_idx][0] += 1

        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"  {i+1}/{n}  running acc={correct/(i+1):.4f}", flush=True)

    elapsed = time.time() - t0
    acc = correct / n if n else 0.0
    # Majority-class baseline (always pick the more frequent gold class).
    totals = [by_gold[k][1] for k in sorted(by_gold)]
    maj = max(totals) / sum(totals) if sum(totals) else 0.0

    out = {
        "checkpoint": checkpoint,
        "task":       task,
        "num_eval":   n,
        "accuracy":   acc,
        "majority":   maj,
        "elapsed_s":  elapsed,
        "by_gold":    {str(k): v for k, v in by_gold.items()},
    }
    print("\n=== EVAL RESULT ===", flush=True)
    print(f"  checkpoint:        {checkpoint}", flush=True)
    print(f"  task:              {task}", flush=True)
    print(f"  num_eval:          {n}", flush=True)
    print(f"  paper-style acc:   {acc*100:.2f}%", flush=True)
    print(f"  majority baseline: {maj*100:.2f}%", flush=True)
    print(f"  per-class:         {out['by_gold']}", flush=True)
    print(f"  elapsed:           {elapsed:.1f}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--num_eval", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--dtype", default="float16",
                    choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
             "float32": torch.float32}[args.dtype]
    eval_task(args.checkpoint, args.task, args.num_eval, args.seed, dtype=dtype)


if __name__ == "__main__":
    main()
