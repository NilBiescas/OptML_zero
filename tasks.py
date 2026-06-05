"""SuperGLUE task definitions for the qwen-superglue comparison harness.

Each task implements:
- `format_train_example(ex, tokenizer)`: prompt + gold verbalizer, returns
  `(input_ids, labels)` where labels mask everything except the verbalizer
  tokens (standard MeZO prompt-based fine-tuning setup).
- `format_eval_example(ex, tokenizer)`: returns the prompt and the list of
  candidate-completion token-id sequences plus the gold-candidate index.
  The harness scores each candidate by total log-likelihood and picks the
  argmax — that's the "logit-level" accuracy metric.

Two tasks live here: MultiRC (binary Yes/No on candidate answers) and COPA
(2-way choice between alternatives). Register a new task by adding it to
the TASKS dict at the bottom — the harness picks it up by name.
"""
from dataclasses import dataclass
from typing import Callable, List, Tuple

from datasets import load_dataset, DatasetDict


@dataclass
class TaskSpec:
    name: str                 # short id used in --task / config
    hf_subset: str            # subset name passed to load_dataset("super_glue", ...)
    format_train: Callable    # (example, tokenizer) -> dict(input_ids, labels)
    format_eval:  Callable    # (example, tokenizer) -> dict(prompt_ids, candidates, gold_idx)
    dataset_path: str = "super_glue" # hf dataset path (e.g. glue or super_glue)


# --------------------------------------------------------------------------
# MultiRC — binary Yes/No on (paragraph, question, candidate answer)
# Prompt template follows the MeZO / HiZOO appendix convention.
# --------------------------------------------------------------------------

def _multirc_prompt(ex) -> str:
    return (
        f"{ex['paragraph']}\n"
        f"Question: {ex['question']}\n"
        f"I found this answer \"{ex['answer']}\". Is that correct? Yes or No?\n"
        f"Answer:"
    )


def _multirc_verbalizer(label: int) -> str:
    # label==1 means the candidate answer is correct
    return " Yes" if label == 1 else " No"


def multirc_format_train(ex, tokenizer):
    prompt = _multirc_prompt(ex)
    verb   = _multirc_verbalizer(ex["label"])
    full   = prompt + verb
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids   = tokenizer(full,   add_special_tokens=False)["input_ids"]
    # Mask everything except the verbalizer tokens
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    # Defensive: pad/truncate so lengths line up exactly
    while len(labels) < len(full_ids):
        labels.append(-100)
    labels = labels[:len(full_ids)]
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


def multirc_format_eval(ex, tokenizer):
    prompt = _multirc_prompt(ex)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    candidates = [
        tokenizer(" No",  add_special_tokens=False)["input_ids"],
        tokenizer(" Yes", add_special_tokens=False)["input_ids"],
    ]
    return {
        "prompt_ids": prompt_ids,
        "candidates": candidates,
        "gold_idx":   int(ex["label"]),  # 0 → No, 1 → Yes
    }


# --------------------------------------------------------------------------
# COPA — pick the more plausible of two alternatives.
# Prompt: premise + connector ("because"/"so") + each candidate; rank by LL.
# --------------------------------------------------------------------------

def _copa_connector(question: str) -> str:
    return "because" if question == "cause" else "so"


def _copa_strip(s: str) -> str:
    # COPA choices are sentences ending in '.'; lowercase first char for natural flow
    s = s.rstrip(".").strip()
    return s[:1].lower() + s[1:] if s else s


def copa_format_train(ex, tokenizer):
    premise   = ex["premise"].rstrip(".").strip()
    connector = _copa_connector(ex["question"])
    gold      = ex["choice1"] if ex["label"] == 0 else ex["choice2"]
    prompt    = f"{premise} {connector}"
    completion = f" {_copa_strip(gold)}."
    full       = prompt + completion
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids   = tokenizer(full,   add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    while len(labels) < len(full_ids):
        labels.append(-100)
    labels = labels[:len(full_ids)]
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


def copa_format_eval(ex, tokenizer):
    premise   = ex["premise"].rstrip(".").strip()
    connector = _copa_connector(ex["question"])
    prompt    = f"{premise} {connector}"
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    cands = []
    for choice in (ex["choice1"], ex["choice2"]):
        cands.append(tokenizer(f" {_copa_strip(choice)}.", add_special_tokens=False)["input_ids"])
    return {
        "prompt_ids": prompt_ids,
        "candidates": cands,
        "gold_idx":   int(ex["label"]),  # 0 → choice1, 1 → choice2
    }


# --------------------------------------------------------------------------
# SST-2 — binary sentiment classification
# --------------------------------------------------------------------------

def _sst2_prompt(ex) -> str:
    return f"{ex['sentence'].strip()} It was"

def sst2_format_train(ex, tokenizer):
    prompt = _sst2_prompt(ex)
    verb = " great" if ex["label"] == 1 else " terrible"
    full = prompt + verb
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
    while len(labels) < len(full_ids):
        labels.append(-100)
    labels = labels[:len(full_ids)]
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

def sst2_format_eval(ex, tokenizer):
    prompt = _sst2_prompt(ex)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    candidates = [
        tokenizer(" terrible", add_special_tokens=False)["input_ids"],
        tokenizer(" great", add_special_tokens=False)["input_ids"],
    ]
    return {
        "prompt_ids": prompt_ids,
        "candidates": candidates,
        "gold_idx": int(ex["label"]),  # 0 → terrible, 1 → great
    }


# --------------------------------------------------------------------------
# Registry — add a new SuperGLUE task here, no other code change needed.
# --------------------------------------------------------------------------

TASKS = {
    "multirc": TaskSpec("multirc", "multirc", multirc_format_train, multirc_format_eval),
    "copa":    TaskSpec("copa",    "copa",    copa_format_train,    copa_format_eval),
    "sst2":    TaskSpec("sst2",    "sst2",    sst2_format_train,    sst2_format_eval, dataset_path="glue"),
}


def load_task(task_name: str, num_train: int, seed: int) -> Tuple[TaskSpec, DatasetDict]:
    """Load a SuperGLUE task and subsample training to `num_train` examples (paper convention)."""
    if task_name not in TASKS:
        raise ValueError(f"Unknown task '{task_name}'. Registered: {list(TASKS)}")
    spec = TASKS[task_name]
    path = getattr(spec, "dataset_path", "super_glue")
    ds = load_dataset(path, spec.hf_subset)
    if num_train and len(ds["train"]) > num_train:
        ds["train"] = ds["train"].shuffle(seed=seed).select(range(num_train))
    return spec, ds
