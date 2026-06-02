"""Task-level dataset loaders for the MeZO/ZO-Muon paper-reproduction track.

For each task, this module:
    1. Downloads the right HF dataset (`datasets.load_dataset`).
    2. Applies the matching MeZO template from `data.mezo_templates`.
    3. Materialises a `formatted_text` column compatible with the existing
       `train.py` tokenisation pipeline (prompt prefix + candidate / gold).
    4. Optionally samples down to MeZO's `num_train=1000 / num_dev=500 /
       num_eval=1000` subsets — the headline numbers in all three papers
       are reported on these subsets, not on the full splits.

This file deliberately depends ONLY on `datasets` and `data.mezo_templates`;
it does not import torch or the optimizer code. That keeps the unit test
trivial: just load, format, count rows.

Usage from train.py:

    if dataset_config.get("template") == "mezo":
        from data.mezo_tasks import load_mezo_task
        task_name = dataset_config["task"]      # e.g. "sst2"
        ds = load_mezo_task(task_name, seed=seed,
                            num_train=1000, num_dev=500, num_eval=1000)
        # ds is a DatasetDict {train, validation, test} with a single
        # "formatted_text" column. answer_start is computed downstream.

Default sample sizes follow MeZO/ZO-Muon's protocol with the two known
overrides: CB and Copa cap dev at 100 because their full dev splits are
smaller than 500.
"""

from typing import Dict, Optional

from datasets import Dataset, DatasetDict, load_dataset

from .mezo_templates import Template, get_template


# Per-task dev-size overrides. MeZO/ZO-Muon repo behaviour.
_DEV_OVERRIDES: Dict[str, int] = {"cb": 100, "copa": 100}


def _format_classification(example: dict, tpl: Template) -> dict:
    """Render a classification example into MeZO's training format.

    For training, we emit prompt + gold-candidate. The training loop then
    masks the prompt tokens with -100 and predicts the candidate tokens.
    Per-candidate scoring at eval time is handled separately in metrics.
    """
    prompt, choices, ans_idx = tpl.format(example)
    answer = choices[ans_idx]
    return {
        "formatted_text": prompt + answer,
        "prompt": prompt,
        "answer": answer,
        "all_choices": list(choices),
        "answer_idx": int(ans_idx),
    }


def _format_generative(example: dict, tpl: Template) -> dict:
    prompt, gold = tpl.format(example)
    return {
        "formatted_text": prompt + " " + gold,
        "prompt": prompt,
        "answer": " " + gold,
    }


def _format_one(example: dict, tpl: Template) -> dict:
    if tpl.type == "classification":
        return _format_classification(example, tpl)
    if tpl.type == "generative":
        return _format_generative(example, tpl)
    raise ValueError(f"Unknown template type: {tpl.type}")


def _safe_select(ds: Dataset, n: Optional[int], seed: int) -> Dataset:
    if n is None or n >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(n))


def load_mezo_task(
    task_name: str,
    seed: int = 42,
    num_train: Optional[int] = 1000,
    num_dev: Optional[int] = 500,
    num_eval: Optional[int] = 1000,
) -> DatasetDict:
    """Return a DatasetDict {train, validation, test} for a MeZO task.

    Each split has a `formatted_text` column ready for the existing
    tokenisation step in train.py. Auxiliary columns (`prompt`, `answer`,
    `all_choices`, `answer_idx`) are kept for downstream eval-time scoring;
    train.py drops them after tokenisation.
    """
    tpl = get_template(task_name)

    # Load raw HF dataset.
    if len(tpl.dataset_id) == 1:
        raw = load_dataset(tpl.dataset_id[0])
    else:
        raw = load_dataset(*tpl.dataset_id)

    # Pick split names robustly. SuperGLUE has {train, validation, test}
    # but test is unlabelled; the MeZO protocol uses validation as test.
    train_split = raw.get("train")
    val_split   = raw.get("validation") or raw.get("validation_matched")
    if val_split is None:
        raise ValueError(f"No validation split for {task_name}")

    # Train -> sampled train + dev. Validation -> eval.
    dev_n = _DEV_OVERRIDES.get(task_name, num_dev)
    train_pool = train_split.shuffle(seed=seed)
    n_train = num_train if num_train is not None else len(train_pool)
    n_dev   = dev_n if dev_n is not None else 0
    train_sel = train_pool.select(range(min(n_train, len(train_pool))))
    dev_sel   = (train_pool.select(range(n_train, n_train + n_dev))
                 if (n_dev and n_train + n_dev <= len(train_pool))
                 else val_split)
    test_sel  = _safe_select(val_split, num_eval, seed)

    out = DatasetDict({
        "train":      train_sel.map(lambda e: _format_one(e, tpl)),
        "validation": dev_sel.map(lambda e: _format_one(e, tpl)),
        "test":       test_sel.map(lambda e: _format_one(e, tpl)),
    })
    return out


def list_available_tasks() -> list:
    """Returns the list of task names known to this module."""
    from .mezo_templates import TEMPLATES
    return sorted(TEMPLATES.keys())
