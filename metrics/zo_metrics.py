"""Eval metrics for the MeZO-style paper-reproduction track.

Three metrics — `accuracy`, `f1`, `em` — covering every task in
`data.mezo_templates.TEMPLATES`. Pure Python / numpy; no torch dependency.

Used by the evaluation loop when `config.dataset.template == "mezo"`. For
classification tasks the trainer scores each candidate completion by per-
token log-likelihood and emits a `pred_idx`; we compare to `answer_idx`.
For generative tasks we get a free-form string prediction and compare to
the gold answer with F1/EM.
"""

import re
import string
from collections import Counter
from typing import Iterable, List


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def accuracy(pred_idx: Iterable[int], gold_idx: Iterable[int]) -> float:
    p = list(pred_idx)
    g = list(gold_idx)
    assert len(p) == len(g), f"length mismatch {len(p)} vs {len(g)}"
    if not p:
        return 0.0
    return sum(int(a == b) for a, b in zip(p, g)) / len(p)


# ---------------------------------------------------------------------------
# Generative — SQuAD/DROP-style F1 and EM.
# These follow the official SQuAD evaluation script's tokenisation.
# ---------------------------------------------------------------------------
def _normalize(s: str) -> str:
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def _f1(pred: str, gold: str) -> float:
    pt = _normalize(pred).split()
    gt = _normalize(gold).split()
    if not pt or not gt:
        return float(pt == gt)
    common = Counter(pt) & Counter(gt)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pt)
    r = num_same / len(gt)
    return 2 * p * r / (p + r)


def _em(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def f1(preds: Iterable[str], golds: Iterable[str]) -> float:
    p = list(preds)
    g = list(golds)
    assert len(p) == len(g)
    if not p:
        return 0.0
    return sum(_f1(a, b) for a, b in zip(p, g)) / len(p)


def em(preds: Iterable[str], golds: Iterable[str]) -> float:
    p = list(preds)
    g = list(golds)
    assert len(p) == len(g)
    if not p:
        return 0.0
    return sum(_em(a, b) for a, b in zip(p, g)) / len(p)


METRIC_FNS = {
    "accuracy": accuracy,
    "f1":       f1,
    "em":       em,
}


def score(metric: str, preds, golds) -> float:
    if metric not in METRIC_FNS:
        raise KeyError(f"unknown metric '{metric}'. Known: {sorted(METRIC_FNS)}")
    return METRIC_FNS[metric](preds, golds)
