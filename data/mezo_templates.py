"""MeZO-style prompt templates for paper-faithful zeroth-order fine-tuning.

These templates match the prompt wording used in the MeZO codebase
(https://github.com/princeton-nlp/MeZO/blob/main/large_models/templates.py)
and the ZO-Muon fork
(https://github.com/OPTML-Group/ZO-Muon/blob/main/llm/templates.py),
which all three of the papers we are reproducing (ZO-Muon, ConMeZO, FZOO)
inherit. The wording is the #1 reproduction-killer for ZO results — do not
paraphrase.

Two template categories:
    - classification: returns (prompt_str, list_of_candidate_completions,
      answer_idx). The training loop computes per-candidate log-likelihood
      and picks the argmax. Used for SST2, RTE, BoolQ, WIC, CB, MultiRC,
      WSC, Copa.
    - generative: returns (prompt_str, gold_answer_str). Used for SQuAD,
      DROP, ReCoRD.

Each template is a small dict with the keys:
    type          : "classification" or "generative"
    needs         : list of dataset columns that must be present
    format        : callable example -> (prompt, choices, answer_idx) or
                    (prompt, gold_answer)
    dataset_id    : HF datasets path, e.g. ("glue", "sst2")
    eval_metric   : "accuracy" | "f1" | "em"
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple, Union


@dataclass
class Template:
    type: str                       # "classification" or "generative"
    dataset_id: Tuple[str, ...]     # args to load_dataset, e.g. ("glue", "sst2")
    needs: Tuple[str, ...]          # required column names
    eval_metric: str                # "accuracy" / "f1" / "em"
    format: Callable                # example -> (prompt, choices, ans_idx)
                                    #          or (prompt, gold_answer)


# ---------------------------------------------------------------------------
# Classification templates
# ---------------------------------------------------------------------------
def _sst2(ex):
    text = ex["sentence"].strip()
    prompt = f"{text} It was"
    choices = [" terrible", " great"]   # leading space matters for tokenisation
    return prompt, choices, int(ex["label"])


def _rte(ex):
    # MeZO wording: "{premise} Does this mean that \"{hypothesis}\" is true?
    # Yes or No?". Label 0 = entailment -> "Yes", label 1 = not_entailment -> "No".
    prem = ex["premise"].strip()
    hyp  = ex["hypothesis"].strip()
    prompt = f"{prem} Does this mean that \"{hyp}\" is true? Yes or No?"
    choices = [" Yes", " No"]
    return prompt, choices, int(ex["label"])


def _boolq(ex):
    passage = ex["passage"].strip()
    question = ex["question"].strip()
    prompt = f"{passage}\nQuestion: {question}?\nAnswer:"
    choices = [" No", " Yes"]
    return prompt, choices, int(ex["label"])


def _wic(ex):
    word = ex["word"]
    s1 = ex["sentence1"].strip()
    s2 = ex["sentence2"].strip()
    prompt = (f"{s1}\n{s2}\nQuestion: Is the word '{word}' used in the same "
              f"way in the two sentences above?\nAnswer:")
    choices = [" No", " Yes"]
    return prompt, choices, int(ex["label"])


def _cb(ex):
    # MeZO wording: "Suppose {premise} Can we infer that \"{hypothesis}\"?
    # Yes, No, or Maybe?". Labels: 0=entailment->Yes, 1=contradiction->No,
    # 2=neutral->Maybe.
    prem = ex["premise"].strip()
    hyp  = ex["hypothesis"].strip()
    prompt = (f"Suppose {prem} Can we infer that \"{hyp}\"? "
              f"Yes, No, or Maybe?")
    choices = [" Yes", " No", " Maybe"]
    return prompt, choices, int(ex["label"])


def _copa(ex):
    premise = ex["premise"].strip().rstrip(".")
    q = ex["question"]
    connector = "because" if q == "cause" else "so"
    prompt = f"{premise} {connector}"
    choices = [" " + ex["choice1"].strip(), " " + ex["choice2"].strip()]
    return prompt, choices, int(ex["label"])


def _multirc(ex):
    para = ex["paragraph"].strip()
    q    = ex["question"].strip()
    ans  = ex["answer"].strip()
    prompt = (f"{para}\nQuestion: {q}\nI found this answer \"{ans}\". Is "
              f"that correct? Yes or No?\nAnswer:")
    choices = [" No", " Yes"]
    return prompt, choices, int(ex["label"])


def _wsc(ex):
    text = ex["text"].strip()
    span1 = ex["span1_text"]
    span2 = ex["span2_text"]
    prompt = (f"{text}\nIn the passage above, does the pronoun \"{span2}\" "
              f"refer to \"{span1}\"?\nAnswer:")
    choices = [" No", " Yes"]
    return prompt, choices, int(ex["label"])


# ---------------------------------------------------------------------------
# Generative templates
# ---------------------------------------------------------------------------
def _squad(ex):
    ctx = ex["context"].strip()
    q   = ex["question"].strip()
    answers = ex["answers"]["text"]
    gold = answers[0] if answers else ""
    prompt = f"Title: {ex.get('title', '').strip()}\n\nContext: {ctx}\n\nQuestion: {q}\n\nAnswer:"
    return prompt, gold


def _drop(ex):
    ctx = ex["passage"].strip()
    q   = ex["question"].strip()
    answers = ex["answers_spans"]["spans"] if "answers_spans" in ex else []
    gold = answers[0] if answers else ""
    prompt = f"Passage: {ctx}\nQuestion: {q}\nAnswer:"
    return prompt, gold


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Dataset IDs use the canonical namespaced form required by recent
# `datasets` releases (which refuse bare "glue" / "super_glue" / "squad").
TEMPLATES = {
    "sst2": Template(
        type="classification", dataset_id=("nyu-mll/glue", "sst2"),
        needs=("sentence", "label"), eval_metric="accuracy", format=_sst2,
    ),
    "rte": Template(
        type="classification", dataset_id=("aps/super_glue", "rte"),
        needs=("premise", "hypothesis", "label"),
        eval_metric="accuracy", format=_rte,
    ),
    "boolq": Template(
        type="classification", dataset_id=("aps/super_glue", "boolq"),
        needs=("passage", "question", "label"),
        eval_metric="accuracy", format=_boolq,
    ),
    "wic": Template(
        type="classification", dataset_id=("aps/super_glue", "wic"),
        needs=("word", "sentence1", "sentence2", "label"),
        eval_metric="accuracy", format=_wic,
    ),
    "cb": Template(
        type="classification", dataset_id=("aps/super_glue", "cb"),
        needs=("premise", "hypothesis", "label"),
        eval_metric="accuracy", format=_cb,
    ),
    "copa": Template(
        type="classification", dataset_id=("aps/super_glue", "copa"),
        needs=("premise", "choice1", "choice2", "question", "label"),
        eval_metric="accuracy", format=_copa,
    ),
    "multirc": Template(
        type="classification", dataset_id=("aps/super_glue", "multirc"),
        needs=("paragraph", "question", "answer", "label"),
        eval_metric="f1", format=_multirc,
    ),
    "wsc": Template(
        type="classification", dataset_id=("aps/super_glue", "wsc.fixed"),
        needs=("text", "span1_text", "span2_text", "label"),
        eval_metric="accuracy", format=_wsc,
    ),
    "squad": Template(
        type="generative", dataset_id=("rajpurkar/squad",),
        needs=("context", "question", "answers"),
        eval_metric="f1", format=_squad,
    ),
    "drop": Template(
        type="generative", dataset_id=("ucinlp/drop",),
        needs=("passage", "question", "answers_spans"),
        eval_metric="f1", format=_drop,
    ),
}


def get_template(name: str) -> Template:
    """Look up a MeZO-style template by task name. Raises KeyError on miss."""
    if name not in TEMPLATES:
        raise KeyError(
            f"Unknown MeZO template '{name}'. Known: {sorted(TEMPLATES)}")
    return TEMPLATES[name]
