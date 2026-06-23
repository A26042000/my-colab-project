"""
metrics.py
==========
Scores model outputs against gold answers using REAL TASK METRICS.

WHY NOT "similarity to the full-compute answer"?
------------------------------------------------
A tempting shortcut is to call a reduced run "good" if its text matches the
full model's text. We deliberately DO NOT do this as the primary metric:
  - the full model is sometimes wrong, and
  - a reduced model can give a DIFFERENT but still CORRECT answer.
Scoring against the gold answer (exact match / MCQ accuracy) avoids rewarding
the model for merely copying the full run. This was a key reviewer point.

Two scoring kinds (set per item in data.py):
  "mcq"   -> did the model output the correct letter (A/B/C/D)?
  "exact" -> does the model's final answer match the gold answer?
"""

import re
from typing import List, Dict


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/articles, collapse spaces for fair compare."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_mcq_letter(output: str) -> str:
    """Pull the first standalone A/B/C/D/E from the model's output."""
    m = re.search(r"\b([A-E])\b", output.strip().upper())
    return m.group(1) if m else ""


def _extract_final_number(output: str) -> str:
    """
    For math/exact items, prefer a number after '####'; otherwise take the
    last number in the text (models often end with the answer).
    """
    after = re.search(r"####\s*(-?[\d,\.]+)", output)
    if after:
        return after.group(1).replace(",", "").rstrip(".")
    nums = re.findall(r"-?\d[\d,\.]*", output)
    return nums[-1].replace(",", "").rstrip(".") if nums else ""


def score_item(item: Dict, output: str) -> int:
    """
    Return 1 if the output is correct for this item, else 0.
    `item` is one of the uniform dicts from data.py; `output` is model text.
    """
    if item["kind"] == "mcq":
        return int(_extract_mcq_letter(output) == item["answer"].upper())

    # exact-match path (medium + hard)
    gold = item["answer"]
    # numeric gold -> compare extracted numbers
    if re.fullmatch(r"-?[\d,\.]+", gold):
        return int(_extract_final_number(output) == gold.replace(",", "").rstrip("."))
    # textual gold -> normalized containment / equality
    out_norm, gold_norm = _normalize(output), _normalize(gold)
    return int(gold_norm != "" and gold_norm in out_norm)


def suite_score(items: List[Dict], outputs: List[str]) -> float:
    """Average correctness (0..1) over a list of items and their outputs."""
    assert len(items) == len(outputs)
    if not items:
        return 0.0
    return sum(score_item(it, out) for it, out in zip(items, outputs)) / len(items)
