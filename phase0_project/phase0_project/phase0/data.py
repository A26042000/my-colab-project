"""
data.py
=======
Loads the easy -> hard workload suites and exposes them in ONE uniform shape.

WHY THIS MATTERS
----------------
The difficulty signal can only prove itself if requests genuinely differ in
difficulty. So we deliberately pull three tiers:

    EASY    : multiple-choice general knowledge   (metric: accuracy)
    MEDIUM  : short extractive / numeric reasoning (metric: exact match)
    HARD    : grade-school math word problems      (metric: exact match)

Each suite is normalised into a list of dicts with the SAME keys, so the rest
of the pipeline never has to care which dataset an item came from:

    {
      "suite":   "easy" | "medium" | "hard",
      "prompt":  str,        # what we feed the model
      "answer":  str,        # the gold answer (for scoring)
      "kind":    "mcq" | "exact",   # how to score it (see metrics.py)
    }

Beginner note: we keep everything tiny (a few dozen items per suite) so a free
Colab GPU can run the whole thing in minutes. Scale up later.
"""

import random
import re
from typing import List, Dict

from datasets import load_dataset


def _clean(text: str) -> str:
    """Collapse whitespace so prompts/answers compare cleanly."""
    return re.sub(r"\s+", " ", text).strip()


def load_easy(n: int) -> List[Dict]:
    """
    EASY: ARC-Easy style multiple-choice science questions.
    Many of these need little reasoning -> high compute tolerance expected.
    """
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    items = []
    for row in ds:
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        # Build a clean A/B/C/D prompt.
        lettered = [f"{chr(65+i)}. {c}" for i, c in enumerate(choices)]
        prompt = (
            "Answer with a single letter.\n"
            f"Question: {_clean(row['question'])}\n"
            + "\n".join(lettered)
            + "\nAnswer:"
        )
        # Gold answer as a letter (A/B/C/D).
        try:
            gold_letter = chr(65 + labels.index(row["answerKey"]))
        except ValueError:
            continue  # skip malformed rows
        items.append(
            {"suite": "easy", "prompt": prompt, "answer": gold_letter, "kind": "mcq"}
        )
        if len(items) >= n:
            break
    return items


def load_medium(n: int) -> List[Dict]:
    """
    MEDIUM: short, factual open-domain questions (TriviaQA, rc.nocontext).
    Needs recall but rarely long multi-step reasoning.
    """
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    items = []
    for row in ds:
        ans = row["answer"]["value"]
        prompt = (
            "Answer concisely with just the answer.\n"
            f"Question: {_clean(row['question'])}\nAnswer:"
        )
        items.append(
            {"suite": "medium", "prompt": prompt, "answer": _clean(ans), "kind": "exact"}
        )
        if len(items) >= n:
            break
    return items


def load_hard(n: int) -> List[Dict]:
    """
    HARD: GSM8K grade-school math word problems (multi-step reasoning).
    Low compute tolerance expected -> the signal should flag these as 'hard'.
    """
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = []
    for row in ds:
        # GSM8K gold answers end with '#### <number>'.
        m = re.search(r"####\s*(.+)\s*$", row["answer"])
        gold = _clean(m.group(1)) if m else _clean(row["answer"])
        prompt = (
            "Solve the problem. Give ONLY the final numeric answer after '####'.\n"
            f"Problem: {_clean(row['question'])}\n"
        )
        items.append(
            {"suite": "hard", "prompt": prompt, "answer": gold, "kind": "exact"}
        )
        if len(items) >= n:
            break
    return items


def load_all_suites(n_per_suite: int, seed: int = 0) -> List[Dict]:
    """
    Load and concatenate all three suites, shuffled together.

    Returns one flat list of uniform item-dicts. If a dataset fails to load
    (network/gating), we surface a clear error rather than silently skipping,
    because a missing difficulty tier would invalidate the whole experiment.
    """
    random.seed(seed)
    items: List[Dict] = []
    items += load_easy(n_per_suite)
    items += load_medium(n_per_suite)
    items += load_hard(n_per_suite)
    random.shuffle(items)
    return items
