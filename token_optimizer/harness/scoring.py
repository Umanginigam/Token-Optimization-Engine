"""Scoring utilities for token optimization."""
"""Objective scoring for extractive QA — the canonical SQuAD metrics.

Exact Match (EM) and token-level F1, computed after normalizing case,
punctuation, and articles. No LLM-as-judge needed, so it's free and
deterministic. This is the part that MUST be correct — everything else
in the harness trusts these numbers, so it's a faithful implementation
of the standard SQuAD scoring functions.
"""

import re
import string
from collections import Counter
from typing import Callable, List


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation, remove articles, collapse whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    # If either is empty, F1 is 1 only if both are empty.
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def metric_max_over_ground_truths(
    metric_fn: Callable[[str, str], float],
    prediction: str,
    ground_truths: List[str],
) -> float:
    """SQuAD allows multiple acceptable answers; take the best match."""
    if not ground_truths:
        return 0.0
    return max(metric_fn(prediction, gt) for gt in ground_truths)