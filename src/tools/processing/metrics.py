"""Lightweight evaluation helpers for pipeline regression tests."""
from __future__ import annotations

import re
from typing import Iterable, Tuple


def _tokenize(text: str) -> list[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [token for token in normalized.split() if token]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute word error rate using a simple Levenshtein distance.

    Returns 0.0 when the reference is empty.
    """

    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)

    if not ref_tokens:
        return 0.0

    rows = len(ref_tokens) + 1
    cols = len(hyp_tokens) + 1
    dist = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dist[i][0] = i
    for j in range(cols):
        dist[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,  # deletion
                dist[i][j - 1] + 1,  # insertion
                dist[i - 1][j - 1] + cost,  # substitution
            )

    return dist[-1][-1] / float(len(ref_tokens))


def tag_precision_recall(expected: Iterable[str], predicted: Iterable[str]) -> Tuple[float, float]:
    """Return precision/recall for simple tag sets."""

    expected_set = {tag.lower() for tag in expected}
    predicted_set = {tag.lower() for tag in predicted}

    if not predicted_set:
        return (0.0, 0.0)

    true_positives = len(expected_set & predicted_set)
    precision = true_positives / len(predicted_set)
    recall = true_positives / len(expected_set) if expected_set else 0.0
    return (precision, recall)


def title_relevance_score(expected_title: str, generated_title: str) -> float:
    """Heuristic overlap score between expected and generated titles."""

    expected_tokens = set(_tokenize(expected_title))
    generated_tokens = set(_tokenize(generated_title))

    if not expected_tokens or not generated_tokens:
        return 0.0

    overlap = len(expected_tokens & generated_tokens)
    return overlap / float(len(expected_tokens))


__all__ = [
    "word_error_rate",
    "tag_precision_recall",
    "title_relevance_score",
]
