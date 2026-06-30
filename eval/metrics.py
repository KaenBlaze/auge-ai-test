"""Evaluation metrics for the RAG system."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class MetricResult:
    """A single metric name and value."""

    name: str
    value: float
    details: dict | None = None


def exact_match(predicted: str, expected: str) -> float:
    """1.0 if normalized strings match exactly, else 0.0."""
    return 1.0 if _normalize(predicted) == _normalize(expected) else 0.0


def contains_answer(predicted: str, expected: str) -> float:
    """1.0 if expected answer appears in prediction (case-insensitive)."""
    return 1.0 if _normalize(expected) in _normalize(predicted) else 0.0


def citation_precision(
    predicted_sources: list[str],
    expected_sources: list[str],
) -> float:
    """Fraction of predicted citations that are in the expected set."""
    if not predicted_sources:
        return 0.0
    expected_set = set(expected_sources)
    hits = sum(1 for s in predicted_sources if s in expected_set)
    return hits / len(predicted_sources)


def citation_recall(
    predicted_sources: list[str],
    expected_sources: list[str],
) -> float:
    """Fraction of expected citations found in predictions."""
    if not expected_sources:
        return 1.0
    predicted_set = set(predicted_sources)
    hits = sum(1 for s in expected_sources if s in predicted_set)
    return hits / len(expected_sources)


def abstention_accuracy(predicted_abstained: bool, should_abstain: bool) -> float:
    """1.0 if abstention decision matches ground truth."""
    return 1.0 if predicted_abstained == should_abstain else 0.0


def rouge_l(predicted: str, expected: str) -> float:
    """ROUGE-L F1 score between prediction and expected answer.

    TODO: Use rouge-score library for full ROUGE-1/2/L suite.
    """
    pred_tokens = _normalize(predicted).split()
    exp_tokens = _normalize(expected).split()
    if not pred_tokens or not exp_tokens:
        return 0.0

    lcs_len = _lcs_length(pred_tokens, exp_tokens)
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def aggregate_metrics(results: list[MetricResult]) -> dict[str, float]:
    """Group metric results by name and compute means."""
    grouped: dict[str, list[float]] = {}
    for r in results:
        grouped.setdefault(r.name, []).append(r.value)
    return {name: sum(vals) / len(vals) for name, vals in grouped.items()}
