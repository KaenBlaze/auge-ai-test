"""Evaluation metrics for the RAG system."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CITATION_RE = re.compile(r"\[source:\s*([^\]]+)\]", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+")

ABSTENTION_MARKERS = (
    "don't know based on the provided corpus",
    "don't have sufficient evidence",
    "do not have sufficient evidence",
)


@dataclass
class MetricResult:
    """A single metric name and value."""

    name: str
    value: float
    details: dict[str, Any] | None = None


@dataclass
class ExampleEvaluation:
    """Per-example evaluation record."""

    example_id: str
    question: str
    answerable: bool
    abstained: bool
    confidence: float
    metrics: dict[str, float] = field(default_factory=dict)
    is_correct: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def is_answerable(example: dict[str, Any]) -> bool:
    """Return True when the example should receive an answer."""
    if "respondible" in example:
        return bool(example["respondible"])
    return not bool(example.get("should_abstain", False))


def retrieval_accuracy(expected_sources: list[str], retrieved_documents: list[str]) -> float:
    """Fraction of expected sources present in retrieved documents."""
    if not expected_sources:
        return 1.0
    retrieved_set = set(retrieved_documents)
    hits = sum(1 for source in expected_sources if _source_matches_any(source, retrieved_set))
    return hits / len(expected_sources)


def citation_coverage(answerable: bool, answer: str, citations: list[dict[str, str]]) -> float:
    """1.0 when an answerable example includes at least one citation."""
    if not answerable:
        return 1.0
    inline_citations = CITATION_RE.findall(answer)
    structured_citations = [c for c in citations if c.get("document") or c.get("source_id")]
    return 1.0 if inline_citations or structured_citations else 0.0


def citation_correctness(
    answer: str,
    citations: list[dict[str, str]],
    expected_sources: list[str],
) -> float:
    """Fraction of expected sources that are correctly cited by document or source_id."""
    if not expected_sources:
        return 1.0 if not _extract_cited_sources(answer, citations) else 0.0

    cited_sources = _extract_cited_sources(answer, citations)
    if not cited_sources:
        return 0.0

    hits = sum(1 for expected in expected_sources if _source_matches_any(expected, cited_sources))
    return hits / len(expected_sources)


def groundedness(answer: str, citations: list[dict[str, str]], expected_answer: str = "") -> float:
    """Simple faithfulness score based on answer token overlap with cited fragments."""
    if _is_abstention_answer(answer):
        return 1.0

    citation_text = " ".join(citation.get("fragment", "") for citation in citations)
    cited_inline = " ".join(CITATION_RE.sub("", answer))
    support_text = f"{citation_text} {cited_inline}".strip()
    if not support_text:
        return 0.0

    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0

    support_tokens = _tokenize(support_text)
    overlap = len(answer_tokens & support_tokens) / len(answer_tokens)

    if expected_answer:
        expected_tokens = _tokenize(expected_answer)
        if expected_tokens:
            expected_overlap = len(answer_tokens & expected_tokens) / len(expected_tokens)
            return 0.6 * overlap + 0.4 * expected_overlap
    return overlap


def hallucination(
    answerable: bool,
    abstained: bool,
    answer: str,
    citations: list[dict[str, str]],
) -> float:
    """1.0 when the system answers without permission or without citation support."""
    if not answerable:
        return 0.0 if abstained else 1.0
    if abstained:
        return 0.0
    return 0.0 if citation_coverage(answerable, answer, citations) == 1.0 else 1.0


def correct_abstention(abstained: bool, answerable: bool) -> float:
    """1.0 when the system correctly abstains on an unanswerable example."""
    if answerable:
        return 0.0
    return 1.0 if abstained else 0.0


def false_abstention(abstained: bool, answerable: bool) -> float:
    """1.0 when the system abstains on an answerable example."""
    if not answerable:
        return 0.0
    return 1.0 if abstained else 0.0


def example_is_correct(
    answerable: bool,
    abstained: bool,
    answer: str,
    expected_answer: str,
    citations: list[dict[str, str]],
    expected_sources: list[str],
) -> bool:
    """Determine whether the pipeline behavior matches the golden label."""
    if not answerable:
        return abstained
    if abstained:
        return False
    if expected_answer and not contains_answer(answer, expected_answer):
        return False
    if citation_coverage(answerable, answer, citations) < 1.0:
        return False
    if expected_sources and citation_correctness(answer, citations, expected_sources) < 1.0:
        return False
    return True


def confidence_calibration(
    labeled_results: list[tuple[float, float]],
    num_bins: int = 5,
) -> dict[str, Any]:
    """Compare mean confidence to empirical correctness rate per confidence bin."""
    if not labeled_results:
        return {"bins": [], "ece": 0.0, "calibration_score": 1.0}

    bins: list[dict[str, Any]] = []
    total = len(labeled_results)
    ece = 0.0

    for index in range(num_bins):
        low = index / num_bins
        high = (index + 1) / num_bins
        bucket = [
            (confidence, correct)
            for confidence, correct in labeled_results
            if (confidence >= low and (confidence < high or (high == 1.0 and confidence <= high)))
        ]
        if not bucket:
            continue
        mean_confidence = sum(conf for conf, _ in bucket) / len(bucket)
        accuracy = sum(correct for _, correct in bucket) / len(bucket)
        weight = len(bucket) / total
        ece += weight * abs(mean_confidence - accuracy)
        bins.append(
            {
                "bin": f"{low:.1f}-{high:.1f}",
                "count": len(bucket),
                "mean_confidence": round(mean_confidence, 4),
                "accuracy": round(accuracy, 4),
            }
        )

    calibration_score = max(0.0, 1.0 - ece)
    return {
        "bins": bins,
        "ece": round(ece, 4),
        "calibration_score": round(calibration_score, 4),
    }


def contains_answer(predicted: str, expected: str) -> bool:
    """Return True when the expected answer appears in the prediction."""
    return _normalize(expected) in _normalize(predicted)


def exact_match(predicted: str, expected: str) -> float:
    """1.0 if normalized strings match exactly, else 0.0."""
    return 1.0 if _normalize(predicted) == _normalize(expected) else 0.0


def aggregate_metrics(results: list[MetricResult]) -> dict[str, float]:
    """Group metric results by name and compute means."""
    grouped: dict[str, list[float]] = {}
    for result in results:
        grouped.setdefault(result.name, []).append(result.value)
    return {name: sum(values) / len(values) for name, values in grouped.items()}


def summarize_example_metrics(example_eval: ExampleEvaluation) -> list[MetricResult]:
    """Flatten an example evaluation into metric results."""
    return [MetricResult(name, value) for name, value in example_eval.metrics.items()]


def _extract_cited_sources(answer: str, citations: list[dict[str, str]]) -> set[str]:
    cited = {match.strip() for match in CITATION_RE.findall(answer)}
    for citation in citations:
        if citation.get("document"):
            cited.add(citation["document"])
        if citation.get("source_id"):
            cited.add(citation["source_id"])
    return {source for source in cited if source}


def _source_matches_any(expected: str, cited_sources: set[str]) -> bool:
    expected_norm = expected.strip().lower()
    expected_stem = Path(expected_norm).stem
    for cited in cited_sources:
        cited_norm = cited.strip().lower()
        cited_stem = Path(cited_norm).stem
        if cited_norm == expected_norm:
            return True
        if cited_stem == expected_stem:
            return True
        if cited_norm.endswith(expected_norm) or expected_norm.endswith(cited_norm):
            return True
    return False


def _is_abstention_answer(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(marker in answer_lower for marker in ABSTENTION_MARKERS)


def _tokenize(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def _normalize(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"\s+", " ", text)
