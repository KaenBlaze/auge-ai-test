"""Confidence scoring and abstention logic driven by evidence signals."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.config import Settings, get_settings
from src.generator import ABSTENTION_PHRASE
from src.retriever import RetrievedChunk

CITATION_RE = re.compile(r"\[source:\s*([^\]]+)\]", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+")

ABSTENTION_PHRASES = (
    "don't know based on the provided corpus",
    "don't have sufficient evidence",
    "do not have sufficient evidence",
    "cannot answer",
    "insufficient information",
    "not enough information",
)

SIGNAL_WEIGHTS = {
    "top_retrieval": 0.25,
    "top_rerank": 0.25,
    "supporting_chunks": 0.15,
    "fragment_agreement": 0.15,
    "citation_support": 0.20,
}


@dataclass
class ConfidenceResult:
    """Confidence assessment for a RAG answer."""

    confidence: float
    abstained: bool
    reason: str
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)

    @property
    def score(self) -> float:
        """Backward-compatible alias."""
        return self.confidence

    @property
    def should_abstain(self) -> bool:
        """Backward-compatible alias."""
        return self.abstained

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "abstained": self.abstained,
            "reason": self.reason,
            "signals": self.signals,
            "reasons": self.reasons,
        }


class ConfidenceScorer:
    """Decide whether to answer or abstain using retrieval and citation evidence."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def score(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> ConfidenceResult:
        """Compute confidence and abstention from evidence signals."""
        _ = query  # reserved for future entailment checks

        if not chunks:
            return abstention_result(0.0, "no_retrieved_chunks")

        supporting_chunks = [
            chunk
            for chunk in chunks
            if chunk.retrieval_score >= self.settings.similarity_threshold
        ]
        if not supporting_chunks:
            return abstention_result(
                0.0,
                "no_chunk_passes_similarity_threshold",
                signals={"supporting_chunks": 0.0},
            )

        top_retrieval = max(chunk.retrieval_score for chunk in chunks)
        top_rerank = max((chunk.rerank_score or 0.0) for chunk in chunks)
        supporting_ratio = min(
            len(supporting_chunks) / max(self.settings.min_supporting_chunks, 1),
            1.0,
        )
        fragment_agreement = _fragment_agreement(chunks)
        citation_support, has_citations = _citation_support(answer, chunks)

        signals = {
            "top_retrieval": round(top_retrieval, 4),
            "top_rerank": round(top_rerank, 4),
            "supporting_chunks": round(supporting_ratio, 4),
            "fragment_agreement": round(fragment_agreement, 4),
            "citation_support": round(citation_support, 4),
        }

        hard_reason = self._check_hard_abstention_rules(
            answer=answer,
            supporting_count=len(supporting_chunks),
            top_retrieval=top_retrieval,
            top_rerank=top_rerank,
            has_citations=has_citations,
            citation_support=citation_support,
            signals=signals,
        )
        if hard_reason:
            confidence = _weighted_confidence(signals)
            return abstention_result(
                confidence,
                hard_reason,
                signals=signals,
                reasons=_build_reason_lines(signals, hard_reason),
            )

        confidence = _weighted_confidence(signals)
        if (
            self.settings.abstain_on_low_confidence
            and confidence < self.settings.confidence_threshold
        ):
            reason = f"confidence_below_threshold_{self.settings.confidence_threshold:.2f}"
            return abstention_result(
                confidence,
                reason,
                signals=signals,
                reasons=_build_reason_lines(signals, reason),
            )

        return ConfidenceResult(
            confidence=round(confidence, 4),
            abstained=False,
            reason="sufficient_evidence",
            reasons=_build_reason_lines(signals, "sufficient_evidence"),
            signals=signals,
        )

    def _check_hard_abstention_rules(
        self,
        answer: str,
        supporting_count: int,
        top_retrieval: float,
        top_rerank: float,
        has_citations: bool,
        citation_support: float,
        signals: dict[str, float],
    ) -> str | None:
        if top_retrieval < self.settings.min_retrieval_score:
            return f"weak_top_retrieval_score_{top_retrieval:.2f}"

        if top_rerank < self.settings.min_rerank_score:
            return f"weak_top_rerank_score_{top_rerank:.2f}"

        if supporting_count < self.settings.min_supporting_chunks:
            return "insufficient_supporting_chunks"

        if _is_abstention_answer(answer):
            return "model_expressed_uncertainty"

        if self.settings.require_citations and not has_citations:
            return "answer_missing_citations"

        if has_citations and citation_support < self.settings.min_citation_support:
            return "answer_not_supported_by_citations"

        if signals["fragment_agreement"] < self.settings.min_fragment_agreement:
            return "low_fragment_agreement"

        return None


def abstention_result(
    confidence: float,
    reason: str,
    signals: dict[str, float] | None = None,
    reasons: list[str] | None = None,
) -> ConfidenceResult:
    """Build a standardized abstention result."""
    signal_map = signals or {}
    detail = reasons or _build_reason_lines(signal_map, reason)
    return ConfidenceResult(
        confidence=round(confidence, 4),
        abstained=True,
        reason=reason,
        reasons=detail,
        signals=signal_map,
    )


def _weighted_confidence(signals: dict[str, float]) -> float:
    return sum(SIGNAL_WEIGHTS[name] * signals.get(name, 0.0) for name in SIGNAL_WEIGHTS)


def _build_reason_lines(signals: dict[str, float], primary_reason: str) -> list[str]:
    lines = [f"decision={primary_reason}"]
    for name, value in signals.items():
        lines.append(f"{name}={value:.3f}")
    return lines


def _is_abstention_answer(answer: str) -> bool:
    answer_lower = answer.lower()
    if ABSTENTION_PHRASE.lower() in answer_lower:
        return True
    return any(phrase in answer_lower for phrase in ABSTENTION_PHRASES)


def _tokenize(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def _fragment_agreement(chunks: list[RetrievedChunk]) -> float:
    """Estimate agreement between retrieved fragments via lexical overlap."""
    if not chunks:
        return 0.0
    if len(chunks) == 1:
        return 1.0

    word_sets = [_tokenize(chunk.fragment) for chunk in chunks]
    overlaps: list[float] = []
    for i in range(len(word_sets)):
        for j in range(i + 1, len(word_sets)):
            union = word_sets[i] | word_sets[j]
            if not union:
                continue
            overlaps.append(len(word_sets[i] & word_sets[j]) / len(union))
    return sum(overlaps) / len(overlaps) if overlaps else 0.0


def _citation_support(answer: str, chunks: list[RetrievedChunk]) -> tuple[float, bool]:
    """Measure whether cited sources support the answer content."""
    cited_sources = [match.strip() for match in CITATION_RE.findall(answer)]
    if not cited_sources:
        return 0.0, False

    chunk_by_document = {chunk.document: chunk for chunk in chunks}
    valid_citations = [source for source in cited_sources if source in chunk_by_document]
    citation_precision = len(valid_citations) / len(cited_sources)

    cited_text = " ".join(chunk_by_document[source].fragment for source in valid_citations)
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return citation_precision, True

    supported_tokens = answer_tokens & _tokenize(cited_text)
    content_support = len(supported_tokens) / len(answer_tokens)
    combined = 0.6 * citation_precision + 0.4 * content_support
    return combined, True
