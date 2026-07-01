"""Confidence scoring and abstention logic driven by evidence signals."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.generator import ABSTENTION_PHRASE
from src.retriever import RetrievedChunk

CITATION_RE = re.compile(r"\[source:\s*([^\]]+)\]", re.IGNORECASE)
WORD_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "with",
    }
)

ABSTENTION_PHRASES = (
    "don't know based on the provided corpus",
    "don't have sufficient evidence",
    "do not have sufficient evidence",
    "cannot answer",
    "insufficient information",
    "not enough information",
)

SIGNAL_WEIGHTS = {
    "top_retrieval": 0.30,
    "top_rerank": 0.25,
    "supporting_chunks": 0.15,
    "fragment_agreement": 0.10,
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

    def assess_retrieval_evidence(self, chunks: list[RetrievedChunk]) -> ConfidenceResult | None:
        """Return an abstention result when retrieval evidence is insufficient to answer."""
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
        top_rerank = _top_rerank_score(chunks)
        signals = {
            "top_retrieval": round(top_retrieval, 4),
            "top_rerank": round(top_rerank, 4),
            "supporting_chunks": round(
                min(len(supporting_chunks) / max(self.settings.min_supporting_chunks, 1), 1.0),
                4,
            ),
            "fragment_agreement": round(_fragment_agreement(chunks), 4),
            "citation_support": 0.0,
        }

        if top_retrieval < self.settings.min_retrieval_score:
            return abstention_result(
                top_retrieval,
                f"weak_top_retrieval_score_{top_retrieval:.2f}",
                signals=signals,
            )

        if top_rerank < self.settings.min_rerank_score:
            return abstention_result(
                top_rerank,
                f"weak_top_rerank_score_{top_rerank:.2f}",
                signals=signals,
            )

        if len(supporting_chunks) < self.settings.min_supporting_chunks:
            return abstention_result(
                _weighted_confidence(signals),
                "insufficient_supporting_chunks",
                signals=signals,
            )

        return None

    def score(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
        *,
        structured_citations: bool = False,
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
        top_rerank = _top_rerank_score(chunks)
        citation_support, has_citations = _citation_support(
            answer,
            chunks,
            min_grounding_overlap=self.settings.min_citation_support,
        )

        signals = {
            "top_retrieval": round(top_retrieval, 4),
            "top_rerank": round(top_rerank, 4),
            "supporting_chunks": round(
                min(len(supporting_chunks) / max(self.settings.min_supporting_chunks, 1), 1.0),
                4,
            ),
            "fragment_agreement": round(_fragment_agreement(chunks), 4),
            "citation_support": round(citation_support, 4),
        }

        hard_reason = self._check_hard_abstention_rules(
            answer=answer,
            supporting_count=len(supporting_chunks),
            top_retrieval=top_retrieval,
            top_rerank=top_rerank,
            has_citations=has_citations,
            citation_support=citation_support,
            chunks=chunks,
            structured_citations=structured_citations,
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
        chunks: list[RetrievedChunk],
        structured_citations: bool = False,
    ) -> str | None:
        if top_retrieval < self.settings.min_retrieval_score:
            return f"weak_top_retrieval_score_{top_retrieval:.2f}"

        if top_rerank < self.settings.min_rerank_score:
            return f"weak_top_rerank_score_{top_rerank:.2f}"

        if supporting_count < self.settings.min_supporting_chunks:
            return "insufficient_supporting_chunks"

        if _is_abstention_answer(answer):
            return "model_expressed_uncertainty"

        if self.settings.require_citations and not has_citations and not structured_citations:
            return "answer_missing_citations"

        if has_citations and citation_support < self.settings.min_citation_support:
            inline = [source.strip() for source in CITATION_RE.findall(answer)]
            if inline and all(_source_in_chunks(source, chunks) for source in inline):
                return None
            if structured_citations:
                return None
            return "answer_not_supported_by_citations"

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


def _top_rerank_score(chunks: list[RetrievedChunk]) -> float:
    """Return a stable top rerank score using raw cross-encoder logits when available."""
    scores: list[float] = []
    for chunk in chunks:
        raw = chunk.metadata.get("raw_rerank_score")
        if raw is not None:
            scores.append(_sigmoid(float(raw)))
        elif chunk.rerank_score is not None:
            scores.append(float(chunk.rerank_score))
        else:
            scores.append(float(chunk.retrieval_score))
    return max(scores) if scores else 0.0


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _is_abstention_answer(answer: str) -> bool:
    answer_lower = answer.lower()
    if ABSTENTION_PHRASE.lower() in answer_lower:
        return True
    return any(phrase in answer_lower for phrase in ABSTENTION_PHRASES)


def _tokenize(text: str) -> set[str]:
    return {token for token in WORD_RE.findall(text.lower()) if token not in STOPWORDS}


def _fragment_agreement(chunks: list[RetrievedChunk]) -> float:
    """Measure how well each retrieved chunk aligns with the best-scoring chunk.

    Diverse sections from the same document (e.g. Protocols vs Sensors) are expected
    in RAG and should not be treated as conflicting evidence.
    """
    if not chunks:
        return 0.0
    if len(chunks) == 1:
        return 1.0

    best_chunk = max(chunks, key=lambda chunk: chunk.score)
    best_tokens = _tokenize(best_chunk.fragment)
    if not best_tokens:
        return 0.0

    overlaps: list[float] = []
    for chunk in chunks:
        if chunk.chunk_id == best_chunk.chunk_id:
            overlaps.append(1.0)
            continue
        chunk_tokens = _tokenize(chunk.fragment)
        union = best_tokens | chunk_tokens
        if not union:
            continue
        overlaps.append(len(best_tokens & chunk_tokens) / len(union))

    same_document = len({chunk.document for chunk in chunks}) == 1
    if same_document:
        return max(sum(overlaps) / len(overlaps), 0.5)

    return sum(overlaps) / len(overlaps)


def _citation_support(
    answer: str,
    chunks: list[RetrievedChunk],
    *,
    min_grounding_overlap: float = 0.20,
) -> tuple[float, bool]:
    """Measure whether the answer is supported by inline citations or retrieved fragments."""
    if _is_abstention_answer(answer):
        return 0.0, False

    cited_sources = [match.strip() for match in CITATION_RE.findall(answer)]
    if cited_sources:
        chunk_by_document = {chunk.document: chunk for chunk in chunks}
        valid_citations = [source for source in cited_sources if _source_in_chunks(source, chunks)]
        citation_precision = len(valid_citations) / len(cited_sources)
        cited_text = " ".join(
            chunk_by_document.get(source, next(iter(chunk_by_document.values()))).fragment
            for source in valid_citations
            if source in chunk_by_document
        )
        answer_tokens = _tokenize(answer)
        if not answer_tokens:
            return citation_precision, True
        supported_tokens = answer_tokens & _tokenize(cited_text)
        content_support = len(supported_tokens) / len(answer_tokens)
        combined = 0.6 * citation_precision + 0.4 * content_support
        return combined, True

    if not chunks:
        return 0.0, False

    support_text = " ".join(chunk.fragment for chunk in chunks)
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0, False
    support_tokens = _tokenize(support_text)
    shared_tokens = answer_tokens & support_tokens
    overlap = len(shared_tokens) / len(answer_tokens)
    grounded = len(shared_tokens) >= 2 and overlap >= min_grounding_overlap
    return overlap, grounded


def _source_in_chunks(source: str, chunks: list[RetrievedChunk]) -> bool:
    source_lower = source.strip().lower()
    for chunk in chunks:
        if chunk.document.lower() == source_lower:
            return True
        if chunk.source_id.lower() == source_lower:
            return True
        if Path(source_lower).stem == Path(chunk.document).stem:
            return True
    return False
