"""Confidence scoring and abstention logic."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings, get_settings
from src.retriever import RetrievedChunk


@dataclass
class ConfidenceResult:
    """Confidence assessment for a RAG answer."""

    score: float
    should_abstain: bool
    reasons: list[str]


class ConfidenceScorer:
    """Compute answer confidence from retrieval signals.

    TODO: Add LLM-based self-consistency or entailment checks.
    TODO: Calibrate thresholds on golden evaluation set.
    TODO: Per-citation confidence breakdown.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def score(
        self,
        query: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> ConfidenceResult:
        """Derive a confidence score from retrieval quality and answer heuristics."""
        reasons: list[str] = []
        component_scores: list[float] = []

        # Retrieval strength
        if chunks:
            top_score = max(c.score for c in chunks)
            avg_score = sum(c.score for c in chunks) / len(chunks)
            retrieval_conf = 0.6 * top_score + 0.4 * avg_score
            component_scores.append(retrieval_conf)
            reasons.append(f"retrieval_confidence={retrieval_conf:.3f}")
        else:
            component_scores.append(0.0)
            reasons.append("no_chunks_retrieved")

        # Coverage: more chunks above threshold → higher confidence
        coverage = min(len(chunks) / max(self.settings.top_k_rerank, 1), 1.0)
        component_scores.append(coverage * 0.5)
        reasons.append(f"context_coverage={coverage:.3f}")

        # Abstention phrase detection
        abstention_phrases = [
            "don't know based on the provided corpus",
            "don't have sufficient evidence",
            "do not have sufficient evidence",
            "cannot answer",
            "insufficient information",
            "not enough information",
        ]
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in abstention_phrases):
            component_scores.append(0.0)
            reasons.append("model_expressed_uncertainty")
        else:
            component_scores.append(0.7)

        final_score = sum(component_scores) / len(component_scores)
        should_abstain = (
            self.settings.abstain_on_low_confidence
            and final_score < self.settings.confidence_threshold
        )
        if should_abstain:
            reasons.append(f"below_threshold={self.settings.confidence_threshold}")

        return ConfidenceResult(
            score=round(final_score, 4),
            should_abstain=should_abstain,
            reasons=reasons,
        )
