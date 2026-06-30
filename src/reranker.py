"""Cross-encoder reranking with similarity-score fallback."""

from __future__ import annotations

import logging
from typing import Protocol

from src.config import Settings, get_settings
from src.retriever import RetrievalResult

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
FALLBACK_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerProtocol(Protocol):
    def rerank(
        self,
        query: str,
        chunks: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]: ...


class SimilarityFallbackReranker:
    """Fallback reranker that ranks by dense retrieval similarity scores."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def rerank(
        self,
        query: str,
        chunks: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        k = top_k or self.settings.top_k_rerank
        ranked = sorted(chunks, key=lambda chunk: chunk.retrieval_score, reverse=True)[:k]
        return [
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                document=chunk.document,
                source_id=chunk.source_id,
                fragment=chunk.fragment,
                retrieval_score=chunk.retrieval_score,
                rerank_score=chunk.retrieval_score,
                metadata={**chunk.metadata, "reranker": "similarity_fallback"},
            )
            for chunk in ranked
        ]


class CrossEncoderReranker:
    """Local cross-encoder reranker using sentence-transformers."""

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        settings: Settings | None = None,
    ) -> None:
        self._requested_model = model_name
        self.model_name = model_name
        self.settings = settings or get_settings()
        self._model = None
        self._fallback = SimilarityFallbackReranker(self.settings)

    def _load(self):
        if self._model is not None:
            return self._model

        from sentence_transformers import CrossEncoder

        candidates = [self._requested_model]
        if self._requested_model != FALLBACK_RERANKER_MODEL:
            candidates.append(FALLBACK_RERANKER_MODEL)

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                self._model = CrossEncoder(candidate)
                self.model_name = candidate
                if candidate != self._requested_model:
                    logger.warning(
                        "Reranker model %s unavailable; using fallback %s",
                        self._requested_model,
                        candidate,
                    )
                return self._model
            except Exception as exc:
                last_error = exc
                logger.warning("Failed to load reranker model %s: %s", candidate, exc)

        raise RuntimeError(
            f"Could not load reranker model {self._requested_model!r} "
            f"or fallback {FALLBACK_RERANKER_MODEL!r}"
        ) from last_error

    def rerank(
        self,
        query: str,
        chunks: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        if not chunks:
            return []

        try:
            model = self._load()
        except Exception as exc:
            logger.warning("Cross-encoder unavailable (%s); using similarity fallback", exc)
            return self._fallback.rerank(query, chunks, top_k=top_k)

        k = top_k or self.settings.top_k_rerank
        pairs = [[query, chunk.fragment] for chunk in chunks]
        raw_scores = [float(score) for score in model.predict(pairs)]
        normalized_scores = _normalize_scores(raw_scores)

        scored = list(zip(chunks, raw_scores, normalized_scores))
        ranked = sorted(scored, key=lambda item: item[2], reverse=True)[:k]

        return [
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                document=chunk.document,
                source_id=chunk.source_id,
                fragment=chunk.fragment,
                retrieval_score=chunk.retrieval_score,
                rerank_score=norm_score,
                metadata={
                    **chunk.metadata,
                    "reranker": self.model_name,
                    "raw_rerank_score": raw_score,
                },
            )
            for chunk, raw_score, norm_score in ranked
        ]


class NoOpReranker:
    """Passthrough reranker when reranking is disabled."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def rerank(
        self,
        query: str,
        chunks: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        k = top_k or len(chunks)
        return [
            RetrievalResult(
                chunk_id=chunk.chunk_id,
                document=chunk.document,
                source_id=chunk.source_id,
                fragment=chunk.fragment,
                retrieval_score=chunk.retrieval_score,
                rerank_score=chunk.retrieval_score,
                metadata=chunk.metadata,
            )
            for chunk in chunks[:k]
        ]


def create_reranker(settings: Settings | None = None) -> RerankerProtocol:
    """Create the configured reranker, or a similarity fallback when disabled."""
    settings = settings or get_settings()
    if not settings.use_reranker:
        return NoOpReranker(settings)
    return CrossEncoderReranker(settings.reranker_model, settings)


# Backward-compatible alias.
Reranker = CrossEncoderReranker


def _normalize_scores(scores: list[float]) -> list[float]:
    """Normalize reranker scores to [0, 1] within the retrieved batch."""
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]
