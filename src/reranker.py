"""Cross-encoder reranking for retrieved chunks."""

from __future__ import annotations

from src.config import Settings, get_settings
from src.retriever import RetrievedChunk


class Reranker:
    """Local cross-encoder reranker (sentence-transformers CrossEncoder)."""

    def __init__(self, model_name: str, settings: Settings | None = None) -> None:
        self.model_name = model_name
        self.settings = settings or get_settings()
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Rerank chunks by cross-encoder relevance scores.

        TODO: Normalize cross-encoder scores to [0, 1] for downstream confidence.
        TODO: Add optional MMR diversification to reduce redundant chunks.
        """
        if not chunks:
            return []

        k = top_k or self.settings.top_k_rerank
        model = self._load()
        pairs = [[query, c.text] for c in chunks]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        return [
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                source=chunk.source,
                score=float(score),
                metadata={**chunk.metadata, "retrieval_score": chunk.score},
            )
            for chunk, score in ranked
        ]


class NoOpReranker:
    """Passthrough reranker when reranking is disabled."""

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k or len(chunks)
        return chunks[:k]
