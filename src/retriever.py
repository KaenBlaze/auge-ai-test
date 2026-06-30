"""Retrieve relevant chunks for a query."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import Settings, get_settings
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    """A chunk returned by the retriever with relevance score."""

    id: str
    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


class Retriever:
    """Embedding-based dense retriever over the local vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        settings: Settings | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.settings = settings or get_settings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for *query*.

        TODO: Add hybrid retrieval (BM25 + dense) for keyword-heavy queries.
        TODO: Filter by metadata (date range, document type, etc.).
        """
        k = top_k or self.settings.top_k_retrieve
        query_embedding = self.embedding_model.embed_query(query)
        raw_hits = self.vector_store.query(query_embedding, top_k=k)

        chunks = [
            RetrievedChunk(
                id=hit["id"],
                text=hit["text"],
                source=hit["metadata"].get("source", "unknown"),
                score=hit["score"],
                metadata=hit["metadata"],
            )
            for hit in raw_hits
            if hit["score"] >= self.settings.similarity_threshold
        ]
        return chunks
