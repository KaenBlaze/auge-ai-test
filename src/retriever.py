"""Retrieve relevant chunks for a query."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import Settings, get_settings
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


@dataclass
class RetrievalResult:
    """A chunk returned by retrieval and optional reranking."""

    chunk_id: str
    document: str
    source_id: str
    fragment: str
    retrieval_score: float
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Alias for downstream citation and vector-store compatibility."""
        return self.chunk_id

    @property
    def text(self) -> str:
        """Alias for downstream generation."""
        return self.fragment

    @property
    def source(self) -> str:
        """Alias for downstream citation formatting."""
        return self.document

    @property
    def score(self) -> float:
        """Primary relevance score: rerank score when present, else retrieval score."""
        return self.rerank_score if self.rerank_score is not None else self.retrieval_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "source_id": self.source_id,
            "fragment": self.fragment,
            "retrieval_score": self.retrieval_score,
            "rerank_score": self.rerank_score if self.rerank_score is not None else 0.0,
            "metadata": self.metadata,
        }


# Backward-compatible alias used by the pipeline and generator.
RetrievedChunk = RetrievalResult


class Retriever:
    """Embedding-based dense retriever over the local FAISS vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        settings: Settings | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.settings = settings or get_settings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Retrieve top-k chunks for *query* from the FAISS index."""
        k = top_k or self.settings.top_k_retrieve
        query_embedding = self.embedding_model.embed_query(query)
        raw_hits = self.vector_store.query(query_embedding, top_k=k)

        results: list[RetrievalResult] = []
        for hit in raw_hits:
            if hit["score"] < self.settings.similarity_threshold:
                continue
            metadata = hit.get("metadata", {})
            results.append(
                RetrievalResult(
                    chunk_id=hit["id"],
                    document=metadata.get("document", metadata.get("source", "unknown")),
                    source_id=metadata.get("source_id", "unknown"),
                    fragment=hit["text"],
                    retrieval_score=float(hit["score"]),
                    rerank_score=None,
                    metadata=metadata,
                )
            )
        return results

    def retrieve_as_dicts(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """Retrieve chunks and return the standard serializable output format."""
        return [result.to_dict() for result in self.retrieve(query, top_k=top_k)]


def retrieve_evidence(
    query: str,
    retriever: Retriever,
    reranker: Any | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve and optionally rerank evidence for a question."""
    retrieved = retriever.retrieve(query, top_k=top_k)
    if reranker is None:
        return [
            {
                **result.to_dict(),
                "rerank_score": result.retrieval_score,
            }
            for result in retrieved
        ]
    reranked = reranker.rerank(query, retrieved, top_k=top_k)
    return [result.to_dict() for result in reranked]
