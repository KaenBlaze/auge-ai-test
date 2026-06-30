"""Local vector store backed by ChromaDB."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.chunking import Chunk


class VectorStore:
    """Persistent local vector store using ChromaDB."""

    COLLECTION_NAME = "auge_rag_chunks"

    def __init__(self, persist_dir: Path) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Index chunks with precomputed embeddings. Returns count added."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            return 0

        collection = self._get_collection()
        collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": c.document_id,
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in chunks
            ],
        )
        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return top-k nearest chunks with scores and metadata."""
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # Chroma cosine distance: 0 = identical; convert to similarity
            similarity = 1.0 - distance
            hits.append(
                {
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": similarity,
                }
            )
        return hits

    def count(self) -> int:
        """Return number of indexed chunks."""
        return self._get_collection().count()

    def reset(self) -> None:
        """Delete and recreate the collection.

        TODO: Implement safer incremental updates instead of full reset.
        """
        if self._client is not None:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = None
