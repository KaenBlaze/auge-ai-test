"""Local vector store backed by FAISS with JSON metadata sidecar."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.chunking import Chunk

INDEX_FILENAME = "index.faiss"
METADATA_FILENAME = "metadata.json"


class VectorStore:
    """Persistent FAISS index with chunk metadata stored separately as JSON."""

    def __init__(self, persist_dir: Path) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / INDEX_FILENAME
        self.metadata_path = self.persist_dir / METADATA_FILENAME
        self._index: faiss.Index | None = None
        self._entries: list[dict[str, Any]] = []
        self._dimension: int | None = None
        self._load_if_exists()

    def build_index(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        """Build a fresh FAISS index from chunks and embeddings, then persist."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        if not chunks:
            self.reset()
            return 0

        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError("embeddings must be a 2-D array")

        self._dimension = vectors.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(vectors)

        self._entries = [self._chunk_to_entry(chunk) for chunk in chunks]
        self.save()
        return len(chunks)

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray | list[list[float]]) -> int:
        """Compatibility alias that rebuilds the full index."""
        vectors = np.asarray(embeddings, dtype=np.float32)
        return self.build_index(chunks, vectors)

    def save(self) -> None:
        """Persist the FAISS index and metadata JSON to disk."""
        if self._index is None:
            return

        faiss.write_index(self._index, str(self.index_path))
        payload = {
            "dimension": self._dimension,
            "count": len(self._entries),
            "index_type": "IndexFlatIP",
            "similarity": "cosine",
            "chunks": self._entries,
        }
        self.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load(self) -> bool:
        """Load index and metadata from disk. Returns True if loaded."""
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False

        self._index = faiss.read_index(str(self.index_path))
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._entries = payload.get("chunks", [])
        self._dimension = payload.get("dimension", self._index.d)
        return True

    def query(
        self,
        query_embedding: np.ndarray | list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Return top-k nearest chunks with cosine similarity scores."""
        if self._index is None or self.count() == 0:
            return []

        query_vector = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        scores, indices = self._index.search(query_vector, min(top_k, self.count()))

        hits: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self._entries[idx]
            hits.append(
                {
                    "id": entry["chunk_id"],
                    "text": entry["text"],
                    "metadata": {
                        "document": entry["document"],
                        "source_id": entry["source_id"],
                        "source": entry["document"],
                        **entry.get("metadata", {}),
                    },
                    "score": float(score),
                }
            )
        return hits

    def count(self) -> int:
        """Return number of indexed chunks."""
        return len(self._entries)

    def reset(self) -> None:
        """Delete persisted artifacts and clear in-memory state."""
        self._index = None
        self._entries = []
        self._dimension = None
        if self.index_path.exists():
            self.index_path.unlink()
        if self.metadata_path.exists():
            self.metadata_path.unlink()

    def _load_if_exists(self) -> None:
        try:
            self.load()
        except Exception:
            self._index = None
            self._entries = []
            self._dimension = None

    @staticmethod
    def _chunk_to_entry(chunk: Chunk) -> dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "document": chunk.document,
            "source_id": chunk.source_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
