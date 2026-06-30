"""Local embedding model wrapper (sentence-transformers)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Thin wrapper around a local sentence-transformers model."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None

    def _load(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        TODO: Add batching for large corpora to control memory usage.
        TODO: Cache embeddings on disk to avoid recomputation.
        """
        if not texts:
            return []
        model = self._load()
        embeddings = model.encode(texts, show_progress_bar=len(texts) > 100)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([query])[0]

    @property
    def dimension(self) -> int:
        """Return embedding vector dimensionality."""
        model = self._load()
        return model.get_sentence_embedding_dimension()
