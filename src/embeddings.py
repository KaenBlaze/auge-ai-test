"""Local embedding model wrapper (sentence-transformers)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
FALLBACK_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_BATCH_SIZE = 32


class EmbeddingModel:
    """Thin wrapper around a local sentence-transformers model."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self._requested_model = model_name
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _load(self) -> "SentenceTransformer":
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        candidates = [self._requested_model]
        if self._requested_model != FALLBACK_EMBEDDING_MODEL:
            candidates.append(FALLBACK_EMBEDDING_MODEL)

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                self._model = SentenceTransformer(candidate, device=self.device)
                self.model_name = candidate
                if candidate != self._requested_model:
                    logger.warning(
                        "Embedding model %s unavailable; using fallback %s",
                        self._requested_model,
                        candidate,
                    )
                return self._model
            except Exception as exc:
                last_error = exc
                logger.warning("Failed to load embedding model %s: %s", candidate, exc)

        raise RuntimeError(
            f"Could not load embedding model {self._requested_model!r} "
            f"or fallback {FALLBACK_EMBEDDING_MODEL!r}"
        ) from last_error

    def embed_texts(self, texts: list[str], show_progress: bool | None = None) -> np.ndarray:
        """Embed document/passage texts and return a float32 numpy array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        model = self._load()
        if show_progress is None:
            show_progress = len(texts) > 100

        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        if self._is_bge_model():
            query = f"{BGE_QUERY_PREFIX}{query}"
        vector = self.embed_texts([query], show_progress=False)
        return vector[0]

    @property
    def dimension(self) -> int:
        """Return embedding vector dimensionality."""
        model = self._load()
        return model.get_sentence_embedding_dimension()

    def _is_bge_model(self) -> bool:
        return "bge" in self.model_name.lower()
