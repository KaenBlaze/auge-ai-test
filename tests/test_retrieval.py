"""Tests for retrieval and reranking."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import Chunk
from src.config import Settings
from src.embeddings import EmbeddingModel
from src.reranker import CrossEncoderReranker, SimilarityFallbackReranker, create_reranker
from src.retriever import Retriever, retrieve_evidence
from src.vector_store import VectorStore


SAMPLE_QUESTION = "What is the electoral silence period before election day?"


def _build_test_index(tmp_path: Path) -> VectorStore:
    chunks = [
        Chunk(
            chunk_id="sensor_catalog__sensor_catalog__chunk_0",
            document="sensor_catalog.md",
            source_id="sensor_catalog",
            text="## Sensor Catalog\n\nThe AUGE platform supports temperature, pressure, and vibration sensors.",
            metadata={"section_heading": "Sensor Catalog"},
        ),
        Chunk(
            chunk_id="auge_overview__architecture__chunk_1",
            document="auge_overview.md",
            source_id="auge_overview",
            text="## Architecture\n\nData flows from edge gateways to the AUGE cloud.",
            metadata={"section_heading": "Architecture"},
        ),
    ]
    embeddings = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
        ],
        dtype=np.float32,
    )
    faiss = pytest.importorskip("faiss")
    faiss.normalize_L2(embeddings)

    store = VectorStore(tmp_path)
    store.build_index(chunks, embeddings)
    return store


def _mock_embedding_model(query_vector: np.ndarray) -> EmbeddingModel:
    model = MagicMock(spec=EmbeddingModel)
    model.embed_query.return_value = query_vector
    return model


def test_retriever_returns_required_fields(tmp_path):
    store = _build_test_index(tmp_path)
    query_vector = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    retriever = Retriever(store, _mock_embedding_model(query_vector), Settings(similarity_threshold=0.0))

    results = retriever.retrieve("sensor support", top_k=2)
    assert results
    payload = results[0].to_dict()
    assert set(payload) >= {
        "chunk_id",
        "document",
        "source_id",
        "fragment",
        "retrieval_score",
        "rerank_score",
    }
    assert payload["document"] == "sensor_catalog.md"
    assert "sensor" in payload["fragment"].lower()


def test_similarity_fallback_reranker_orders_by_retrieval_score():
    from src.retriever import RetrievalResult

    chunks = [
        RetrievalResult("a", "a.md", "a", "text a", retrieval_score=0.4),
        RetrievalResult("b", "b.md", "b", "text b", retrieval_score=0.9),
    ]
    reranker = SimilarityFallbackReranker()
    ranked = reranker.rerank("query", chunks, top_k=2)

    assert ranked[0].chunk_id == "b"
    assert ranked[0].rerank_score == 0.9
    assert ranked[1].chunk_id == "a"


def test_retrieve_evidence_with_fallback_reranker(tmp_path):
    store = _build_test_index(tmp_path)
    query_vector = np.array([0.95, 0.05, 0.0], dtype=np.float32)
    retriever = Retriever(store, _mock_embedding_model(query_vector), Settings(similarity_threshold=0.0))
    reranker = SimilarityFallbackReranker()

    evidence = retrieve_evidence(
        "Which sensors are supported?",
        retriever=retriever,
        reranker=reranker,
        top_k=2,
    )

    assert evidence
    assert evidence[0]["document"] == "sensor_catalog.md"
    assert evidence[0]["retrieval_score"] >= evidence[-1]["retrieval_score"]
    assert evidence[0]["rerank_score"] == evidence[0]["retrieval_score"]


def test_retrieve_sample_question():
    """Retrieve evidence for a sample question against the project index."""
    settings = Settings()
    store = VectorStore(settings.faiss_index_dir)
    if store.count() == 0:
        pytest.skip("FAISS index not built; run scripts/build_index.py first")

    embedding_model = EmbeddingModel(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    retriever = Retriever(store, embedding_model, settings)
    reranker = SimilarityFallbackReranker(settings)

    evidence = retrieve_evidence(
        SAMPLE_QUESTION,
        retriever=retriever,
        reranker=reranker,
        top_k=settings.top_k_rerank,
    )

    assert evidence, "Expected at least one retrieved chunk"
    top = evidence[0]
    assert top["chunk_id"]
    assert top["document"]
    assert top["source_id"]
    assert top["fragment"]
    assert top["retrieval_score"] > 0.0
    assert top["rerank_score"] >= 0.0
    assert (
        "silence" in top["fragment"].lower()
        or "electoral" in top["fragment"].lower()
        or "72" in top["fragment"]
    )


def test_create_reranker_disabled_uses_noop():
    reranker = create_reranker(Settings(use_reranker=False))
    from src.reranker import NoOpReranker

    assert isinstance(reranker, NoOpReranker)


def test_cross_encoder_reranker_falls_back_when_model_unavailable(monkeypatch):
    from src.retriever import RetrievalResult

    reranker = CrossEncoderReranker("fake-model", Settings())

    def _fail_load():
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(reranker, "_load", _fail_load)
    chunks = [
        RetrievalResult("a", "a.md", "a", "text a", retrieval_score=0.2),
        RetrievalResult("b", "b.md", "b", "text b", retrieval_score=0.8),
    ]
    ranked = reranker.rerank("query", chunks, top_k=2)
    assert ranked[0].chunk_id == "b"
    assert ranked[0].metadata.get("reranker") == "similarity_fallback"
