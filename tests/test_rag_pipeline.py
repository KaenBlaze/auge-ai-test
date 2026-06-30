"""Tests for end-to-end RAG pipeline output."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence import ConfidenceResult
from src.config import Settings
from src.generator import ABSTENTION_PHRASE, GenerationResult
from src.rag_pipeline import RAGPipeline
from src.retriever import RetrievalResult


def _chunk(document: str = "sensor_catalog.md", fragment: str = "temperature pressure vibration") -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk_1",
        document=document,
        source_id="sensor_catalog",
        fragment=fragment,
        retrieval_score=0.9,
        rerank_score=0.9,
    )


def test_output_schema_shape(monkeypatch):
    pipeline = RAGPipeline(Settings(similarity_threshold=0.2, require_citations=False))

    monkeypatch.setattr(pipeline.retriever, "retrieve", lambda q: [_chunk()])
    monkeypatch.setattr(pipeline.reranker, "rerank", lambda q, chunks, top_k=None: chunks)
    monkeypatch.setattr(
        pipeline.generator,
        "generate",
        lambda q, chunks: GenerationResult(
            raw_output="Temperature sensors are listed.",
            answer="Temperature sensors are listed.",
            backend="ollama",
            model_name="test",
        ),
    )
    monkeypatch.setattr(
        pipeline.confidence_scorer,
        "score",
        lambda q, answer, chunks: ConfidenceResult(
            confidence=0.8,
            abstained=False,
            reason="sufficient_evidence",
        ),
    )

    result = pipeline.ask("Which sensors are supported?")
    payload = result.to_dict()

    assert set(payload) == {
        "question",
        "answer",
        "citations",
        "confidence",
        "abstained",
        "reason",
    }
    assert payload["question"] == "Which sensors are supported?"
    assert payload["abstained"] is False
    assert payload["citations"][0] == {
        "document": "sensor_catalog.md",
        "source_id": "sensor_catalog",
        "fragment": "temperature pressure vibration",
    }


def test_abstains_before_generation_when_retrieval_weak(monkeypatch):
    pipeline = RAGPipeline(Settings(similarity_threshold=0.2, min_retrieval_score=0.8))

    weak_chunk = RetrievalResult(
        chunk_id="chunk_1",
        document="sensor_catalog.md",
        source_id="sensor_catalog",
        fragment="weak",
        retrieval_score=0.3,
        rerank_score=0.3,
    )
    monkeypatch.setattr(pipeline.retriever, "retrieve", lambda q: [weak_chunk])
    monkeypatch.setattr(pipeline.reranker, "rerank", lambda q, chunks, top_k=None: chunks)

    generate = MagicMock()
    monkeypatch.setattr(pipeline.generator, "generate", generate)

    result = pipeline.ask("Which sensors?")
    generate.assert_not_called()
    assert result.abstained is True
    assert result.answer == ABSTENTION_PHRASE
    assert result.citations == []
