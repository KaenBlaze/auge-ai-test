"""Tests for confidence scoring and abstention."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.confidence import ConfidenceScorer, abstention_result
from src.config import Settings
from src.generator import ABSTENTION_PHRASE
from src.retriever import RetrievalResult


def _chunk(
    retrieval_score: float,
    rerank_score: float | None = None,
    document: str = "sensor_catalog.md",
    fragment: str = "temperature pressure vibration sensors",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id="chunk_1",
        document=document,
        source_id="sensor_catalog",
        fragment=fragment,
        retrieval_score=retrieval_score,
        rerank_score=rerank_score if rerank_score is not None else retrieval_score,
    )


def test_abstains_when_no_chunks():
    scorer = ConfidenceScorer()
    result = scorer.score("question", "answer", [])
    payload = result.to_dict()
    assert payload["abstained"] is True
    assert payload["confidence"] == 0.0
    assert payload["reason"] == "no_retrieved_chunks"


def test_abstains_when_no_chunk_passes_threshold():
    scorer = ConfidenceScorer(Settings(similarity_threshold=0.5))
    chunks = [_chunk(retrieval_score=0.2, rerank_score=0.2)]
    result = scorer.score("question", "answer", chunks)
    assert result.abstained
    assert result.reason == "no_chunk_passes_similarity_threshold"


def test_abstains_on_weak_retrieval_score():
    scorer = ConfidenceScorer(Settings(min_retrieval_score=0.5, similarity_threshold=0.2))
    chunks = [_chunk(retrieval_score=0.3, rerank_score=0.8)]
    result = scorer.score("question", "answer with enough words", chunks)
    assert result.abstained
    assert result.reason.startswith("weak_top_retrieval_score")


def test_abstains_on_weak_rerank_score():
    scorer = ConfidenceScorer(Settings(min_rerank_score=0.7, similarity_threshold=0.2))
    chunks = [_chunk(retrieval_score=0.8, rerank_score=0.3)]
    result = scorer.score("question", "answer with enough words", chunks)
    assert result.abstained
    assert result.reason.startswith("weak_top_rerank_score")


def test_abstains_when_answer_missing_citations():
    scorer = ConfidenceScorer(Settings(require_citations=True, similarity_threshold=0.2))
    chunks = [_chunk(retrieval_score=0.9, rerank_score=0.9)]
    result = scorer.score("Which sensors?", "Temperature sensors are supported.", chunks)
    assert result.abstained
    assert result.reason == "answer_missing_citations"


def test_abstains_when_model_expresses_uncertainty():
    scorer = ConfidenceScorer(Settings(similarity_threshold=0.2, require_citations=False))
    chunks = [_chunk(retrieval_score=0.9, rerank_score=0.9)]
    result = scorer.score("question", ABSTENTION_PHRASE, chunks)
    assert result.abstained
    assert result.reason == "model_expressed_uncertainty"


def test_accepts_answer_with_citation_support():
    scorer = ConfidenceScorer(
        Settings(
            similarity_threshold=0.2,
            min_retrieval_score=0.3,
            min_rerank_score=0.3,
            confidence_threshold=0.4,
            require_citations=True,
            min_citation_support=0.1,
        )
    )
    chunks = [
        _chunk(
            retrieval_score=0.9,
            rerank_score=0.9,
            fragment="The AUGE platform supports temperature, pressure, and vibration sensors.",
        )
    ]
    answer = (
        "The platform supports temperature and pressure sensors "
        "[source: sensor_catalog.md]."
    )
    result = scorer.score("Which sensors?", answer, chunks)
    assert not result.abstained
    assert result.reason == "sufficient_evidence"
    assert result.confidence >= 0.4
    assert "top_retrieval" in result.signals
    assert "citation_support" in result.signals


def test_abstains_when_confidence_below_threshold():
    scorer = ConfidenceScorer(
        Settings(
            similarity_threshold=0.2,
            min_retrieval_score=0.2,
            min_rerank_score=0.2,
            confidence_threshold=0.99,
            require_citations=False,
        )
    )
    chunks = [_chunk(retrieval_score=0.4, rerank_score=0.4)]
    result = scorer.score("question", "Some answer without citations.", chunks)
    assert result.abstained
    assert result.reason.startswith("confidence_below_threshold")


def test_output_dict_shape():
    result = abstention_result(0.2, "test_reason", signals={"top_retrieval": 0.2})
    payload = result.to_dict()
    assert set(payload) >= {"confidence", "abstained", "reason"}
    assert payload["abstained"] is True
    assert payload["reason"] == "test_reason"


def test_fragment_agreement_single_chunk_defaults_high():
    scorer = ConfidenceScorer(Settings(similarity_threshold=0.2, require_citations=False))
    chunks = [_chunk(retrieval_score=0.8, rerank_score=0.8)]
    result = scorer.score("q", "supported answer text", chunks)
    assert result.signals["fragment_agreement"] == 1.0
