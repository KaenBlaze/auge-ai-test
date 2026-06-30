"""Tests for evaluation harness."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluate import run_evaluation
from src.confidence import ConfidenceResult
from src.rag_pipeline import Citation, RAGResponse


def test_run_evaluation_writes_results_json(tmp_path, monkeypatch):
    golden_path = tmp_path / "golden_seed.jsonl"
    golden_path.write_text(
        '{"id": "1", "question": "Q1?", "expected_answer": "temperature", '
        '"expected_sources": ["sensor_catalog.md"], "should_abstain": false}\n'
        '{"id": "2", "question": "Q2?", "expected_answer": "", '
        '"expected_sources": [], "should_abstain": true}\n',
        encoding="utf-8",
    )
    results_path = tmp_path / "results.json"

    pipeline = MagicMock()
    pipeline.vector_store.count.return_value = 1

    answerable_response = RAGResponse(
        question="Q1?",
        answer="Supports temperature sensors [source: sensor_catalog.md]",
        citations=[
            Citation(
                document="sensor_catalog.md",
                source_id="sensor_catalog",
                fragment="temperature sensors",
            )
        ],
        confidence=ConfidenceResult(confidence=0.8, abstained=False, reason="sufficient_evidence"),
        abstained=False,
        retrieved_chunks=[],
    )
    unanswerable_response = RAGResponse(
        question="Q2?",
        answer="I don't know based on the provided corpus.",
        citations=[],
        confidence=ConfidenceResult(confidence=0.1, abstained=True, reason="weak_top_retrieval_score_0.10"),
        abstained=True,
        retrieved_chunks=[],
    )
    pipeline.ask.side_effect = [answerable_response, unanswerable_response]

    payload = run_evaluation(
        golden_path=golden_path,
        results_path=results_path,
        pipeline=pipeline,
    )

    assert results_path.exists()
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert saved["n_examples"] == 2
    assert "retrieval_accuracy" in saved["summary"]
    assert "confidence_calibration" in saved["summary"]
    assert len(saved["per_example"]) == 2
    assert payload["summary"]["overall_accuracy"] >= 0.0
