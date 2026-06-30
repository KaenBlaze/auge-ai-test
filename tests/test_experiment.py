"""Tests for before/after reranking experiment."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.experiment import (
    _build_explanation,
    _compare_summaries,
    _percent_change,
    run_experiment,
)
from eval.metrics import ExampleEvaluation


def test_percent_change_higher_is_better():
    assert _percent_change(0.5, 0.75, higher_is_better=True) == 50.0


def test_percent_change_lower_is_better_for_hallucination():
    assert _percent_change(0.4, 0.2, higher_is_better=False) == 50.0


def test_compare_summaries_marks_improvement():
    comparison = _compare_summaries(
        {
            "retrieval_accuracy": 0.5,
            "citation_correctness": 0.5,
            "hallucination": 0.2,
            "correct_abstention_rate": 1.0,
        },
        {
            "retrieval_accuracy": 1.0,
            "citation_correctness": 1.0,
            "hallucination": 0.0,
            "correct_abstention_rate": 1.0,
        },
    )
    assert comparison["retrieval_accuracy"]["improved"] is True
    assert comparison["hallucination"]["improved"] is True


def test_build_explanation_mentions_reranking():
    comparison = _compare_summaries(
        {"retrieval_accuracy": 0.5, "citation_correctness": 0.5, "hallucination": 0.0, "correct_abstention_rate": 1.0},
        {"retrieval_accuracy": 1.0, "citation_correctness": 1.0, "hallucination": 0.0, "correct_abstention_rate": 1.0},
    )
    text = _build_explanation(comparison)
    assert "cross-encoder" in text


def test_run_experiment_writes_results_and_report(tmp_path, monkeypatch):
    golden_path = tmp_path / "golden_seed.jsonl"
    golden_path.write_text(
        '{"id": "1", "question": "Q1?", "expected_answer": "temperature", '
        '"expected_sources": ["sensor_catalog.md"], "should_abstain": false}\n',
        encoding="utf-8",
    )
    results_path = tmp_path / "experiment_results.json"
    report_path = tmp_path / "REPORT.md"
    report_path.write_text("# Evaluation Report\n", encoding="utf-8")

    evaluation = ExampleEvaluation(
        example_id="1",
        question="Q1?",
        answerable=True,
        abstained=False,
        confidence=0.8,
        metrics={
            "retrieval_accuracy": 1.0,
            "citation_correctness": 1.0,
            "hallucination": 0.0,
            "correct_abstention": 0.0,
            "false_abstention": 0.0,
            "citation_coverage": 1.0,
            "groundedness": 1.0,
        },
        is_correct=True,
    )

    pipeline = MagicMock()
    pipeline.vector_store.count.return_value = 1
    monkeypatch.setattr("eval.experiment._build_pipeline", lambda settings, use_reranker: pipeline)
    monkeypatch.setattr("eval.experiment.collect_evaluations", lambda pipeline, examples: [evaluation])

    payload = run_experiment(
        golden_path=golden_path,
        results_path=results_path,
        report_path=report_path,
    )

    assert results_path.exists()
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    assert "baseline" in saved and "reranked" in saved
    assert "comparison" in saved
    report_text = report_path.read_text(encoding="utf-8")
    assert "## Before / After Experiment" in report_text
    assert payload["comparison"]["retrieval_accuracy"]["baseline"] == 1.0
