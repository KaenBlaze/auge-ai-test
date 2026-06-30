"""Tests for evaluation metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (
    citation_correctness,
    citation_coverage,
    confidence_calibration,
    correct_abstention,
    example_is_correct,
    false_abstention,
    groundedness,
    hallucination,
    is_answerable,
    retrieval_accuracy,
)


def test_is_answerable_supports_respondible_field():
    assert is_answerable({"respondible": True}) is True
    assert is_answerable({"respondible": False}) is False
    assert is_answerable({"should_abstain": True}) is False


def test_retrieval_accuracy():
    assert retrieval_accuracy(["a.md"], ["a.md", "b.md"]) == 1.0
    assert retrieval_accuracy(["a.md"], ["b.md"]) == 0.0
    assert retrieval_accuracy([], ["a.md"]) == 1.0


def test_citation_coverage_requires_citation_when_answerable():
    assert citation_coverage(True, "Answer [source: a.md]", []) == 1.0
    assert citation_coverage(True, "Answer without citation", []) == 0.0
    assert citation_coverage(False, "Answer without citation", []) == 1.0


def test_citation_correctness_matches_source_id_or_document():
    citations = [{"document": "sensor_catalog.md", "source_id": "sensor_catalog", "fragment": "text"}]
    assert citation_correctness("See [source: sensor_catalog.md]", citations, ["sensor_catalog.md"]) == 1.0
    assert citation_correctness("See [source: sensor_catalog]", citations, ["sensor_catalog.md"]) == 1.0


def test_hallucination_on_unanswerable_answer():
    assert hallucination(False, False, "Made up answer", []) == 1.0
    assert hallucination(False, True, "I don't know based on the provided corpus.", []) == 0.0


def test_abstention_rates():
    assert correct_abstention(True, False) == 1.0
    assert false_abstention(True, True) == 1.0
    assert false_abstention(False, True) == 0.0


def test_example_is_correct_answerable_with_citation():
    assert example_is_correct(
        answerable=True,
        abstained=False,
        answer="Supports temperature sensors [source: sensor_catalog.md]",
        expected_answer="temperature",
        citations=[{"document": "sensor_catalog.md", "source_id": "sensor_catalog", "fragment": "temperature"}],
        expected_sources=["sensor_catalog.md"],
    )


def test_confidence_calibration_bins():
    calibration = confidence_calibration(
        [
            (0.9, 1.0),
            (0.85, 1.0),
            (0.2, 0.0),
            (0.15, 0.0),
        ],
        num_bins=2,
    )
    assert calibration["bins"]
    assert 0.0 <= calibration["ece"] <= 1.0
    assert 0.0 <= calibration["calibration_score"] <= 1.0


def test_groundedness_uses_citation_fragments():
    score = groundedness(
        "Temperature sensors are supported.",
        [{"document": "a.md", "source_id": "a", "fragment": "Temperature sensors are supported in the catalog."}],
        "temperature sensors",
    )
    assert score > 0.5
