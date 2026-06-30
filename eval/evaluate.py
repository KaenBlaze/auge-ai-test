"""Run evaluation harness against golden seed dataset."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (
    ExampleEvaluation,
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
from src.config import get_settings
from src.data_loader import load_golden_seed
from src.rag_pipeline import RAGPipeline

DEFAULT_RESULTS_PATH = Path("eval/results.json")


def evaluate_example(pipeline: RAGPipeline, example: dict) -> ExampleEvaluation:
    """Run the pipeline on one golden example and compute metrics."""
    question = example["question"]
    example_id = str(example.get("id", question))
    expected_answer = example.get("expected_answer", "")
    expected_sources = example.get("expected_sources", [])
    answerable = is_answerable(example)

    result = pipeline.ask(question)
    citations = [citation.to_dict() for citation in result.citations]
    retrieved_documents = [chunk.document for chunk in result.retrieved_chunks]
    confidence = result.confidence.confidence

    metrics = {
        "retrieval_accuracy": retrieval_accuracy(expected_sources, retrieved_documents),
        "citation_coverage": citation_coverage(answerable, result.answer, citations),
        "citation_correctness": citation_correctness(result.answer, citations, expected_sources),
        "groundedness": groundedness(result.answer, citations, expected_answer),
        "hallucination": hallucination(answerable, result.abstained, result.answer, citations),
        "correct_abstention": correct_abstention(result.abstained, answerable),
        "false_abstention": false_abstention(result.abstained, answerable),
    }
    is_correct = example_is_correct(
        answerable=answerable,
        abstained=result.abstained,
        answer=result.answer,
        expected_answer=expected_answer,
        citations=citations,
        expected_sources=expected_sources,
    )

    return ExampleEvaluation(
        example_id=example_id,
        question=question,
        answerable=answerable,
        abstained=result.abstained,
        confidence=confidence,
        metrics=metrics,
        is_correct=is_correct,
        details={
            "answer": result.answer,
            "reason": result.reason,
            "expected_answer": expected_answer,
            "expected_sources": expected_sources,
            "predicted_citations": citations,
            "retrieved_documents": retrieved_documents,
        },
    )


def run_evaluation(
    golden_path: Path | None = None,
    results_path: Path | None = None,
    pipeline: RAGPipeline | None = None,
) -> dict:
    """Evaluate all golden examples and write summary results."""
    settings = get_settings()
    golden_path = Path(golden_path or settings.golden_seed_path)
    results_path = Path(results_path or DEFAULT_RESULTS_PATH)

    examples = load_golden_seed(golden_path)
    if not examples:
        print(f"No examples found in {golden_path}")
        return {}

    pipeline = pipeline or RAGPipeline()
    if pipeline.vector_store.count() == 0:
        print("Warning: Index is empty. Building index before evaluation...")
        pipeline.build_index()

    example_evaluations: list[ExampleEvaluation] = []
    for index, example in enumerate(examples):
        print(f"Evaluating [{index + 1}/{len(examples)}]: {example['question'][:70]}")
        example_evaluations.append(evaluate_example(pipeline, example))

    summary = _summarize_run(example_evaluations)
    calibration = confidence_calibration(
        [(evaluation.confidence, 1.0 if evaluation.is_correct else 0.0) for evaluation in example_evaluations]
    )
    summary["confidence_calibration"] = calibration["calibration_score"]

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "golden_set": str(golden_path),
        "n_examples": len(examples),
        "summary": summary,
        "calibration": calibration,
        "per_example": [
            {
                "id": evaluation.example_id,
                "question": evaluation.question,
                "answerable": evaluation.answerable,
                "abstained": evaluation.abstained,
                "confidence": evaluation.confidence,
                "is_correct": evaluation.is_correct,
                "metrics": evaluation.metrics,
                "details": evaluation.details,
            }
            for evaluation in example_evaluations
        ],
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_summary_table(summary, calibration)
    print(f"\nWrote results to {results_path}")
    return payload


def _summarize_run(example_evaluations: list[ExampleEvaluation]) -> dict[str, float]:
    metric_names = [
        "retrieval_accuracy",
        "citation_coverage",
        "citation_correctness",
        "groundedness",
        "hallucination",
        "correct_abstention",
        "false_abstention",
    ]
    summary: dict[str, float] = {}
    for name in metric_names:
        values = [evaluation.metrics[name] for evaluation in example_evaluations]
        summary[name] = sum(values) / len(values) if values else 0.0

    unanswerable = [evaluation for evaluation in example_evaluations if not evaluation.answerable]
    answerable = [evaluation for evaluation in example_evaluations if evaluation.answerable]

    if unanswerable:
        summary["correct_abstention_rate"] = sum(
            1 for evaluation in unanswerable if evaluation.abstained
        ) / len(unanswerable)
    else:
        summary["correct_abstention_rate"] = 0.0

    if answerable:
        summary["false_abstention_rate"] = sum(
            1 for evaluation in answerable if evaluation.abstained
        ) / len(answerable)
    else:
        summary["false_abstention_rate"] = 0.0

    summary["overall_accuracy"] = (
        sum(1 for evaluation in example_evaluations if evaluation.is_correct) / len(example_evaluations)
    )
    return {key: round(value, 4) for key, value in summary.items()}


def _print_summary_table(summary: dict[str, float], calibration: dict) -> None:
    print("\n=== Evaluation Summary ===")
    print(f"{'Metric':<32} {'Value':>10}")
    print("-" * 44)
    for name, value in summary.items():
        print(f"{name:<32} {value:10.4f}")

    print("\n=== Confidence Calibration ===")
    print(f"{'ECE':<32} {calibration['ece']:10.4f}")
    print(f"{'Calibration score':<32} {calibration['calibration_score']:10.4f}")
    if calibration["bins"]:
        print(f"\n{'Bin':<12} {'Count':>8} {'Confidence':>12} {'Accuracy':>12}")
        print("-" * 46)
        for bucket in calibration["bins"]:
            print(
                f"{bucket['bin']:<12} "
                f"{bucket['count']:8d} "
                f"{bucket['mean_confidence']:12.4f} "
                f"{bucket['accuracy']:12.4f}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline against golden set")
    parser.add_argument(
        "--golden-set",
        "--golden",
        dest="golden_set",
        type=Path,
        default=None,
        help="Path to golden_seed.jsonl or golden_set.jsonl",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to write eval/results.json",
    )
    args = parser.parse_args()

    run_evaluation(golden_path=args.golden_set, results_path=args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
