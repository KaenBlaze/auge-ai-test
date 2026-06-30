"""Run evaluation harness against golden seed dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import (
    MetricResult,
    abstention_accuracy,
    aggregate_metrics,
    citation_precision,
    citation_recall,
    contains_answer,
    exact_match,
    rouge_l,
)
from src.config import get_settings
from src.data_loader import load_golden_seed
from src.rag_pipeline import RAGPipeline


def evaluate_example(pipeline: RAGPipeline, example: dict) -> list[MetricResult]:
    """Run pipeline on one example and compute metrics."""
    question = example["question"]
    expected_answer = example.get("expected_answer", "")
    expected_sources = example.get("expected_sources", [])
    should_abstain = example.get("should_abstain", False)

    result = pipeline.ask(question)
    predicted_sources = [c.source for c in result.citations]

    metrics = [
        MetricResult("exact_match", exact_match(result.answer, expected_answer)),
        MetricResult("contains_answer", contains_answer(result.answer, expected_answer)),
        MetricResult("rouge_l", rouge_l(result.answer, expected_answer)),
        MetricResult("citation_precision", citation_precision(predicted_sources, expected_sources)),
        MetricResult("citation_recall", citation_recall(predicted_sources, expected_sources)),
        MetricResult("abstention_accuracy", abstention_accuracy(result.abstained, should_abstain)),
        MetricResult("confidence", result.confidence.score),
    ]
    return metrics


def run_evaluation(
    golden_path: Path | None = None,
    output_csv: Path | None = None,
) -> dict[str, float]:
    """Evaluate all golden examples and optionally append results to CSV.

    TODO: Add latency and token usage tracking per example.
    TODO: Generate per-example failure analysis report.
    TODO: Compare against historical_results.csv for regression detection.
    """
    settings = get_settings()
    golden_path = golden_path or settings.golden_seed_path
    output_csv = output_csv or settings.historical_results_path

    examples = load_golden_seed(golden_path)
    if not examples:
        print(f"No examples found in {golden_path}")
        return {}

    pipeline = RAGPipeline()
    if pipeline.vector_store.count() == 0:
        print("Warning: Index is empty. Building index before evaluation...")
        pipeline.build_index()

    all_metrics: list[MetricResult] = []
    for i, example in enumerate(examples):
        print(f"Evaluating [{i + 1}/{len(examples)}]: {example['question'][:60]}...")
        all_metrics.extend(evaluate_example(pipeline, example))

    summary = aggregate_metrics(all_metrics)
    _append_historical_results(output_csv, summary, len(examples))

    print("\n=== Evaluation Summary ===")
    for name, value in sorted(summary.items()):
        print(f"  {name}: {value:.4f}")

    return summary


def _append_historical_results(csv_path: Path, summary: dict[str, float], n_examples: int) -> None:
    """Append evaluation run to historical results CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["timestamp", "n_examples", *sorted(summary.keys())]
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_examples": n_examples,
        **{k: f"{v:.4f}" for k, v in summary.items()},
    }

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline against golden seed")
    parser.add_argument("--golden", type=Path, default=None, help="Path to golden_seed.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="Path to historical_results.csv")
    args = parser.parse_args()

    run_evaluation(args.golden, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
