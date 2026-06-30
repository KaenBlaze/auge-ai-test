"""Before/after experiment: retrieval-only baseline vs retrieval + reranking."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.evaluate import collect_evaluations, evaluate_example, summarize_run
from src.config import Settings, get_settings
from src.data_loader import load_golden_seed
from src.rag_pipeline import RAGPipeline

EXPERIMENT_METRICS = (
    "retrieval_accuracy",
    "citation_correctness",
    "hallucination",
    "correct_abstention_rate",
)

DEFAULT_RESULTS_PATH = Path("eval/experiment_results.json")
REPORT_PATH = Path("REPORT.md")
EXPERIMENT_START = "<!-- EXPERIMENT_START -->"
EXPERIMENT_END = "<!-- EXPERIMENT_END -->"


def run_experiment(
    golden_path: Path | None = None,
    results_path: Path | None = None,
    report_path: Path | None = None,
) -> dict:
    """Compare baseline retrieval against retrieval + reranking."""
    settings = get_settings()
    golden_path = Path(golden_path or settings.golden_seed_path)
    results_path = Path(results_path or DEFAULT_RESULTS_PATH)
    report_path = Path(report_path or REPORT_PATH)

    examples = load_golden_seed(golden_path)
    if not examples:
        print(f"No examples found in {golden_path}")
        return {}

    baseline_pipeline = _build_pipeline(settings, use_reranker=False)
    reranked_pipeline = _build_pipeline(settings, use_reranker=True)

    for pipeline in (baseline_pipeline, reranked_pipeline):
        if pipeline.vector_store.count() == 0:
            print("Warning: Index is empty. Building index before experiment...")
            pipeline.build_index()
            break

    print("\n=== Baseline: retrieval without reranking ===")
    baseline_evaluations = collect_evaluations(baseline_pipeline, examples)
    baseline_summary = summarize_run(baseline_evaluations)

    print("\n=== After: retrieval with reranking ===")
    reranked_evaluations = collect_evaluations(reranked_pipeline, examples)
    reranked_summary = summarize_run(reranked_evaluations)

    comparison = _compare_summaries(baseline_summary, reranked_summary)
    explanation = _build_explanation(comparison)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "golden_set": str(golden_path),
        "n_examples": len(examples),
        "baseline": {
            "label": "retrieval_without_reranking",
            "metrics": _select_metrics(baseline_summary),
            "summary": baseline_summary,
        },
        "reranked": {
            "label": "retrieval_with_reranking",
            "metrics": _select_metrics(reranked_summary),
            "summary": reranked_summary,
        },
        "comparison": comparison,
        "explanation": explanation,
    }

    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    _print_comparison_table(baseline_summary, reranked_summary, comparison)
    print(explanation)
    print(f"\nWrote experiment results to {results_path}")

    _update_report(report_path, payload)
    print(f"Updated {report_path}")
    return payload


def _build_pipeline(settings: Settings, use_reranker: bool) -> RAGPipeline:
    variant = settings.model_copy(update={"use_reranker": use_reranker})
    return RAGPipeline(variant)


def _select_metrics(summary: dict[str, float]) -> dict[str, float]:
    return {name: summary.get(name, 0.0) for name in EXPERIMENT_METRICS}


def _compare_summaries(
    baseline: dict[str, float],
    reranked: dict[str, float],
) -> dict[str, dict[str, float]]:
    comparison: dict[str, dict[str, float]] = {}
    higher_is_better = {
        "retrieval_accuracy": True,
        "citation_correctness": True,
        "hallucination": False,
        "correct_abstention_rate": True,
    }
    for metric in EXPERIMENT_METRICS:
        base_value = baseline.get(metric, 0.0)
        after_value = reranked.get(metric, 0.0)
        comparison[metric] = {
            "baseline": round(base_value, 4),
            "reranked": round(after_value, 4),
            "delta": round(after_value - base_value, 4),
            "percent_change": round(
                _percent_change(base_value, after_value, higher_is_better[metric]),
                2,
            ),
            "improved": _improved(base_value, after_value, higher_is_better[metric]),
        }
    return comparison


def _percent_change(baseline: float, after: float, higher_is_better: bool) -> float:
    if baseline == 0:
        if higher_is_better:
            return 100.0 if after > 0 else 0.0
        return 100.0 if after < baseline else 0.0
    if higher_is_better:
        return ((after - baseline) / baseline) * 100
    return ((baseline - after) / baseline) * 100


def _improved(baseline: float, after: float, higher_is_better: bool) -> bool:
    if higher_is_better:
        return after > baseline
    return after < baseline


def _build_explanation(comparison: dict[str, dict[str, float]]) -> str:
    improved = [metric for metric, values in comparison.items() if values["improved"]]
    declined = [metric for metric, values in comparison.items() if not values["improved"] and values["delta"] != 0]
    unchanged = [metric for metric, values in comparison.items() if values["delta"] == 0]

    lines = ["\n=== Interpretation ==="]
    if improved:
        lines.append(
            "Reranking improved: "
            + ", ".join(f"{metric} ({comparison[metric]['percent_change']:+.1f}%)" for metric in improved)
            + "."
        )
    if declined:
        lines.append(
            "Reranking did not improve: "
            + ", ".join(f"{metric} ({comparison[metric]['percent_change']:+.1f}%)" for metric in declined)
            + "."
        )
    if unchanged:
        lines.append("Unchanged metrics: " + ", ".join(unchanged) + ".")

    lines.append(
        "Baseline uses dense FAISS retrieval only (top-k by embedding similarity). "
        "The reranked condition adds a local cross-encoder that re-orders retrieved "
        "chunks before generation and confidence scoring."
    )

    if not improved or declined:
        lines.append(
            "On this small golden set, reranking may show little gain when dense retrieval "
            "already places the correct document near the top, when the corpus is tiny, or when "
            "confidence thresholds and citation requirements dominate downstream abstention. "
            "Reranking helps most on ambiguous queries where lexical overlap misleads bi-encoder retrieval."
        )
    elif declined and not improved:
        lines.append(
            "Possible reasons for decline: cross-encoder scores can disagree with citation-focused "
            "labels on a tiny evaluation set, rerank latency is traded for marginal reordering gains, "
            "or min_rerank_score abstention filters answers that would have passed retrieval-only ordering."
        )

    return "\n".join(lines)


def _print_comparison_table(
    baseline: dict[str, float],
    reranked: dict[str, float],
    comparison: dict[str, dict[str, float]],
) -> None:
    print("\n=== Before / After Comparison ===")
    print(f"{'Metric':<28} {'Baseline':>10} {'Reranked':>10} {'Delta':>10} {'Change':>10}")
    print("-" * 70)
    for metric in EXPERIMENT_METRICS:
        row = comparison[metric]
        print(
            f"{metric:<28} "
            f"{row['baseline']:10.4f} "
            f"{row['reranked']:10.4f} "
            f"{row['delta']:+10.4f} "
            f"{row['percent_change']:+9.1f}%"
        )


def _update_report(report_path: Path, payload: dict) -> None:
    section = _render_report_section(payload)
    if report_path.exists():
        content = report_path.read_text(encoding="utf-8")
    else:
        content = "# Evaluation Report\n"

    if EXPERIMENT_START in content and EXPERIMENT_END in content:
        start = content.index(EXPERIMENT_START)
        end = content.index(EXPERIMENT_END) + len(EXPERIMENT_END)
        updated = content[:start] + section + content[end:]
    elif "## Before / After Experiment" in content:
        start = content.index("## Before / After Experiment")
        next_heading = content.find("\n## ", start + 1)
        updated = content[:start] + section.strip() + "\n"
        if next_heading != -1:
            updated += content[next_heading:]
    else:
        updated = content.rstrip() + "\n\n" + section

    report_path.write_text(updated, encoding="utf-8")


def _render_report_section(payload: dict) -> str:
    comparison = payload["comparison"]
    lines = [
        "## Before / After Experiment",
        "",
        EXPERIMENT_START,
        "",
        f"**Run date:** {payload['timestamp']}",
        f"**Golden set:** `{payload['golden_set']}` ({payload['n_examples']} examples)",
        "",
        "| Metric | Baseline | Reranked | Delta | % Change |",
        "|--------|----------|----------|-------|----------|",
    ]
    for metric in EXPERIMENT_METRICS:
        row = comparison[metric]
        lines.append(
            f"| {metric} | {row['baseline']:.4f} | {row['reranked']:.4f} | "
            f"{row['delta']:+.4f} | {row['percent_change']:+.1f}% |"
        )

    lines.extend(
        [
            "",
            "### What changed",
            "",
            "- **Baseline:** dense FAISS retrieval only (`use_reranker=false`).",
            "- **After:** dense retrieval followed by local cross-encoder reranking (`use_reranker=true`).",
            "",
            "### Interpretation",
            "",
            payload["explanation"].replace("=== Interpretation ===", "").strip(),
            "",
            "Reproduce with:",
            "",
            "```bash",
            "python eval/experiment.py",
            "```",
            "",
            EXPERIMENT_END,
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run before/after reranking experiment")
    parser.add_argument("--golden-set", type=Path, default=None, help="Path to golden JSONL")
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to write experiment_results.json",
    )
    parser.add_argument("--report", type=Path, default=REPORT_PATH, help="Path to REPORT.md")
    args = parser.parse_args()

    run_experiment(golden_path=args.golden_set, results_path=args.results, report_path=args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
