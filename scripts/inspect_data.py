#!/usr/bin/env python3
"""Print counts for loaded corpus and auxiliary datasets."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.data_loader import summarize_loaded_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect loaded document and dataset counts")
    parser.add_argument("--documents-dir", type=Path, default=None)
    parser.add_argument("--golden-seed", type=Path, default=None)
    parser.add_argument("--historical-results", type=Path, default=None)
    args = parser.parse_args()

    settings = get_settings()
    summary = summarize_loaded_data(
        documents_dir=args.documents_dir or settings.documents_dir,
        golden_seed_path=args.golden_seed or settings.golden_seed_path,
        historical_results_path=args.historical_results or settings.historical_results_path,
    )

    print(f"Documents loaded: {summary['documents']}")
    print(f"Records loaded: {summary['records']}")
    print(f"Golden seed examples: {summary['golden_seed_examples']}")
    print(f"Historical result runs: {summary['historical_result_runs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
