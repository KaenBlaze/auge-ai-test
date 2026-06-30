#!/usr/bin/env python3
"""Run a single question through the RAG pipeline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_pipeline import RAGPipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask a question via the RAG pipeline")
    parser.add_argument("question", nargs="?", default=None, help="Question to ask")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    if not args.question:
        args.question = input("Question: ").strip()
        if not args.question:
            print("No question provided.", file=sys.stderr)
            return 1

    pipeline = RAGPipeline()
    if pipeline.vector_store.count() == 0:
        print("Error: Index is empty. Run scripts/build_index.py first.", file=sys.stderr)
        return 1

    result = pipeline.ask(args.question)
    output = result.to_dict()

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"\nQ: {output['question']}\n")
        print(f"A: {output['answer']}\n")
        print(
            f"Confidence: {output['confidence']:.3f}  "
            f"Abstained: {output['abstained']}  "
            f"Reason: {output['reason']}"
        )
        if output["citations"]:
            print("\nCitations:")
            for index, citation in enumerate(output["citations"], 1):
                print(f"  [{index}] {citation['document']} ({citation['source_id']})")
                print(f"      {citation['fragment'][:200]}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
