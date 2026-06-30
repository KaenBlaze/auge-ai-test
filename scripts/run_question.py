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
    output = {
        "question": result.question,
        "answer": result.answer,
        "abstained": result.abstained,
        "confidence": result.confidence.score,
        "citations": [
            {"source": c.source, "chunk_id": c.chunk_id, "score": c.score}
            for c in result.citations
        ],
    }

    if args.json:
        print(json.dumps(output, indent=2))
    else:
        print(f"\nQ: {result.question}\n")
        print(f"A: {result.answer}\n")
        print(f"Confidence: {result.confidence.score:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
