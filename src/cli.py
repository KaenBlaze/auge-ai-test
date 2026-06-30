"""Command-line interface for the RAG system."""

from __future__ import annotations

import argparse
import json
import sys

from src.chunking import preview_chunks
from src.config import get_settings
from src.data_loader import load_document_records, summarize_loaded_data
from src.rag_pipeline import RAGPipeline


def _print_result(result, as_json: bool) -> int:
    output = result.to_dict()
    if as_json:
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


def build_index_cmd(args: argparse.Namespace) -> int:
    """Build the vector index from documents."""
    pipeline = RAGPipeline()
    count = pipeline.build_index(args.documents_dir)
    print(f"Indexed {count} chunks.")
    return 0 if count > 0 else 1


def ask_cmd(args: argparse.Namespace) -> int:
    """Ask a question and print the response."""
    pipeline = RAGPipeline()
    if pipeline.vector_store.count() == 0:
        print("Error: Index is empty. Run 'build-index' first.", file=sys.stderr)
        return 1

    result = pipeline.ask(args.question)
    return _print_result(result, args.json)


def inspect_data_cmd(args: argparse.Namespace) -> int:
    """Print how many documents and records were loaded."""
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


def preview_chunks_cmd(args: argparse.Namespace) -> int:
    """Preview the first chunks with metadata."""
    settings = get_settings()
    documents_dir = args.documents_dir or settings.documents_dir
    records = load_document_records(documents_dir)
    chunks = preview_chunks(
        records,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps(chunks, indent=2))
        return 0

    print(f"Previewing {len(chunks)} chunk(s)\n")
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        print(f"--- Chunk {index} ---")
        print(f"chunk_id:   {chunk['chunk_id']}")
        print(f"document:   {chunk['document']}")
        print(f"source_id:  {chunk['source_id']}")
        print(f"section:    {metadata.get('section_heading')}")
        print(f"lines:      {metadata.get('line_start')}–{metadata.get('line_end')}")
        print(f"tokens:     {metadata.get('token_count')}")
        print(f"text:\n{chunk['text'][:400]}{'...' if len(chunk['text']) > 400 else ''}\n")
    return 0


def serve_cmd(args: argparse.Namespace) -> int:
    """Start the API server."""
    from src.api import main as serve_main

    serve_main()
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="auge-rag",
        description="AUGE Intelligence local RAG system",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-index", help="Build vector index from documents")
    build_parser.add_argument(
        "--documents-dir",
        type=str,
        default=None,
        help="Override documents directory path",
    )
    build_parser.set_defaults(func=build_index_cmd)

    ask_parser = subparsers.add_parser("ask", help="Ask a question")
    ask_parser.add_argument("question", type=str, help="Question to answer")
    ask_parser.add_argument("--json", action="store_true", help="Output JSON")
    ask_parser.set_defaults(func=ask_cmd)

    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.set_defaults(func=serve_cmd)

    inspect_parser = subparsers.add_parser(
        "inspect-data",
        help="Print loaded document, record, and dataset counts",
    )
    inspect_parser.add_argument("--documents-dir", type=str, default=None)
    inspect_parser.add_argument("--golden-seed", type=str, default=None)
    inspect_parser.add_argument("--historical-results", type=str, default=None)
    inspect_parser.set_defaults(func=inspect_data_cmd)

    preview_parser = subparsers.add_parser("preview-chunks", help="Preview first chunks with metadata")
    preview_parser.add_argument("--documents-dir", type=str, default=None)
    preview_parser.add_argument("--limit", type=int, default=5)
    preview_parser.add_argument("--json", action="store_true")
    preview_parser.set_defaults(func=preview_chunks_cmd)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
