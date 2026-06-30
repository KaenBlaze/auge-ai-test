#!/usr/bin/env python3
"""Preview the first chunks produced by the chunking strategy."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import preview_chunks
from src.config import get_settings
from src.data_loader import load_document_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview chunk output with metadata")
    parser.add_argument("--documents-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    settings = get_settings()
    documents_dir = args.documents_dir or settings.documents_dir
    records = load_document_records(documents_dir)
    chunks = preview_chunks(
        records,
        chunk_size=args.chunk_size or settings.chunk_size,
        chunk_overlap=args.chunk_overlap or settings.chunk_overlap,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps(chunks, indent=2))
        return 0

    print(f"Previewing {len(chunks)} of {len(records)} source records\n")
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


if __name__ == "__main__":
    sys.exit(main())
