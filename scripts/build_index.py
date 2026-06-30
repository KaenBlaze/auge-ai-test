#!/usr/bin/env python3
"""Build the vector index from corpus documents."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings
from src.rag_pipeline import RAGPipeline


def main() -> int:
    settings = get_settings()
    pipeline = RAGPipeline()

    print(f"Loading documents from: {settings.documents_dir}")
    count = pipeline.build_index()
    print(f"Successfully indexed {count} chunks into {settings.vector_store_dir}")
    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
