#!/usr/bin/env python3
"""Build the FAISS vector index from corpus documents."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_records
from src.config import get_settings
from src.data_loader import load_document_records
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


def main() -> int:
    settings = get_settings()

    print(f"1. Loading documents from: {settings.documents_dir}")
    records = load_document_records(settings.documents_dir)
    if not records:
        print("No documents found.", file=sys.stderr)
        return 1
    print(f"   Loaded {len(records)} paragraph records")

    print("2. Chunking documents")
    chunks = chunk_records(
        records,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    if not chunks:
        print("No chunks produced.", file=sys.stderr)
        return 1
    print(f"   Produced {len(chunks)} chunks")

    print(f"3. Embedding chunks with: {settings.embedding_model}")
    embedding_model = EmbeddingModel(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    texts = [chunk.text for chunk in chunks]
    embeddings = embedding_model.embed_texts(texts)
    print(f"   Embedding dimension: {embeddings.shape[1]}")

    print("4. Building FAISS index")
    vector_store = VectorStore(settings.faiss_index_dir)
    count = vector_store.build_index(chunks, embeddings)
    print(f"   Indexed {count} vectors")

    print("5. Saving FAISS index and metadata")
    print(f"   Index:    {vector_store.index_path}")
    print(f"   Metadata: {vector_store.metadata_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
