"""Tests for FAISS vector store."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import Chunk
from src.vector_store import METADATA_FILENAME, INDEX_FILENAME, VectorStore


def _make_chunk(index: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc__section__chunk_{index}",
        document="sample.md",
        source_id="sample",
        text=f"Chunk text number {index}.",
        metadata={"chunk_index": index, "section_heading": "Section"},
    )


def test_build_index_persists_faiss_and_metadata(tmp_path):
    chunks = [_make_chunk(0), _make_chunk(1)]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    store = VectorStore(tmp_path)
    count = store.build_index(chunks, embeddings)

    assert count == 2
    assert (tmp_path / INDEX_FILENAME).exists()
    assert (tmp_path / METADATA_FILENAME).exists()
    assert store.count() == 2


def test_query_returns_nearest_chunk(tmp_path):
    chunks = [_make_chunk(0), _make_chunk(1)]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Normalize for cosine IP search
    faiss = pytest.importorskip("faiss")
    faiss.normalize_L2(embeddings)

    store = VectorStore(tmp_path)
    store.build_index(chunks, embeddings)

    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    faiss.normalize_L2(query.reshape(1, -1))

    hits = store.query(query, top_k=1)
    assert len(hits) == 1
    assert hits[0]["id"] == "doc__section__chunk_0"
    assert hits[0]["metadata"]["document"] == "sample.md"
    assert "Chunk text number 0" in hits[0]["text"]


def test_reload_from_disk(tmp_path):
    chunks = [_make_chunk(0)]
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    store = VectorStore(tmp_path)
    store.build_index(chunks, embeddings)

    reloaded = VectorStore(tmp_path)
    assert reloaded.count() == 1
    assert reloaded._entries[0]["chunk_id"] == "doc__section__chunk_0"


def test_reset_clears_artifacts(tmp_path):
    chunks = [_make_chunk(0)]
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

    store = VectorStore(tmp_path)
    store.build_index(chunks, embeddings)
    store.reset()

    assert store.count() == 0
    assert not (tmp_path / INDEX_FILENAME).exists()
    assert not (tmp_path / METADATA_FILENAME).exists()
