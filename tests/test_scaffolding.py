"""Basic tests for the RAG project scaffolding."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_documents, DEFAULT_CHUNK_SIZE
from src.config import Settings
from src.data_loader import Document


@pytest.fixture
def sample_document() -> Document:
    return Document(
        id="test_doc",
        text="A" * 2500,
        source="test.txt",
        metadata={"source_id": "test", "section_heading": None, "paragraph_index": 0},
    )


def test_settings_defaults():
    settings = Settings()
    assert settings.chunk_size == DEFAULT_CHUNK_SIZE
    assert settings.use_reranker is True


def test_chunk_documents_overlap(sample_document):
    chunks = chunk_documents([sample_document], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert chunks[0].chunk_id.startswith("test__")
