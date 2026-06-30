"""Basic tests for the RAG project scaffolding."""

import sys
from pathlib import Path

import pytest

# Project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_documents
from src.config import Settings
from src.data_loader import Document, load_documents
from eval.metrics import exact_match, citation_precision, abstention_accuracy


@pytest.fixture
def sample_document() -> Document:
    return Document(
        id="test_doc",
        text="A" * 600,
        source="test.txt",
    )


def test_settings_defaults():
    settings = Settings()
    assert settings.chunk_size == 512
    assert settings.use_reranker is True


def test_chunk_documents_overlap(sample_document):
    chunks = chunk_documents([sample_document], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert chunks[0].document_id == "test_doc"


def test_load_documents(tmp_path):
    doc_file = tmp_path / "sample.md"
    doc_file.write_text("# Hello\n\nWorld.", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert "Hello" in docs[0].text


def test_exact_match():
    assert exact_match("Hello World", "hello world") == 1.0
    assert exact_match("Hello", "World") == 0.0


def test_citation_precision():
    assert citation_precision(["a.md", "b.md"], ["a.md"]) == 0.5


def test_abstention_accuracy():
    assert abstention_accuracy(True, True) == 1.0
    assert abstention_accuracy(False, True) == 0.0
