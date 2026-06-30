"""Tests for heading-aware chunking."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    chunk_documents,
    chunk_records,
    preview_chunks,
)
from src.config import Settings
from src.data_loader import Document, DocumentRecord, load_document_records


@pytest.fixture
def sample_document() -> Document:
    return Document(
        id="test_doc",
        text="A" * 2500,
        source="test.txt",
        metadata={"source_id": "test", "section_heading": None, "paragraph_index": 0},
    )


def test_settings_chunk_defaults():
    settings = Settings()
    assert settings.chunk_size == DEFAULT_CHUNK_SIZE
    assert settings.chunk_overlap == DEFAULT_CHUNK_OVERLAP


def test_chunk_documents_splits_long_text(sample_document):
    chunks = chunk_documents([sample_document], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert chunks[0].chunk_id.startswith("test__")
    assert chunks[0].document == "test.txt"
    assert chunks[0].source_id == "test"


def test_chunks_include_required_fields(tmp_path):
    doc_file = tmp_path / "sample.md"
    doc_file.write_text(
        "# Intro\n\nShort intro paragraph.\n\n"
        "# Details\n\nDetail paragraph one.\n\nDetail paragraph two.",
        encoding="utf-8",
    )
    records = load_document_records(tmp_path)
    chunks = chunk_records(records, chunk_size=640, chunk_overlap=100)

    assert chunks
    chunk = chunks[0]
    assert chunk.chunk_id
    assert chunk.document == "sample.md"
    assert chunk.source_id == "sample"
    assert chunk.text
    assert "section_heading" in chunk.metadata
    assert "paragraph_indices" in chunk.metadata


def test_sections_are_not_merged_across_headings(tmp_path):
    doc_file = tmp_path / "sample.md"
    doc_file.write_text(
        "# Section A\n\nAlpha content.\n\n# Section B\n\nBeta content.",
        encoding="utf-8",
    )
    records = load_document_records(tmp_path)
    chunks = chunk_records(records, chunk_size=640, chunk_overlap=0)

    headings = {chunk.metadata["section_heading"] for chunk in chunks}
    assert "Section A" in headings
    assert "Section B" in headings
    assert all("Alpha" not in chunk.text or "Beta" not in chunk.text for chunk in chunks)


def test_paragraphs_in_same_section_merge(tmp_path):
    doc_file = tmp_path / "sample.md"
    doc_file.write_text(
        "# Section\n\n"
        + "\n\n".join(f"Paragraph {index} with a little content." for index in range(4)),
        encoding="utf-8",
    )
    records = load_document_records(tmp_path)
    chunks = chunk_records(records, chunk_size=640, chunk_overlap=0)
    assert len(chunks) == 1
    assert len(chunks[0].metadata["paragraph_indices"]) == 4


def test_overlap_carries_context(tmp_path):
    long_paragraph = "Sentence one. " * 80
    doc_file = tmp_path / "sample.md"
    doc_file.write_text(f"# Section\n\n{long_paragraph}", encoding="utf-8")
    records = load_document_records(tmp_path)
    chunks = chunk_records(records, chunk_size=120, chunk_overlap=30)
    assert len(chunks) > 1
    assert chunks[1].text


def test_preview_chunks_limits_output(tmp_path):
    doc_file = tmp_path / "sample.md"
    doc_file.write_text(
        "# One\n\nFirst.\n\n# Two\n\nSecond.\n\n# Three\n\nThird.",
        encoding="utf-8",
    )
    records = load_document_records(tmp_path)
    preview = preview_chunks(records, limit=2)
    assert len(preview) == 2
    assert preview[0]["chunk_id"]
