"""Tests for the data loading layer."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunking import chunk_documents
from src.config import Settings
from src.data_loader import (
    Document,
    load_document_records,
    load_documents,
    load_golden_seed,
    load_historical_results,
    record_to_document,
    summarize_loaded_data,
)
from eval.metrics import abstention_accuracy, citation_precision, exact_match


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
    assert settings.chunk_size == 640
    assert settings.use_reranker is True


def test_chunk_documents_overlap(sample_document):
    chunks = chunk_documents([sample_document], chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    assert chunks[0].source_id == "test"


def test_load_markdown_records_with_metadata(tmp_path):
    doc_file = tmp_path / "sample.md"
    doc_file.write_text(
        "---\nsource_id: sample-doc\n---\n"
        "# Hello\n\n"
        "First paragraph.\n\n"
        "## Details\n\n"
        "Second paragraph.",
        encoding="utf-8",
    )

    records = load_document_records(tmp_path)
    assert len(records) == 2
    assert records[0].document == "sample.md"
    assert records[0].source_id == "sample-doc"
    assert records[0].metadata["section_heading"] == "Hello"
    assert records[0].metadata["paragraph_index"] == 0
    assert records[0].metadata["line_start"] == 6
    assert "First paragraph." in records[0].text
    assert records[1].metadata["section_heading"] == "Details"
    assert "Second paragraph." in records[1].text


def test_load_documents_from_records(tmp_path):
    doc_file = tmp_path / "sample.md"
    doc_file.write_text("# Hello\n\nWorld.", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert len(docs) == 1
    assert "World" in docs[0].text
    assert docs[0].source == "sample.md"


def test_record_to_document_builds_stable_id():
    from src.data_loader import DocumentRecord

    doc = record_to_document(
        DocumentRecord(
            document="auge_overview.md",
            source_id="auge_overview",
            text="Example text.",
            metadata={"section_heading": "Core Capabilities", "paragraph_index": 2},
        )
    )
    assert doc.id == "auge_overview__core_capabilities__p2"
    assert doc.metadata["source_id"] == "auge_overview"


def test_load_golden_seed(tmp_path):
    seed_path = tmp_path / "golden_seed.jsonl"
    seed_path.write_text(
        '{"id": "a", "question": "Q1?"}\n{"id": "b", "question": "Q2?"}\n',
        encoding="utf-8",
    )
    examples = load_golden_seed(seed_path)
    assert len(examples) == 2
    assert examples[0]["id"] == "a"


def test_load_historical_results_empty(tmp_path):
    csv_path = tmp_path / "historical_results.csv"
    csv_path.write_text("timestamp,n_examples,exact_match\n", encoding="utf-8")
    df = load_historical_results(csv_path)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 0
    assert "timestamp" in df.columns


def test_load_historical_results_with_rows(tmp_path):
    csv_path = tmp_path / "historical_results.csv"
    csv_path.write_text(
        "timestamp,n_examples,exact_match\n"
        "2026-01-01T00:00:00+00:00,3,0.5000\n",
        encoding="utf-8",
    )
    df = load_historical_results(csv_path)
    assert len(df) == 1
    assert df.loc[0, "n_examples"] == 3


def test_summarize_loaded_data_project_corpus():
    root = Path(__file__).resolve().parent.parent
    summary = summarize_loaded_data(
        documents_dir=root / "data" / "documents",
        golden_seed_path=root / "data" / "golden_seed.jsonl",
        historical_results_path=root / "data" / "historical_results.csv",
    )
    assert summary["documents"] >= 2
    assert summary["records"] >= 2
    assert summary["golden_seed_examples"] == 3


def test_exact_match():
    assert exact_match("Hello World", "hello world") == 1.0
    assert exact_match("Hello", "World") == 0.0


def test_citation_precision():
    assert citation_precision(["a.md", "b.md"], ["a.md"]) == 0.5


def test_abstention_accuracy():
    assert abstention_accuracy(True, True) == 1.0
    assert abstention_accuracy(False, True) == 0.0
