"""Load and normalize documents and auxiliary datasets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, TypedDict

import pandas as pd

HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$")
SUPPORTED_MARKDOWN_EXTENSIONS = {".md", ".markdown"}


class DocumentRecordDict(TypedDict):
    """Normalized document record shape."""

    document: str
    source_id: str
    text: str
    metadata: dict[str, Any]


@dataclass
class DocumentRecord:
    """A normalized paragraph-level record from the corpus."""

    document: str
    source_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> DocumentRecordDict:
        return {
            "document": self.document,
            "source_id": self.source_id,
            "text": self.text,
            "metadata": self.metadata,
        }


@dataclass
class Document:
    """Legacy document object used by the chunking pipeline."""

    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def load_document_records(documents_dir: Path) -> list[DocumentRecord]:
    """Load Markdown files and return normalized paragraph-level records."""
    documents_dir = Path(documents_dir)
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    records: list[DocumentRecord] = []
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in SUPPORTED_MARKDOWN_EXTENSIONS:
            records.extend(_load_markdown_file(path, documents_dir))
        elif suffix == ".txt":
            records.extend(_load_plaintext_file(path, documents_dir))
    return records


def load_documents(documents_dir: Path) -> list[Document]:
    """Load documents as legacy Document objects for the indexing pipeline."""
    return [record_to_document(record) for record in load_document_records(documents_dir)]


def load_historical_results(path: Path) -> pd.DataFrame:
    """Load evaluation history from CSV using pandas."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return df


def load_golden_seed(path: Path) -> list[dict[str, Any]]:
    """Load golden evaluation examples from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden seed file not found: {path}")

    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                examples.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}") from exc
    return examples


def summarize_loaded_data(
    documents_dir: Path | None = None,
    golden_seed_path: Path | None = None,
    historical_results_path: Path | None = None,
) -> dict[str, int]:
    """Return counts of loaded documents, records, and auxiliary datasets."""
    from src.config import get_settings

    settings = get_settings()
    documents_dir = Path(documents_dir or settings.documents_dir)
    golden_seed_path = Path(golden_seed_path or settings.golden_seed_path)
    historical_results_path = Path(historical_results_path or settings.historical_results_path)

    records = load_document_records(documents_dir)
    unique_documents = {record.document for record in records}
    golden_seed = load_golden_seed(golden_seed_path)
    historical_results = load_historical_results(historical_results_path)

    return {
        "documents": len(unique_documents),
        "records": len(records),
        "golden_seed_examples": len(golden_seed),
        "historical_result_runs": len(historical_results),
    }


def iter_document_records(documents_dir: Path) -> Iterator[DocumentRecord]:
    """Lazy iterator over normalized document records."""
    for record in load_document_records(documents_dir):
        yield record


def record_to_document(record: DocumentRecord) -> Document:
    """Convert a normalized record into a legacy Document for chunking."""
    section = record.metadata.get("section_heading") or "root"
    section_slug = re.sub(r"[^a-z0-9]+", "_", section.lower()).strip("_")[:40] or "root"
    paragraph_index = record.metadata.get("paragraph_index", 0)
    record_id = f"{record.source_id}__{section_slug}__p{paragraph_index}"

    return Document(
        id=record_id,
        text=record.text,
        source=record.document,
        metadata={**record.metadata, "source_id": record.source_id},
    )


def _load_markdown_file(path: Path, root: Path) -> list[DocumentRecord]:
    lines = path.read_text(encoding="utf-8").splitlines()
    frontmatter, body_start = _parse_frontmatter(lines)
    filename = str(path.relative_to(root))
    source_id = str(frontmatter.get("source_id") or path.stem)

    records: list[DocumentRecord] = []
    current_heading: str | None = None
    buffered_lines: list[tuple[int, str]] = []

    def flush_section() -> None:
        nonlocal buffered_lines, current_heading
        for paragraph_index, (line_start, line_end, text) in enumerate(
            _lines_to_paragraphs(buffered_lines)
        ):
            records.append(
                DocumentRecord(
                    document=filename,
                    source_id=source_id,
                    text=text,
                    metadata={
                        "filename": filename,
                        "section_heading": current_heading,
                        "paragraph_index": paragraph_index,
                        "line_start": line_start,
                        "line_end": line_end,
                        **frontmatter,
                    },
                )
            )
        buffered_lines = []

    for index in range(body_start, len(lines)):
        line_number = index + 1
        line = lines[index]
        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_section()
            current_heading = heading_match.group(1).strip()
            continue
        buffered_lines.append((line_number, line))

    flush_section()
    return records


def _load_plaintext_file(path: Path, root: Path) -> list[DocumentRecord]:
    """Load a plain-text file as paragraph-level records."""
    lines = path.read_text(encoding="utf-8").splitlines()
    filename = str(path.relative_to(root))
    source_id = path.stem
    records: list[DocumentRecord] = []

    numbered_lines = [(index + 1, line) for index, line in enumerate(lines)]
    for paragraph_index, (line_start, line_end, text) in enumerate(_lines_to_paragraphs(numbered_lines)):
        records.append(
            DocumentRecord(
                document=filename,
                source_id=source_id,
                text=text,
                metadata={
                    "filename": filename,
                    "section_heading": None,
                    "paragraph_index": paragraph_index,
                    "line_start": line_start,
                    "line_end": line_end,
                },
            )
        )
    return records


def _parse_frontmatter(lines: list[str]) -> tuple[dict[str, Any], int]:
    """Parse optional YAML-style frontmatter and return metadata plus body start index."""
    if not lines or lines[0].strip() != "---":
        return {}, 0

    frontmatter: dict[str, Any] = {}
    index = 1
    while index < len(lines):
        line = lines[index].strip()
        if line == "---":
            return frontmatter, index + 1
        if ":" in line:
            key, value = line.split(":", 1)
            frontmatter[key.strip()] = value.strip().strip("\"'")
        index += 1
    return frontmatter, 0


def _lines_to_paragraphs(lines: list[tuple[int, str]]) -> list[tuple[int, int, str]]:
    """Group consecutive non-empty lines into paragraphs with line ranges."""
    paragraphs: list[tuple[int, int, str]] = []
    current: list[tuple[int, str]] = []

    def flush() -> None:
        if not current:
            return
        text = "\n".join(content for _, content in current).strip()
        if text:
            paragraphs.append((current[0][0], current[-1][0], text))
        current.clear()

    for line_number, content in lines:
        if not content.strip():
            flush()
            continue
        current.append((line_number, content))

    flush()
    return paragraphs
