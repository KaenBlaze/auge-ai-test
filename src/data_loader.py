"""Load and normalize documents from the corpus directory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Document:
    """A single source document with metadata."""

    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".json"}


def load_documents(documents_dir: Path) -> list[Document]:
    """Load all supported documents from *documents_dir*.

    TODO: Add support for PDF, HTML, and structured JSON corpora.
    TODO: Deduplicate documents and validate required metadata fields.
    """
    documents_dir = Path(documents_dir)
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    docs: list[Document] = []
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        docs.extend(_load_file(path, documents_dir))
    return docs


def _load_file(path: Path, root: Path) -> list[Document]:
    """Parse a single file into one or more Document objects."""
    rel_source = str(path.relative_to(root))
    doc_id = rel_source.replace("/", "_").replace("\\", "_")

    if path.suffix.lower() == ".json":
        return _load_json(path, doc_id, rel_source)

    text = path.read_text(encoding="utf-8")
    return [Document(id=doc_id, text=text, source=rel_source)]


def _load_json(path: Path, doc_id: str, source: str) -> list[Document]:
    """Load a JSON file — single object or list of objects with 'text' field."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [
            Document(
                id=f"{doc_id}_{i}",
                text=item.get("text", ""),
                source=source,
                metadata={k: v for k, v in item.items() if k != "text"},
            )
            for i, item in enumerate(data)
            if item.get("text")
        ]
    return [
        Document(
            id=doc_id,
            text=data.get("text", ""),
            source=source,
            metadata={k: v for k, v in data.items() if k != "text"},
        )
    ]


def iter_documents(documents_dir: Path) -> Iterator[Document]:
    """Lazy iterator over documents (memory-efficient for large corpora)."""
    for doc in load_documents(documents_dir):
        yield doc
