"""Split documents into retrieval-sized chunks."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.data_loader import Document


@dataclass
class Chunk:
    """A text chunk derived from a parent document."""

    id: str
    text: str
    document_id: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split documents into overlapping character-based chunks.

    TODO: Replace character splitting with token-aware chunking (tiktoken).
    TODO: Preserve section headings and table structure where possible.
    """
    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(_chunk_single(doc, chunk_size, chunk_overlap))
    return chunks


def _chunk_single(doc: Document, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Split one document into chunks."""
    text = doc.text.strip()
    if not text:
        return []

    result: list[Chunk] = []
    start = 0
    index = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end].strip()
        if chunk_text:
            result.append(
                Chunk(
                    id=f"{doc.id}__chunk_{index}",
                    text=chunk_text,
                    document_id=doc.id,
                    source=doc.source,
                    chunk_index=index,
                    metadata=dict(doc.metadata),
                )
            )
            index += 1
        if end >= len(text):
            break
        start = end - chunk_overlap
    return result
