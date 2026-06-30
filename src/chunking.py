"""Heading-aware, paragraph-aware chunking for Markdown records."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from src.data_loader import Document, DocumentRecord

# Default chunk_size=640 tokens sits in the middle of the 500–800 token target range.
# At this size, chunks are large enough to capture a complete idea (heading + a few
# paragraphs) for embedding and generation, but small enough to keep citations
# localized and reduce noise in retrieval. We measure with tiktoken instead of
# blind fixed-character windows so chunk boundaries track model context units.
# Default overlap=100 tokens (~80–120 target) preserves continuity across chunk
# boundaries when answers span adjacent paragraphs.
DEFAULT_CHUNK_SIZE = 640
DEFAULT_CHUNK_OVERLAP = 100

PARAGRAPH_SEPARATOR = "\n\n"
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """A retrieval-ready text chunk with provenance metadata."""

    chunk_id: str
    document: str
    source_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Alias used by the vector store and retriever."""
        return self.chunk_id

    @property
    def source(self) -> str:
        """Alias for the source document filename."""
        return self.document

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document": self.document,
            "source_id": self.source_id,
            "text": self.text,
            "metadata": self.metadata,
        }


@lru_cache(maxsize=1)
def _get_encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    if not text:
        return 0
    return len(_get_encoding().encode(text))


def _tail_tokens(text: str, token_count: int) -> str:
    tokens = _get_encoding().encode(text)
    if len(tokens) <= token_count:
        return text
    return _get_encoding().decode(tokens[-token_count:])


def chunk_records(
    records: list[DocumentRecord],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk normalized document records using section and paragraph boundaries."""
    if not records:
        return []

    chunks: list[Chunk] = []
    chunk_counters: dict[str, int] = {}

    for section_key, section_records in _group_by_section(records).items():
        document, _section_heading = section_key
        section_chunks = _chunk_section(
            section_records,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for section_chunk in section_chunks:
            source_id = section_records[0].source_id
            counter = chunk_counters.get(source_id, 0)
            section_slug = _slugify(section_chunk["section_heading"])
            chunk_id = f"{source_id}__{section_slug}__chunk_{counter}"
            chunk_counters[source_id] = counter + 1

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document=document,
                    source_id=source_id,
                    text=section_chunk["text"],
                    metadata={
                        "section_heading": section_chunk["section_heading"],
                        "chunk_index": counter,
                        "paragraph_indices": section_chunk["paragraph_indices"],
                        "line_start": section_chunk["line_start"],
                        "line_end": section_chunk["line_end"],
                        "token_count": _token_count(section_chunk["text"]),
                        **section_chunk["metadata"],
                    },
                )
            )
    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk legacy Document objects by converting them back to records."""
    records = [
        DocumentRecord(
            document=doc.source,
            source_id=str(doc.metadata.get("source_id", doc.source)),
            text=doc.text,
            metadata=dict(doc.metadata),
        )
        for doc in documents
    ]
    return chunk_records(records, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def preview_chunks(
    records: list[DocumentRecord],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return the first *limit* chunks as serializable dicts for inspection."""
    chunks = chunk_records(records, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [chunk.to_dict() for chunk in chunks[:limit]]


def _group_by_section(
    records: list[DocumentRecord],
) -> dict[tuple[str, str | None], list[DocumentRecord]]:
    """Group records by document and section heading, preserving order."""
    grouped: dict[tuple[str, str | None], list[DocumentRecord]] = {}
    for record in records:
        section_heading = record.metadata.get("section_heading")
        key = (record.document, section_heading)
        grouped.setdefault(key, []).append(record)
    return grouped


def _chunk_section(
    records: list[DocumentRecord],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Merge paragraphs within one section into size-limited chunks."""
    section_heading = records[0].metadata.get("section_heading")
    section_chunks: list[dict[str, Any]] = []
    current_paragraphs: list[DocumentRecord] = []
    overlap_prefix = ""

    def flush(paragraphs: list[DocumentRecord], prefix: str = "") -> None:
        merged_body = _merge_parts([prefix, *[record.text for record in paragraphs]])
        if not merged_body:
            return
        text = _format_chunk_text(section_heading, merged_body)
        section_chunks.append(
            {
                "text": text,
                "section_heading": section_heading,
                "paragraph_indices": [record.metadata.get("paragraph_index") for record in paragraphs],
                "line_start": paragraphs[0].metadata.get("line_start") if paragraphs else None,
                "line_end": paragraphs[-1].metadata.get("line_end") if paragraphs else None,
                "metadata": {
                    "filename": records[0].document,
                    "source_id": records[0].source_id,
                },
            }
        )

    for record in records:
        paragraph_text = record.text.strip()
        if not paragraph_text:
            continue

        if _token_count(paragraph_text) > chunk_size:
            if current_paragraphs:
                flush(current_paragraphs, overlap_prefix)
                overlap_prefix = _chunk_overlap_text(section_chunks[-1]["text"], chunk_overlap)
                current_paragraphs = []
            split_chunks = _split_oversized_paragraph(
                record,
                section_heading=section_heading,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                initial_prefix=overlap_prefix,
            )
            section_chunks.extend(split_chunks)
            if section_chunks:
                overlap_prefix = _chunk_overlap_text(section_chunks[-1]["text"], chunk_overlap)
            else:
                overlap_prefix = ""
            continue

        candidate_paragraphs = current_paragraphs + [record]
        candidate_text = _build_chunk_text(section_heading, candidate_paragraphs, overlap_prefix)
        if current_paragraphs and _token_count(candidate_text) > chunk_size:
            flush(current_paragraphs, overlap_prefix)
            overlap_prefix = _chunk_overlap_text(section_chunks[-1]["text"], chunk_overlap)
            current_paragraphs = [record]
        else:
            current_paragraphs = candidate_paragraphs

    if current_paragraphs:
        flush(current_paragraphs, overlap_prefix)

    return section_chunks


def _split_oversized_paragraph(
    record: DocumentRecord,
    section_heading: str | None,
    chunk_size: int,
    chunk_overlap: int,
    initial_prefix: str = "",
) -> list[dict[str, Any]]:
    """Split a long paragraph on sentence boundaries, then token windows."""
    pieces = _split_sentences(record.text)
    if len(pieces) <= 1:
        pieces = [record.text]

    chunks: list[dict[str, Any]] = []
    current_sentences: list[str] = []
    overlap_prefix = initial_prefix

    def flush_sentences(sentences: list[str], prefix: str = "") -> None:
        merged_body = _merge_parts([prefix, *sentences])
        if not merged_body:
            return
        text = _format_chunk_text(section_heading, merged_body)
        chunks.append(
            {
                "text": text,
                "section_heading": section_heading,
                "paragraph_indices": [record.metadata.get("paragraph_index")],
                "line_start": record.metadata.get("line_start"),
                "line_end": record.metadata.get("line_end"),
                "metadata": {
                    "filename": record.document,
                    "source_id": record.source_id,
                    "split_paragraph": True,
                },
            }
        )

    for sentence in pieces:
        if _token_count(sentence) > chunk_size:
            if current_sentences:
                flush_sentences(current_sentences, overlap_prefix)
                overlap_prefix = _chunk_overlap_text(chunks[-1]["text"], chunk_overlap)
                current_sentences = []
            for index, token_chunk in enumerate(_split_text_by_tokens(sentence, chunk_size, chunk_overlap)):
                prefix = overlap_prefix if index == 0 else ""
                flush_sentences([token_chunk], prefix)
                overlap_prefix = _chunk_overlap_text(chunks[-1]["text"], chunk_overlap)
            continue

        candidate_sentences = current_sentences + [sentence]
        candidate_text = _format_chunk_text(
            section_heading,
            _merge_parts([overlap_prefix, *candidate_sentences]),
        )
        if current_sentences and _token_count(candidate_text) > chunk_size:
            flush_sentences(current_sentences, overlap_prefix)
            overlap_prefix = _chunk_overlap_text(chunks[-1]["text"], chunk_overlap)
            current_sentences = [sentence]
        else:
            current_sentences = candidate_sentences

    if current_sentences:
        flush_sentences(current_sentences, overlap_prefix)

    return chunks


def _split_text_by_tokens(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    tokens = _get_encoding().encode(text)
    if len(tokens) <= chunk_size:
        return [text]
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_text = _get_encoding().decode(tokens[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        if end >= len(tokens):
            break
        start = max(end - chunk_overlap, start + 1)
    return chunks


def _split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def _merge_parts(parts: list[str]) -> str:
    return PARAGRAPH_SEPARATOR.join(part.strip() for part in parts if part and part.strip())


def _build_chunk_text(
    section_heading: str | None,
    paragraphs: list[DocumentRecord],
    overlap_prefix: str,
) -> str:
    merged_body = _merge_parts([overlap_prefix, *[record.text for record in paragraphs]])
    return _format_chunk_text(section_heading, merged_body)


def _format_chunk_text(section_heading: str | None, body: str) -> str:
    body = body.strip()
    if not body:
        return ""
    if section_heading:
        return f"## {section_heading}{PARAGRAPH_SEPARATOR}{body}"
    return body


def _chunk_overlap_text(text: str, chunk_overlap: int) -> str:
    if not text or chunk_overlap <= 0:
        return ""
    return _tail_tokens(text, chunk_overlap)


def _slugify(value: str | None) -> str:
    if not value:
        return "root"
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug[:40] or "root"
