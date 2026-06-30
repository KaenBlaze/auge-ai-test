"""End-to-end RAG pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.chunking import chunk_records
from src.confidence import ConfidenceResult, ConfidenceScorer, abstention_result
from src.config import Settings, get_settings
from src.data_loader import load_corpus_records
from src.embeddings import EmbeddingModel
from src.generator import ABSTENTION_PHRASE, Generator
from src.reranker import create_reranker
from src.retriever import Retriever, RetrievedChunk
from src.vector_store import VectorStore


@dataclass
class Citation:
    """A source citation attached to an answer."""

    document: str
    source_id: str
    fragment: str

    @property
    def source(self) -> str:
        """Backward-compatible alias used by evaluation metrics."""
        return self.document

    def to_dict(self) -> dict[str, str]:
        return {
            "document": self.document,
            "source_id": self.source_id,
            "fragment": self.fragment,
        }


@dataclass
class RAGResponse:
    """Complete RAG pipeline output."""

    question: str
    answer: str
    citations: list[Citation]
    confidence: ConfidenceResult
    abstained: bool
    retrieved_chunks: list[RetrievedChunk] = field(repr=False, default_factory=list)

    @property
    def reason(self) -> str:
        return self.confidence.reason

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical pipeline response schema."""
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "confidence": self.confidence.confidence,
            "abstained": self.abstained,
            "reason": self.confidence.reason,
        }


class RAGPipeline:
    """Orchestrates indexing and question answering."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedding_model = EmbeddingModel(
            self.settings.embedding_model,
            device=self.settings.embedding_device,
        )
        self.vector_store = VectorStore(self.settings.faiss_index_dir)
        self.retriever = Retriever(self.vector_store, self.embedding_model, self.settings)
        self.reranker = create_reranker(self.settings)
        self.generator = Generator(self.settings)
        self.confidence_scorer = ConfidenceScorer(self.settings)

    def build_index(self, documents_dir=None) -> int:
        """Load documents, chunk, embed, and index. Returns chunk count."""
        docs_dir = documents_dir or self.settings.documents_dir
        records = load_corpus_records(
            docs_dir,
            csv_paths=[self.settings.historical_results_path],
        )
        chunks = chunk_records(
            records,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        return self.vector_store.add_chunks(chunks, embeddings)

    def ask(self, question: str) -> RAGResponse:
        """Run the full RAG pipeline for a question."""
        question = question.strip()
        if not question:
            confidence = abstention_result(0.0, "empty_question")
            return self._abstain_response(question, confidence, [])

        # 1. Retrieve candidate chunks
        retrieved = self.retriever.retrieve(question)

        # 2. Rerank chunks
        reranked = self.reranker.rerank(question, retrieved)

        # 3. Decide if retrieval evidence is sufficient before generation
        retrieval_abstention = self.confidence_scorer.assess_retrieval_evidence(reranked)
        if retrieval_abstention is not None:
            return self._abstain_response(question, retrieval_abstention, reranked)

        # 4. Generate answer from evidence
        generation = self.generator.generate(question, reranked)

        # 5. Score answer support and decide final abstention
        confidence = self.confidence_scorer.score(question, generation.answer, reranked)
        abstained = confidence.abstained
        answer = ABSTENTION_PHRASE if abstained else generation.answer

        # 6. Return answer with citations, confidence, abstention, and reason
        citations = _build_citations(reranked)
        return RAGResponse(
            question=question,
            answer=answer,
            citations=citations,
            confidence=confidence,
            abstained=abstained,
            retrieved_chunks=reranked,
        )

    def _abstain_response(
        self,
        question: str,
        confidence: ConfidenceResult,
        chunks: list[RetrievedChunk],
    ) -> RAGResponse:
        return RAGResponse(
            question=question,
            answer=ABSTENTION_PHRASE,
            citations=_build_citations(chunks),
            confidence=confidence,
            abstained=True,
            retrieved_chunks=chunks,
        )


def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    """Build deduplicated citations from retrieved chunks."""
    citations: list[Citation] = []
    seen: set[tuple[str, str, str]] = set()

    for chunk in chunks:
        key = (chunk.document, chunk.source_id, chunk.fragment)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            Citation(
                document=chunk.document,
                source_id=chunk.source_id,
                fragment=chunk.fragment,
            )
        )
    return citations
