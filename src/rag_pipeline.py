"""End-to-end RAG pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.chunking import chunk_records
from src.confidence import ConfidenceScorer, ConfidenceResult
from src.config import Settings, get_settings
from src.data_loader import load_document_records
from src.embeddings import EmbeddingModel
from src.generator import GenerationResult, Generator
from src.reranker import NoOpReranker, Reranker
from src.retriever import Retriever, RetrievedChunk
from src.vector_store import VectorStore


@dataclass
class Citation:
    """A source citation attached to an answer."""

    source: str
    chunk_id: str
    excerpt: str
    score: float


@dataclass
class RAGResponse:
    """Complete RAG pipeline output."""

    question: str
    answer: str
    citations: list[Citation]
    confidence: ConfidenceResult
    retrieved_chunks: list[RetrievedChunk] = field(repr=False)
    abstained: bool = False


class RAGPipeline:
    """Orchestrates indexing and question answering."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.embedding_model = EmbeddingModel(
            self.settings.embedding_model,
            device=self.settings.embedding_device,
        )
        self.vector_store = VectorStore(self.settings.vector_store_dir)
        self.retriever = Retriever(self.vector_store, self.embedding_model, self.settings)
        self.reranker = (
            Reranker(self.settings.reranker_model, self.settings)
            if self.settings.use_reranker
            else NoOpReranker()
        )
        self.generator = Generator(self.settings)
        self.confidence_scorer = ConfidenceScorer(self.settings)

    def build_index(self, documents_dir=None) -> int:
        """Load documents, chunk, embed, and index. Returns chunk count."""
        docs_dir = documents_dir or self.settings.documents_dir
        records = load_document_records(docs_dir)
        chunks = chunk_records(
            records,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        return self.vector_store.add_chunks(chunks, embeddings)

    def ask(self, question: str) -> RAGResponse:
        """Answer a question with citations, confidence, and optional abstention."""
        retrieved = self.retriever.retrieve(question)
        reranked = self.reranker.rerank(question, retrieved)

        if not reranked:
            confidence = ConfidenceResult(
                score=0.0,
                should_abstain=True,
                reasons=["no_relevant_chunks"],
            )
            return RAGResponse(
                question=question,
                answer="I don't have sufficient evidence to answer this question.",
                citations=[],
                confidence=confidence,
                retrieved_chunks=[],
                abstained=True,
            )

        generation = self.generator.generate(question, reranked)
        confidence = self.confidence_scorer.score(question, generation.answer, reranked)

        abstained = confidence.should_abstain
        answer = (
            "I don't have sufficient evidence to answer this question."
            if abstained
            else generation.answer
        )

        citations = [
            Citation(
                source=c.source,
                chunk_id=c.id,
                excerpt=c.text[:200] + ("..." if len(c.text) > 200 else ""),
                score=c.score,
            )
            for c in reranked
        ]

        return RAGResponse(
            question=question,
            answer=answer,
            citations=citations,
            confidence=confidence,
            retrieved_chunks=reranked,
            abstained=abstained,
        )
