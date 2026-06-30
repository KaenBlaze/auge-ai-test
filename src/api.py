"""FastAPI HTTP interface for the RAG pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_settings
from src.rag_pipeline import RAGPipeline


pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize pipeline on startup."""
    global pipeline
    pipeline = RAGPipeline()
    yield
    pipeline = None


app = FastAPI(
    title="AUGE Intelligence RAG API",
    description="Local RAG system with citations, confidence, and abstention.",
    version="0.1.0",
    lifespan=lifespan,
)


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


class CitationResponse(BaseModel):
    document: str
    source_id: str
    fragment: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationResponse]
    confidence: float
    abstained: bool
    reason: str


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check with index status."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return HealthResponse(
        status="ok",
        indexed_chunks=pipeline.vector_store.count(),
    )


@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest) -> AnswerResponse:
    """Answer a question using the RAG pipeline."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    if pipeline.vector_store.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Index is empty. Run scripts/build_index.py first.",
        )

    result = pipeline.ask(request.question)
    payload = result.to_dict()
    return AnswerResponse(**payload)


def main() -> None:
    """Run the API server."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
