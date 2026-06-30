"""Application configuration loaded from environment variables."""

from pathlib import Path
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ModelBackend = Literal["ollama", "vllm", "transformers"]


class Settings(BaseSettings):
    """Central configuration for the local RAG pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    data_dir: Path = Field(default=Path("./data"))
    documents_dir: Path = Field(default=Path("./data/documents"))
    faiss_index_dir: Path = Field(default=Path("./storage/faiss_index"))
    golden_seed_path: Path = Field(default=Path("./data/golden_seed.jsonl"))
    historical_results_path: Path = Field(default=Path("./data/historical_results.csv"))

    # Embeddings (local sentence-transformers; BGE preferred, MiniLM fallback)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"

    # Chunking (token counts; see src/chunking.py for rationale)
    chunk_size: int = 640
    chunk_overlap: int = 100

    # Retrieval
    top_k_retrieve: int = 10
    top_k_rerank: int = 5
    similarity_threshold: float = 0.3

    # Reranker (local cross-encoder; BGE preferred, ms-marco fallback)
    reranker_model: str = "BAAI/bge-reranker-base"
    use_reranker: bool = True

    # Generator (local open-weight models only)
    model_backend: ModelBackend = "ollama"
    model_name: str = "qwen2.5-7b"
    generator_temperature: float = 0.1
    generator_max_tokens: int = 512

    # Ollama backend
    ollama_base_url: str = "http://localhost:11434"

    # vLLM OpenAI-compatible backend
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_api_key: str = "EMPTY"

    # Hugging Face Transformers backend
    transformers_device: str = "cpu"
    transformers_torch_dtype: str = "auto"

    # Confidence / abstention
    confidence_threshold: float = 0.45
    min_retrieval_score: float = 0.30
    min_rerank_score: float = 0.20
    min_supporting_chunks: int = 1
    min_citation_support: float = 0.20
    min_fragment_agreement: float = 0.0  # soft signal only; diverse sections are OK
    require_citations: bool = True
    abstain_on_low_confidence: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
