"""Application configuration loaded from environment variables."""

from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    vector_store_dir: Path = Field(default=Path("./data/vector_store"))
    golden_seed_path: Path = Field(default=Path("./data/golden_seed.jsonl"))
    historical_results_path: Path = Field(default=Path("./data/historical_results.csv"))

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # Chunking (token counts; see src/chunking.py for rationale)
    chunk_size: int = 640
    chunk_overlap: int = 100

    # Retrieval
    top_k_retrieve: int = 10
    top_k_rerank: int = 5
    similarity_threshold: float = 0.3

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

    # Generator (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    generator_temperature: float = 0.1
    generator_max_tokens: int = 512

    # Confidence / abstention
    confidence_threshold: float = 0.5
    abstain_on_low_confidence: bool = True

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
