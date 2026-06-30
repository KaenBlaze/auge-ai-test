"""Tests for local LLM generation."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings
from src.generator import (
    ABSTENTION_PHRASE,
    Generator,
    OllamaBackend,
    VLLMBackend,
    create_generation_backend,
    parse_answer,
)
from src.retriever import RetrievalResult


@pytest.fixture
def sample_chunks() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="sensor_catalog__sensor_catalog__chunk_0",
            document="sensor_catalog.md",
            source_id="sensor_catalog",
            fragment="The AUGE platform supports temperature and pressure sensors.",
            retrieval_score=0.9,
            rerank_score=0.9,
        )
    ]


def test_parse_answer_strips_answer_prefix():
    assert parse_answer("Answer: Supported sensors include temperature.") == (
        "Supported sensors include temperature."
    )


def test_parse_answer_parses_json_block():
    raw = 'Thoughts...\n{"answer": "Temperature sensors are supported."}'
    assert parse_answer(raw) == "Temperature sensors are supported."


def test_parse_answer_returns_abstention_for_empty_output():
    assert parse_answer("") == ABSTENTION_PHRASE


def test_format_context_includes_source_and_fragment(sample_chunks):
    prompt = Generator._build_user_prompt("Which sensors?", sample_chunks)
    assert "sensor_catalog.md" in prompt
    assert "temperature and pressure sensors" in prompt
    assert "Context fragments:" in prompt


def test_ollama_backend_generate(monkeypatch):
    settings = Settings(model_backend="ollama", model_name="qwen2.5-7b")
    backend = OllamaBackend(settings)
    mock_client = MagicMock()
    mock_client.chat.return_value = {
        "message": {"content": "Answer: Temperature sensors [source: sensor_catalog.md]"},
        "prompt_eval_count": 12,
        "eval_count": 8,
    }
    monkeypatch.setattr(backend, "_get_client", lambda: mock_client)

    result = backend.generate([{"role": "user", "content": "hi"}])
    assert "Temperature sensors" in result["content"]
    assert result["prompt_tokens"] == 12
    assert result["completion_tokens"] == 8


def test_vllm_backend_generate(monkeypatch):
    settings = Settings(model_backend="vllm", model_name="Qwen/Qwen2.5-7B-Instruct")
    backend = VLLMBackend(settings)

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{"message": {"content": "Temperature sensors are supported."}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            }

    class FakeClient:
        def __init__(self, timeout=120.0):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def post(self, url, json, headers):
            assert url.endswith("/chat/completions")
            assert json["model"] == "Qwen/Qwen2.5-7B-Instruct"
            return FakeResponse()

    monkeypatch.setattr("httpx.Client", FakeClient)

    result = backend.generate([{"role": "user", "content": "hi"}])
    assert result["content"] == "Temperature sensors are supported."
    assert result["prompt_tokens"] == 20


def test_generator_uses_backend_and_parses_answer(monkeypatch, sample_chunks):
    settings = Settings(model_backend="ollama", model_name="qwen2.5-7b")
    generator = Generator(settings)

    mock_backend = MagicMock()
    mock_backend.name = "ollama"
    mock_backend.generate.return_value = {
        "content": "Answer: Temperature sensors [source: sensor_catalog.md]",
        "prompt_tokens": 30,
        "completion_tokens": 12,
    }
    generator.backend = mock_backend

    result = generator.generate("Which sensors are supported?", sample_chunks)
    assert result.backend == "ollama"
    assert result.model_name == "qwen2.5-7b"
    assert result.raw_output.startswith("Answer:")
    assert result.answer == "Temperature sensors [source: sensor_catalog.md]"
    assert result.parsed_answer == result.answer


def test_create_generation_backend_variants():
    assert create_generation_backend(Settings(model_backend="ollama")).__class__.__name__ == "OllamaBackend"
    assert create_generation_backend(Settings(model_backend="vllm")).__class__.__name__ == "VLLMBackend"
    assert (
        create_generation_backend(Settings(model_backend="transformers")).__class__.__name__
        == "TransformersBackend"
    )


def test_create_generation_backend_rejects_unknown():
    settings = Settings()
    settings.model_backend = "ollama"  # type: ignore[assignment]
    object.__setattr__(settings, "model_backend", "openai")

    with pytest.raises(ValueError, match="Unsupported MODEL_BACKEND"):
        create_generation_backend(settings)
