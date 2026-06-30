"""Local LLM answer generation via Ollama."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Settings, get_settings
from src.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a precise assistant that answers questions using ONLY the provided context.

Rules:
1. Base your answer strictly on the context below.
2. Cite sources using [source: <filename>] inline where claims are made.
3. If the context does not contain enough information, say "I don't have sufficient evidence to answer this question."
4. Do not invent facts or use outside knowledge.
"""


@dataclass
class GenerationResult:
    """Output from the generator."""

    answer: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class Generator:
    """Generate answers using a local Ollama model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.settings.ollama_base_url)
        return self._client

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> GenerationResult:
        """Generate an answer grounded in retrieved context.

        TODO: Support llama.cpp as an alternative backend.
        TODO: Structured output (JSON) for citations and confidence hints.
        TODO: Streaming responses for API use.
        """
        context = self._format_context(context_chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

        client = self._get_client()
        response = client.chat(
            model=self.settings.ollama_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": self.settings.generator_temperature,
                "num_predict": self.settings.generator_max_tokens,
            },
        )

        answer = response["message"]["content"].strip()
        return GenerationResult(answer=answer)

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "(No relevant context found.)"
        parts = []
        for i, chunk in enumerate(chunks, 1):
            parts.append(f"[{i}] source: {chunk.source}\n{chunk.text}")
        return "\n\n".join(parts)
