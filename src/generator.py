"""Local open-weight LLM generation with multiple backend support."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from src.config import Settings, get_settings
from src.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

ABSTENTION_PHRASE = "I don't know based on the provided corpus."

SYSTEM_PROMPT = f"""You are a precise assistant that answers questions using ONLY the provided context.

Rules:
1. Answer only using the provided context fragments.
2. Cite source fragments inline using [source: <document>] wherever a claim is made.
3. If the evidence is insufficient, respond exactly with: "{ABSTENTION_PHRASE}"
4. Do not invent facts or use outside knowledge.
5. Keep the answer concise and grounded in the cited fragments.
"""


@dataclass
class GenerationResult:
    """Output from the generator."""

    raw_output: str
    answer: str
    backend: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    @property
    def parsed_answer(self) -> str:
        """Alias for the parsed answer text."""
        return self.answer


class GenerationBackend(ABC):
    """Interface for local model backends."""

    name: str

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Return a dict with at least `content` and optional token usage."""


class OllamaBackend(GenerationBackend):
    name = "ollama"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.settings.ollama_base_url)
        return self._client

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        response = self._get_client().chat(
            model=self.settings.model_name,
            messages=messages,
            options={
                "temperature": self.settings.generator_temperature,
                "num_predict": self.settings.generator_max_tokens,
            },
        )
        message = response.get("message", {})
        return {
            "content": message.get("content", "").strip(),
            "prompt_tokens": response.get("prompt_eval_count"),
            "completion_tokens": response.get("eval_count"),
        }


class VLLMBackend(GenerationBackend):
    """OpenAI-compatible chat completions against a local vLLM server."""

    name = "vllm"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        import httpx

        url = f"{self.settings.vllm_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.model_name,
            "messages": messages,
            "temperature": self.settings.generator_temperature,
            "max_tokens": self.settings.generator_max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.settings.vllm_api_key}"}

        with httpx.Client(timeout=120.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return {
            "content": choice.strip(),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }


class TransformersBackend(GenerationBackend):
    """Local Hugging Face Transformers causal LM backend."""

    name = "transformers"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tokenizer = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._tokenizer, self._model

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = self._resolve_dtype()
        tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_name,
            torch_dtype=dtype,
            device_map=self.settings.transformers_device,
        )
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        return tokenizer, model

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        import torch

        tokenizer, model = self._load()
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = _messages_to_plain_prompt(messages)

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.settings.generator_max_tokens,
                temperature=max(self.settings.generator_temperature, 1e-5),
                do_sample=self.settings.generator_temperature > 0,
            )

        generated_ids = output_ids[0][input_len:]
        content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return {
            "content": content,
            "prompt_tokens": int(input_len),
            "completion_tokens": int(generated_ids.shape[-1]),
        }

    def _resolve_dtype(self):
        import torch

        dtype_name = self.settings.transformers_torch_dtype.lower()
        if dtype_name == "auto":
            return "auto"
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return mapping.get(dtype_name, "auto")


class Generator:
    """Generate evidence-grounded answers using a configured local backend."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.backend = create_generation_backend(self.settings)

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> GenerationResult:
        """Generate an answer grounded in retrieved context."""
        user_prompt = self._build_user_prompt(query, context_chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        response = self.backend.generate(messages)
        raw_output = response.get("content", "").strip()
        answer = parse_answer(raw_output)

        return GenerationResult(
            raw_output=raw_output,
            answer=answer,
            backend=self.backend.name,
            model_name=self.settings.model_name,
            prompt_tokens=response.get("prompt_tokens"),
            completion_tokens=response.get("completion_tokens"),
        )

    @staticmethod
    def _build_user_prompt(query: str, context_chunks: list[RetrievedChunk]) -> str:
        context = Generator._format_context(context_chunks)
        return (
            "Context fragments:\n"
            f"{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above. Cite sources inline."
        )

    @staticmethod
    def _format_context(chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "(No relevant context found.)"
        parts = []
        for index, chunk in enumerate(chunks, 1):
            parts.append(
                f"[{index}] chunk_id: {chunk.chunk_id}\n"
                f"source: {chunk.document}\n"
                f"fragment:\n{chunk.fragment}"
            )
        return "\n\n".join(parts)


def create_generation_backend(settings: Settings | None = None) -> GenerationBackend:
    """Instantiate the configured local generation backend."""
    settings = settings or get_settings()
    backend = settings.model_backend.lower()

    if backend == "ollama":
        return OllamaBackend(settings)
    if backend == "vllm":
        return VLLMBackend(settings)
    if backend == "transformers":
        return TransformersBackend(settings)
    raise ValueError(
        f"Unsupported MODEL_BACKEND={settings.model_backend!r}. "
        "Expected one of: ollama, vllm, transformers."
    )


def parse_answer(raw_output: str) -> str:
    """Parse the model's raw output into a final answer when possible."""
    text = raw_output.strip()
    if not text:
        return ABSTENTION_PHRASE

    if text.lower().startswith("answer:"):
        return text.split(":", 1)[1].strip()

    json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            if isinstance(payload, dict) and payload.get("answer"):
                return str(payload["answer"]).strip()
        except json.JSONDecodeError:
            pass

    return text


def _messages_to_plain_prompt(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message["role"].upper()
        parts.append(f"{role}:\n{message['content']}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)
