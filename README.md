# AUGE Intelligence — Local RAG System

A take-home AI/ML Engineer project: a **fully local Retrieval-Augmented Generation (RAG)** system that answers questions over a synthetic corpus with **citations**, **confidence scores**, and **honest abstention** when evidence is insufficient.

No paid external APIs. All embeddings, retrieval, reranking, and generation run on open-weight models and local infrastructure.

---

## Project Overview

The system ingests Markdown documents, chunks them with heading/paragraph awareness, embeds them with sentence-transformers, indexes them in FAISS, and answers questions via a configurable local LLM (Ollama, vLLM, or Hugging Face Transformers).

Every answer includes:
- Source citations (`document`, `source_id`, `fragment`)
- A confidence score derived from retrieval and citation signals
- Abstention with reason when evidence is weak or unsupported

An evaluation harness measures retrieval quality, citation correctness, hallucination rate, abstention behavior, and confidence calibration. A before/after experiment compares retrieval-only vs retrieval + reranking.

---

## Architecture

```
data/documents/*.md
        │
        ▼
  data_loader ──► chunking (heading/paragraph aware, tiktoken-sized)
        │
        ▼
  embeddings (BAAI/bge-small-en-v1.5)
        │
        ▼
  FAISS index + JSON metadata  (storage/faiss_index/)
        │
        ▼
  Question ──► retriever (dense top-k)
        │
        ▼
  reranker (cross-encoder, optional)
        │
        ▼
  confidence pre-check ──► generator (local LLM)
        │
        ▼
  confidence post-check ──► answer + citations + abstention
```

| Layer | Module | Technology |
|-------|--------|------------|
| Data | `src/data_loader.py` | Markdown parsing, golden seed, CSV |
| Chunking | `src/chunking.py` | Section/paragraph merge, tiktoken limits |
| Embeddings | `src/embeddings.py` | sentence-transformers (BGE-small) |
| Vector store | `src/vector_store.py` | FAISS `IndexFlatIP` + JSON sidecar |
| Retrieval | `src/retriever.py` | Dense cosine similarity |
| Reranking | `src/reranker.py` | BGE reranker / ms-marco fallback |
| Generation | `src/generator.py` | Ollama / vLLM / Transformers |
| Confidence | `src/confidence.py` | Multi-signal abstention |
| Pipeline | `src/rag_pipeline.py` | End-to-end orchestration |
| API | `src/api.py` | FastAPI `POST /ask` |
| Evaluation | `eval/` | Metrics, experiment, results |

---

## Installation

**Requirements:** Python 3.10+, ~4 GB RAM minimum (CPU), more if running a 7B LLM locally.

```bash
# Clone and enter project
cd auge-ai-test

# Virtual environment
python -m venv .venv
source .venv/bin/activate

# Dependencies
pip install -r requirements.txt

# Configuration (no secrets required for local-only stack)
cp .env.example .env

# Local LLM via Ollama (default backend)
# Install Ollama: https://ollama.ai
ollama pull qwen2.5-7b
```

Optional backends (set in `.env`):
- `MODEL_BACKEND=ollama` — default
- `MODEL_BACKEND=vllm` — OpenAI-compatible local server
- `MODEL_BACKEND=transformers` — requires `pip install transformers torch accelerate`

---

## Build the Index

```bash
python scripts/build_index.py
```

This will:
1. Load Markdown from `data/documents/`
2. Chunk with heading/paragraph awareness
3. Embed with `BAAI/bge-small-en-v1.5`
4. Build and save FAISS index to `storage/faiss_index/index.faiss`
5. Save chunk metadata to `storage/faiss_index/metadata.json`

---

## Ask One Question

```bash
python scripts/run_question.py "What sensors does the AUGE platform support?"
```

JSON output:

```bash
python scripts/run_question.py "What is the campaign spending reporting deadline?" --json
```

CLI alternative:

```bash
python -m src.cli ask "What is the primary purpose of the AUGE monitoring system?" --json
```

Example response schema:

```json
{
  "question": "...",
  "answer": "...",
  "citations": [
    {"document": "sensor_catalog.md", "source_id": "sensor_catalog", "fragment": "..."}
  ],
  "confidence": 0.72,
  "abstained": false,
  "reason": "sufficient_evidence"
}
```

---

## Run the API

```bash
python -m src.api
# or
python -m src.cli serve
```

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which sensors does the AUGE platform support?"}'
```

Health check: `GET http://localhost:8000/health`

---

## Run Evaluation

```bash
# Full golden-set evaluation
python eval/evaluate.py --golden-set data/golden_seed.jsonl

# Before/after reranking experiment
python eval/experiment.py
```

Outputs:
- `eval/results.json` — per-example metrics and summary
- `eval/experiment_results.json` — baseline vs reranked comparison
- `REPORT.md` — experiment section auto-updated

Utility commands:

```bash
python scripts/inspect_data.py          # document/record counts
python scripts/preview_chunks.py        # chunk preview with metadata
python -m src.cli inspect-data
python -m src.cli preview-chunks
```

---

## Models & Infrastructure

| Component | Default | Notes |
|-----------|---------|-------|
| **Embedding model** | `BAAI/bge-small-en-v1.5` | 384-dim, CPU-friendly; falls back to `all-MiniLM-L6-v2` |
| **Reranker** | `BAAI/bge-reranker-base` | Falls back to `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | `qwen2.5-7b` via Ollama | Also supports vLLM and Transformers |
| **Vector store** | FAISS (`IndexFlatIP`) | Persistent at `storage/faiss_index/` |
| **Hardware tested** | Linux x86_64, CPU-only | Embeddings + reranker on CPU; Ollama for generation |

---

## Known Limitations

- **Tiny demo corpus** (2 documents, 9 chunks) — metrics saturate quickly; not representative of production scale.
- **CPU inference** — embedding and reranking are slow on large corpora without GPU.
- **Citation correctness** depends on the LLM following `[source: <document>]` format; strict `REQUIRE_CITATIONS=true` causes false abstention when the model omits inline citations.
- **No hybrid retrieval** — dense-only; exact keyword/SKU matches may be missed.
- **Confidence heuristics** — not fully calibrated; tuned thresholds can over-abstain on answerable questions.
- **Synchronous API** — no streaming; long generations block the request.
- **Golden set size** — 3 examples; statistical conclusions require a larger eval set.

See `REPORT.md` and `TRADEOFFS.md` for design rationale and v2 roadmap.

---

## Project Structure

```
src/                  # Core RAG pipeline
eval/                 # Evaluation harness and experiment
scripts/              # build_index, run_question, inspect_data, preview_chunks
data/documents/       # Synthetic Markdown corpus
data/golden_seed.jsonl
storage/faiss_index/  # Generated FAISS index + metadata (gitignored)
tests/                # 67 unit/integration tests
README.md
REPORT.md
TRADEOFFS.md
```

---

## Final Checklist

- [x] Index builds successfully
- [x] Question runs end to end
- [x] Citations are returned
- [x] Abstention works for no-evidence cases
- [x] Evaluation script runs
- [x] Before/after experiment has numbers
- [x] README is clear
- [x] No paid external APIs are used
- [x] No secrets are committed

---

## License

Internal use — AUGE Intelligence take-home assignment.
