# AUGE Intelligence — Local RAG System

A take-home AI/ML Engineer project: a **fully local Retrieval-Augmented Generation (RAG)** system that answers questions over a synthetic corpus with **citations**, **confidence scores**, and **honest abstention** when evidence is insufficient.

No paid external APIs. All embeddings, retrieval, reranking, and generation run on open-weight models and local infrastructure.

---

## Project Overview

The system ingests Markdown documents and tabular CSV data, chunks them with heading/paragraph awareness, embeds them with sentence-transformers, indexes them in FAISS, and answers questions via a configurable local LLM (Ollama, vLLM, or Hugging Face Transformers).

The bundled **Republic of Valdoria** synthetic corpus includes electoral law, campaign finance rules, budget reports, procurement policy, and regional metrics (`historical_results.csv`, 2021–2025).

Every answer includes:
- Source citations (`document`, `source_id`, `fragment`)
- A confidence score derived from retrieval and citation signals
- Abstention with reason when evidence is weak or unsupported

An evaluation harness measures retrieval quality, citation correctness, hallucination rate, abstention behavior, and confidence calibration. A before/after experiment compares retrieval-only vs retrieval + reranking.

---

## Architecture

```
data/documents/*.md
data/historical_results.csv
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
ollama pull qwen2.5:7b
```

Optional backends (set in `.env`):
- `MODEL_BACKEND=ollama` — default
- `MODEL_BACKEND=vllm` — OpenAI-compatible local server
- `MODEL_BACKEND=transformers` — requires `pip install transformers torch accelerate`

---

## Docker

Run the full stack (Ollama + RAG API) with Docker Compose:

```bash
docker compose up --build
```

First startup downloads:
- Hugging Face embedding/reranker models (cached in a Docker volume)
- Ollama LLM `qwen2.5:7b` (cached in `ollama_data` volume)
- Builds the FAISS index if `storage/faiss_index/` is empty

API: `http://localhost:8000`  
Ollama: `http://localhost:11434`

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the electoral silence period before election day?"}'
```

Useful commands:

```bash
docker compose up --build -d          # detached
docker compose logs -f api            # follow API logs
docker compose run --rm api build-index
docker compose run --rm api pull-model
docker compose down                   # stop services
docker compose down -v                # stop and remove volumes
```

**GPU (optional):** Uncomment the `deploy.resources` GPU block under `ollama` in `docker-compose.yml` if NVIDIA Container Toolkit is installed.

**Environment:** The `api` service sets `OLLAMA_BASE_URL=http://ollama:11434`. For local (non-Docker) runs, use `http://localhost:11434`.

---

## Build the Index

```bash
python scripts/build_index.py
```

This will:
1. Load Markdown from `data/documents/`
2. Load tabular rows from `data/historical_results.csv`
3. Chunk with heading/paragraph awareness
4. Embed with `BAAI/bge-small-en-v1.5`
5. Build and save FAISS index to `storage/faiss_index/index.faiss`
6. Save chunk metadata to `storage/faiss_index/metadata.json`

---

## Dataset

| Path | Description |
|------|-------------|
| `data/documents/` | 9 Valdoria laws, policies, and reports |
| `data/historical_results.csv` | Regional turnout and budget metrics (2021–2025) |
| `data/golden_set.jsonl` | 26 evaluation questions (factual, tabular, unanswerable) |
| `data/golden_seed.jsonl` | Alias of golden_set (same content) |
| `data/quick_test_questions.md` | Manual smoke-test prompts |
| `data/README_data.md` | Dataset schema and notes |

See `data/quick_test_questions.md` for answerable vs should-abstain examples.

---

## Ask One Question

```bash
python scripts/run_question.py "What is the electoral silence period before election day?"
```

JSON output:

```bash
python scripts/run_question.py "Which region had the lowest turnout in 2025?" --json
```

CLI alternative:

```bash
python -m src.cli ask "What disclosure is required for campaign expenses above V\$10,000?" --json
```

Example response schema:

```json
{
  "question": "Which region had the lowest turnout in 2025?",
  "answer": "West Region had the lowest turnout in 2025 at 49%.",
  "citations": [
    {
      "document": "REP-2026-003_Turnout_and_Service_Report.md",
      "source_id": "REP-2026-003",
      "fragment": "..."
    },
    {
      "document": "historical_results.csv",
      "source_id": "historical_results",
      "fragment": "..."
    }
  ],
  "confidence": 0.77,
  "abstained": false,
  "reason": "sufficient_evidence"
}
```

Unanswerable questions (not in the corpus) should abstain, for example:

```bash
python scripts/run_question.py "Who won the 2026 presidential election in Valdoria?" --json
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
  -d '{"question": "What is the electoral silence period before election day?"}'
```

Health check: `GET http://localhost:8000/health`

---

## Run Evaluation

```bash
# Full golden-set evaluation
python eval/evaluate.py --golden-set data/golden_set.jsonl

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
| **LLM** | `qwen2.5:7b` via Ollama | Also supports vLLM and Transformers |
| **Vector store** | FAISS (`IndexFlatIP`) | Persistent at `storage/faiss_index/` |
| **Hardware tested** | Linux x86_64, CPU-only | Embeddings + reranker on CPU; Ollama for generation |

---

## Known Limitations

- **Demo corpus size** (9 documents + CSV, ~39 chunks) — useful for evaluation design, not production scale.
- **CPU inference** — embedding and reranking are slow on large corpora without GPU.
- **Citation correctness** depends on the LLM following `[source: <document>]` format; fragment grounding is used as a fallback when inline citations are missing.
- **No hybrid retrieval** — dense-only; exact keyword/SKU matches may be missed.
- **Confidence heuristics** — not fully calibrated; thresholds may still over-abstain on edge cases.
- **Synchronous API** — no streaming; long generations block the request.
- **Golden set size** — 26 examples in `data/golden_set.jsonl`; extend further for production CI gates.
- **Docker** — requires a host with a running Docker daemon; nested dev containers without systemd cannot run `docker compose` locally.

See `REPORT.md` and `TRADEOFFS.md` for design rationale and v2 roadmap.  
See `TEST_PROJECT_REPORT.md` for the submission report mapped to the official scoring criteria.

---

## Project Structure

```
src/                  # Core RAG pipeline
eval/                 # Evaluation harness and experiment
scripts/              # build_index, run_question, inspect_data, preview_chunks
docker/               # Container entrypoint
data/documents/       # Valdoria Markdown corpus (9 files)
data/historical_results.csv
data/golden_seed.jsonl
data/quick_test_questions.md
data/README_data.md
storage/faiss_index/  # Generated FAISS index + metadata (gitignored)
tests/                # Unit/integration tests
Dockerfile
docker-compose.yml
README.md
REPORT.md
TEST_PROJECT_REPORT.md
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
