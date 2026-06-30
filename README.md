# AUGE Intelligence — Local RAG System

A take-home AI/ML Engineer test project: a **local Retrieval-Augmented Generation (RAG)** system that answers questions over a synthetic corpus with **citations**, **confidence scores**, and **honest abstention** when evidence is insufficient.

## Features

- **Fully local** — no paid external APIs; uses sentence-transformers, ChromaDB, and Ollama
- **Citations** — every answer links back to source chunks
- **Confidence & abstention** — retrieval-based confidence scoring with configurable threshold
- **Evaluation harness** — metrics against a golden seed dataset with historical tracking
- **API & CLI** — FastAPI HTTP service and command-line tools

## Project Structure

```
src/           # Core RAG pipeline modules
eval/          # Evaluation harness and metrics
scripts/       # Convenience scripts (build index, ask question)
data/          # Corpus, golden seed, evaluation history
tests/         # Unit tests
```

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) running locally with a model pulled (e.g. `ollama pull llama3.2`)

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env

# 4. Build the vector index
python scripts/build_index.py

# 5. Ask a question
python scripts/run_question.py "What sensors does the AUGE platform support?"

# 6. Or use the CLI
python -m src.cli ask "What is the primary purpose of the AUGE monitoring system?"

# 7. Start the API server
python -m src.api
# POST http://localhost:8000/ask  {"question": "..."}

# 8. Run evaluation
python eval/evaluate.py
```

## Configuration

All settings are loaded from `.env` (see `.env.example`). Key options:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local embedding model |
| `OLLAMA_MODEL` | `llama3.2` | Local LLM via Ollama |
| `TOP_K_RETRIEVE` | `10` | Initial retrieval count |
| `TOP_K_RERANK` | `5` | Chunks after reranking |
| `CONFIDENCE_THRESHOLD` | `0.5` | Abstention threshold |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `exact_match` | Normalized exact answer match |
| `contains_answer` | Expected answer substring present |
| `rouge_l` | ROUGE-L F1 overlap |
| `citation_precision` | Fraction of correct citations |
| `citation_recall` | Fraction of expected citations found |
| `abstention_accuracy` | Correct abstain/answer decisions |

Results are appended to `data/historical_results.csv` for regression tracking.

## TODO (Roadmap)

See inline `TODO` comments throughout the codebase. High-priority items:

- [ ] Token-aware chunking with tiktoken
- [ ] Hybrid retrieval (BM25 + dense)
- [ ] Calibrate confidence thresholds on golden set
- [ ] Streaming API responses
- [ ] llama.cpp backend alternative to Ollama

## License

Internal use — AUGE Intelligence take-home assignment.
