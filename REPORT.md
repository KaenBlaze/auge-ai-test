# Technical Report — AUGE Local RAG System

## Run Metadata

| Field | Value |
|-------|-------|
| Date | 2026-06-30 |
| Embedding Model | `BAAI/bge-small-en-v1.5` (384-dim) |
| Reranker | `BAAI/bge-reranker-base` |
| LLM | `qwen2.5:7b` via Ollama (`MODEL_BACKEND=ollama`) |
| Vector Store | FAISS `IndexFlatIP` at `storage/faiss_index/` |
| Corpus | 9 Valdoria documents + `historical_results.csv` → ~39 chunks |
| Golden Set | 26 examples (`data/golden_set.jsonl`) — 20 answerable, 6 unanswerable |
| Hardware | Linux x86_64, CPU embeddings; Ollama for generation |
| Test Suite | 73 tests passing |

---

## Architecture Overview

The pipeline is a modular, local-only RAG stack:

1. **Ingest** — Markdown laws/policies/reports plus CSV rows indexed as searchable text.
2. **Chunk** — Heading/paragraph-aware merging (640 tokens, 100 overlap).
3. **Embed & index** — BGE-small embeddings in FAISS with JSON metadata sidecar.
4. **Retrieve** — Dense top-k with similarity scores.
5. **Rerank** — Optional cross-encoder re-ordering.
6. **Tabular path** — Deterministic pandas answers for CSV aggregation questions.
7. **Generate** — Local LLM with evidence-only prompt and inline citations.
8. **Score & abstain** — Multi-signal confidence gates final output.

Design goal: **verifiable answers** with honest abstention when evidence is missing.

---

## Chunking Strategy and Justification

**Approach:** Heading-aware, paragraph-aware merging with tiktoken sizing.

- Records grouped by `(document, section_heading)` — chunks do not cross sections.
- Adjacent paragraphs merged up to ~640 tokens (500–800 target range).
- Oversized paragraphs split on sentence boundaries.
- Each chunk prefixed with `## {section_heading}` for self-contained context.
- 100-token overlap preserves continuity.

**Why:** Preserves legal section boundaries for localized citations. Token-aware sizing aligns with embedding and LLM context limits.

**Tradeoff:** Tables in Markdown may split awkwardly; v2 should add table-aware chunking.

---

## Retrieval Strategy

**Approach:** Dense retrieval (bi-encoder → FAISS cosine search).

- Default `top_k_retrieve=10`, `similarity_threshold=0.3`.
- BGE query instruction prefix at embed time.
- Retrieved chunks include full traceability metadata.

**Measured:** `retrieval_accuracy` on golden set — see latest `eval/results.json`.

**Tradeoff:** No BM25 hybrid yet; exact keyword matches can be missed.

---

## Reranking Strategy

**Approach:** Local cross-encoder (`BAAI/bge-reranker-base`).

- Re-orders candidates before generation and confidence scoring.
- Before/after experiment in `eval/experiment.py`.

**Tradeoff:** Adds CPU latency; benefit is corpus-dependent.

---

## Tabular Query Handling

**Approach:** `src/tabular_query.py` answers CSV aggregation questions deterministically with pandas.

Supported patterns:
- Lowest/highest metric by region and year (turnout, budgets, trust)
- Region + year lookups (e.g., "North Region turnout in 2025")

**Why:** Numeric answers should not be guessed by the LLM from raw row text. Code computes the value; LLM handles prose synthesis for document-based questions.

---

## Confidence and Abstention Logic

Abstention gates the final answer — it is not decorative.

**Pre-generation checks:**
- No chunks retrieved
- No chunk passes `similarity_threshold`
- Weak top retrieval / rerank scores
- Insufficient supporting chunks

**Post-generation checks:**
- Model expresses uncertainty
- Citation support below threshold (with exceptions for valid inline citations and structured pipeline citations)
- Weighted confidence below `confidence_threshold`

**Weighted signals:** top retrieval (30%), top rerank (25%), supporting chunks (15%), fragment agreement (10%), citation support (20%).

**Fragment agreement:** Soft signal only — diverse sections from the same document are expected and not treated as conflicting evidence.

When abstaining: `"I don't know based on the provided corpus."` with `abstained=true`, reason code, and retrieved citations for audit.

---

## Evaluation Metrics

| Metric | Definition |
|--------|------------|
| `retrieval_accuracy` | Expected sources found in retrieved documents |
| `citation_coverage` | Answerable examples include ≥1 citation |
| `citation_correctness` | Cited sources match expected document IDs |
| `groundedness` | Answer token overlap with cited fragments |
| `hallucination` | Answered when should abstain, or without citations |
| `correct_abstention_rate` | Abstained on unanswerable examples |
| `false_abstention_rate` | Abstained on answerable examples |
| `confidence_calibration` | Binned confidence vs correctness (ECE) |

Run:

```bash
python eval/evaluate.py --golden-set data/golden_set.jsonl
```

---

## Before / After Experiment

<!-- EXPERIMENT_START -->

**Run date:** 2026-07-01T00:23:32.875381+00:00
**Golden set:** `data/golden_set.jsonl` (26 examples)

| Metric | Baseline | Reranked | Delta | % Change |
|--------|----------|----------|-------|----------|
| retrieval_accuracy | 1.0000 | 1.0000 | +0.0000 | +0.0% |
| citation_correctness | 0.7692 | 0.7692 | +0.0000 | +0.0% |
| hallucination | 0.0000 | 0.0000 | +0.0000 | +0.0% |
| correct_abstention_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% |

### What changed

- **Baseline:** dense FAISS retrieval only (`use_reranker=false`).
- **After:** dense retrieval followed by local cross-encoder reranking (`use_reranker=true`).

### Interpretation

Unchanged metrics: retrieval_accuracy, citation_correctness, hallucination, correct_abstention_rate.
Baseline uses dense FAISS retrieval only (top-k by embedding similarity). The reranked condition adds a local cross-encoder that re-orders retrieved chunks before generation and confidence scoring.
On this small golden set, reranking may show little gain when dense retrieval already places the correct document near the top, when the corpus is tiny, or when confidence thresholds and citation requirements dominate downstream abstention. Reranking helps most on ambiguous queries where lexical overlap misleads bi-encoder retrieval.

Reproduce with:

```bash
python eval/experiment.py
```

<!-- EXPERIMENT_END -->



---

## Limitations

1. **Corpus size** — Synthetic demo; not production scale.
2. **Golden set** — 26 examples; extend further for production CI gates.
3. **Calibration** — Confidence heuristics need threshold tuning on held-out data.
4. **No entailment model** — Groundedness uses lexical overlap.
5. **LLM variance** — Answer quality depends on Ollama model version.
6. **Docker** — Requires host Docker daemon; nested dev containers may not run Compose.

---

## What I Would Improve in v2

| Priority | Improvement |
|----------|-------------|
| High | Expand golden set to 50–100 with human review |
| High | Hybrid BM25 + dense retrieval |
| High | NLI-based groundedness check |
| Medium | Structured JSON citation output from LLM |
| Medium | SQL/pandas query router for all tabular questions |
| Medium | GPU batching for index builds |
| Low | Streaming `/ask` responses |
| Low | Incremental index updates |

---

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5:7b

python scripts/build_index.py
python scripts/run_question.py "What is the electoral silence period before election day?"
python eval/evaluate.py --golden-set data/golden_set.jsonl
python eval/experiment.py --golden-set data/golden_set.jsonl
pytest tests/ -q
```
