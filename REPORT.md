# Technical Report — AUGE Local RAG System

## Run Metadata

| Field | Value |
|-------|-------|
| Date | 2026-06-30 |
| Embedding Model | `BAAI/bge-small-en-v1.5` (384-dim) |
| Reranker | `BAAI/bge-reranker-base` |
| LLM | `qwen2.5-7b` via Ollama (`MODEL_BACKEND=ollama`) |
| Vector Store | FAISS `IndexFlatIP` at `storage/faiss_index/` |
| Corpus | 2 Markdown documents → 10 paragraph records → 9 chunks |
| Golden Examples | 3 (2 answerable, 1 unanswerable) |
| Hardware | Linux x86_64, CPU-only inference |
| Test Suite | 67 tests passing |

---

## Architecture Overview

The pipeline is a modular, local-only RAG stack:

1. **Ingest** — Markdown files parsed into paragraph-level records with section headings, line numbers, and `source_id`.
2. **Chunk** — Records merged within sections up to a tiktoken budget (640 tokens, 100 overlap), preserving headings in chunk text.
3. **Embed & index** — BGE-small embeddings stored in a FAISS inner-product index (cosine on L2-normalized vectors) with a JSON metadata sidecar.
4. **Retrieve** — Query embedded with BGE query prefix; top-k chunks returned with similarity scores.
5. **Rerank** — Optional cross-encoder re-orders candidates before generation.
6. **Generate** — Local LLM answers strictly from retrieved fragments with inline citations.
7. **Score & abstain** — Multi-signal confidence decides whether to return the answer or `"I don't know based on the provided corpus."`

Design goal: **verifiable answers** — every non-abstained response should be traceable to corpus fragments.

---

## Chunking Strategy and Justification

**Approach:** Heading-aware, paragraph-aware merging with tiktoken sizing.

- Records are grouped by `(document, section_heading)` so chunks never cross section boundaries.
- Adjacent paragraphs within a section are merged until ~640 tokens (middle of 500–800 target).
- Oversized paragraphs split on sentence boundaries, then token windows.
- Each chunk is prefixed with `## {section_heading}` for self-contained retrieval context.
- 100-token overlap carries context across chunk boundaries.

**Why:** Blind fixed-character splitting breaks Markdown structure and separates headings from body text. Token-aware sizing aligns chunk boundaries with embedding model and LLM context limits while keeping citations localized.

**Tradeoff:** List-heavy or table-heavy documents may still split awkwardly; v2 should add table-aware chunking.

---

## Retrieval Strategy

**Approach:** Dense retrieval only (bi-encoder embeddings → FAISS cosine search).

- Default `top_k_retrieve=10`, filtered by `similarity_threshold=0.3`.
- BGE query instruction prefix applied at embed time.
- Retrieved chunks include `chunk_id`, `document`, `source_id`, `fragment`, `retrieval_score`.

**Why:** Simple, fast to implement, strong on semantic paraphrase for a small technical corpus.

**Tradeoff:** Misses exact keyword matches (error codes, SKUs). No BM25 hybrid yet.

---

## Reranking Strategy

**Approach:** Optional local cross-encoder (`BAAI/bge-reranker-base`, fallback `ms-marco-MiniLM-L-6-v2`).

- Reranks top-k retrieved chunks before generation and confidence scoring.
- Scores normalized to [0, 1] within the batch.
- `SimilarityFallbackReranker` used when cross-encoder unavailable.

**Why:** Cross-encoders score query–passage pairs jointly, improving precision when bi-encoder retrieval returns plausible but wrong chunks.

**Tradeoff:** Adds latency (~50–200 ms per query on CPU). Benefit is query-dependent.

---

## Confidence and Abstention Logic

Abstention is **not decorative** — it gates the final answer.

**Pre-generation checks** (retrieval evidence only):
- No chunks retrieved
- No chunk passes `similarity_threshold`
- `top_retrieval` < `min_retrieval_score` (0.35)
- `top_rerank` < `min_rerank_score` (0.40)
- Insufficient supporting chunks
- Low fragment agreement

**Post-generation checks** (answer + citations):
- Model expresses uncertainty / abstention phrase
- `REQUIRE_CITATIONS=true` but answer has no `[source: ...]` tags
- Citation support below `min_citation_support` (0.25)
- Weighted confidence below `confidence_threshold` (0.55)

**Weighted signals:** top retrieval (25%), top rerank (25%), supporting chunks (15%), fragment agreement (15%), citation support (20%).

When abstaining, the system returns: `"I don't know based on the provided corpus."` with `abstained=true` and a machine-readable `reason`.

---

## Evaluation Metrics

Deterministic metrics implemented in `eval/metrics.py`:

| Metric | Definition |
|--------|------------|
| `retrieval_accuracy` | Expected sources found in retrieved documents |
| `citation_coverage` | Answerable examples include ≥1 citation |
| `citation_correctness` | Cited `document`/`source_id` matches expected source |
| `groundedness` | Answer token overlap with cited fragments |
| `hallucination` | Answered when unanswerable, or answerable without citations |
| `correct_abstention_rate` | Abstained on unanswerable examples |
| `false_abstention_rate` | Abstained on answerable examples |
| `confidence_calibration` | Binned confidence vs empirical correctness (ECE) |

Run: `python eval/evaluate.py --golden-set data/golden_seed.jsonl`

---

## Before / After Experiment

<!-- EXPERIMENT_START -->

**Run date:** 2026-06-30T16:25:47.663584+00:00
**Golden set:** `data/golden_seed.jsonl` (3 examples)

| Metric | Baseline | Reranked | Delta | % Change |
|--------|----------|----------|-------|----------|
| retrieval_accuracy | 1.0000 | 1.0000 | +0.0000 | +0.0% |
| citation_correctness | 0.3333 | 0.3333 | +0.0000 | +0.0% |
| hallucination | 0.0000 | 0.0000 | +0.0000 | +0.0% |
| correct_abstention_rate | 1.0000 | 1.0000 | +0.0000 | +0.0% |

### What changed

- **Baseline:** dense FAISS retrieval only (`use_reranker=false`).
- **After:** dense retrieval followed by local cross-encoder reranking (`use_reranker=true`).

### Interpretation

**Reranking did not improve measurable metrics on this run.** All four tracked metrics were unchanged.

Plausible explanations:
1. **Corpus is tiny** — bi-encoder retrieval already ranks the correct document first.
2. **Bottleneck is generation/citations** — citation correctness (33%) is limited by LLM citation formatting, not chunk order.
3. **Confidence gates dominate** — both conditions hit the same abstention decisions (`false_abstention_rate=100%` on answerable examples due to strict citation requirements).
4. **Golden set too small** — 3 examples cannot show statistically meaningful reranker gains.

Reranking remains valuable for larger, noisier corpora where dense retrieval returns semantically similar but factually wrong passages.

Reproduce with:

```bash
python eval/experiment.py
```

<!-- EXPERIMENT_END -->

---

## Limitations

1. **Scale** — Demo corpus only; FAISS flat index does not scale to millions of chunks without IVF/HNSW.
2. **Eval set size** — 3 golden examples; metrics are directional, not production-grade.
3. **False abstention** — Strict citation and confidence thresholds cause the system to abstain on answerable questions when the LLM omits inline `[source: ...]` tags.
4. **No entailment check** — Groundedness uses lexical overlap, not NLI/entailment models.
5. **Single-machine** — No distributed indexing, caching, or async job queue.
6. **LLM variance** — Answer quality depends on locally pulled model weights and Ollama version.

---

## What I Would Improve in v2

| Priority | Improvement |
|----------|-------------|
| High | Expand golden set to 50–100 examples with human labels |
| High | Hybrid retrieval (BM25 + dense) for keyword-heavy queries |
| High | Calibrate confidence thresholds on held-out eval set |
| Medium | NLI-based groundedness check (local cross-encoder entailment) |
| Medium | Structured citation output (JSON schema from LLM) |
| Medium | GPU batching for embedding index builds |
| Medium | FAISS IVF/HNSW for larger corpora |
| Low | Streaming `/ask` responses |
| Low | Incremental index updates (no full rebuild) |
| Low | PDF/HTML ingestion |

---

## Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull qwen2.5-7b

python scripts/build_index.py
python scripts/run_question.py "Which sensors does the AUGE platform support?"
python eval/evaluate.py --golden-set data/golden_seed.jsonl
python eval/experiment.py
pytest tests/ -q
```
