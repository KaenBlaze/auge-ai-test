# Design Tradeoffs

Key architectural decisions and rationale for the AUGE Intelligence local RAG system.

---

## Why RAG Instead of Fine-Tuning?

| | RAG | Fine-tuning |
|---|-----|-------------|
| **Knowledge updates** | Re-index documents in minutes | Requires retraining |
| **Citations** | Natural — answers point to source chunks | No built-in provenance |
| **Hallucination control** | Retrieval bounds the context window | Model may invent facts from weights |
| **Cost** | Inference + small index | GPU training cost, data labeling |
| **This project** | ✅ Fits take-home scope and auditability | ❌ Overkill for 2-document demo corpus |

**Decision:** RAG because the goal is **verifiable, citeable answers** over a changing document corpus — not teaching the model new parametric knowledge.

---

## Why Local Embeddings?

| Decision | Choice | Rationale | Tradeoff |
|----------|--------|-----------|----------|
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers | Strong quality/size ratio; runs on CPU; no API keys | Slower than API embeddings at scale |
| Fallback | `all-MiniLM-L6-v2` | Reliable offline fallback if BGE unavailable | Lower retrieval quality |
| Query prefix | BGE instruction prefix | Improves asymmetric retrieval (query vs passage) | Model-specific; must match embedder |

**Why not OpenAI/Cohere embeddings?** Project requirement: no paid external APIs. Local embeddings also keep the full pipeline air-gappable for on-prem / regulated environments.

---

## Why This Vector Store?

| Decision | Choice | Rationale | Tradeoff |
|----------|--------|-----------|----------|
| Vector store | FAISS `IndexFlatIP` + JSON metadata sidecar | Zero external services; exact cosine search; simple persistence | `IndexFlat` is O(n) — fine for demo, not for millions of vectors |
| Metadata | Separate `metadata.json` | Human-readable; easy to inspect/debug chunks | Not atomic with index — rebuild on mismatch |
| Alternative considered | ChromaDB | Easier CRUD | Heavier dependency; replaced with FAISS per project spec |

**Persistence:** `storage/faiss_index/index.faiss` + `metadata.json` (gitignored).

---

## Why This Model?

| Decision | Choice | Rationale | Tradeoff |
|----------|--------|-----------|----------|
| LLM default | `qwen2.5-7b` via Ollama | Strong instruction-following; easy local pull; no API cost | Requires Ollama install; 7B needs ~8 GB RAM |
| Backends | Ollama / vLLM / Transformers | Flexibility for different deployment targets | Three code paths to maintain |
| Temperature | 0.1 | Reduces creative hallucination | Less fluent paraphrase |

**Why not GPT-4/Claude?** Explicit requirement: local open-weight models only. Also enables reproducible eval without API spend.

**Reranker:** `BAAI/bge-reranker-base` — same model family as embedder; cross-encoder precision without cloud APIs.

---

## What Should NOT Be Handled by the LLM

The LLM is responsible **only** for synthesizing an answer from provided context fragments. It must **not**:

| Task | Handled by |
|------|------------|
| Deciding whether evidence exists | `confidence.py` + `assess_retrieval_evidence()` |
| Retrieving relevant documents | FAISS + embeddings |
| Ranking candidate chunks | Cross-encoder reranker |
| Storing corpus knowledge | Indexed chunks in FAISS |
| Validating citation correctness | Deterministic eval metrics + confidence signals |
| Knowing when to abstain (final gate) | Weighted confidence threshold |

The prompt instructs the model to cite sources and say *"I don't know based on the provided corpus"* when evidence is insufficient — but the **system overrides** weak answers regardless of model confidence.

---

## How the System Avoids Confident Wrong Answers

1. **Retrieval gate** — Abstain before generation if no chunk passes similarity and score thresholds.
2. **Evidence-only prompt** — System prompt forbids outside knowledge; requires `[source: <document>]` citations.
3. **Citation requirement** — `REQUIRE_CITATIONS=true` abstains if the answer has no parseable citations.
4. **Citation support check** — Answer tokens must overlap cited fragment text.
5. **Fragment agreement** — Low lexical agreement across retrieved chunks triggers abstention.
6. **Weighted confidence** — Final score must exceed `confidence_threshold` (0.55).
7. **Abstention phrase detection** — Model uncertainty triggers abstention even if other signals are strong.
8. **Answer replacement** — On abstention, the system replaces the LLM output with the canonical abstention string.

Confidence is derived from **retrieval and citation signals**, not from the LLM's self-reported certainty.

---

## What Remains Production Risk

| Risk | Severity | Mitigation in v2 |
|------|----------|------------------|
| LLM ignores citation format | High | Structured output / JSON schema; post-process validation |
| False abstention on valid answers | Medium | Calibrate thresholds; relax `REQUIRE_CITATIONS` in prod |
| Dense retrieval misses keywords | Medium | Hybrid BM25 + dense |
| Small eval set masks regressions | High | Expand golden set; CI eval gate |
| FAISS flat index at scale | Medium | IVF/HNSW; shard by collection |
| No auth on `/ask` API | High | Add API keys / network policy |
| CPU latency under load | Medium | GPU inference; request queue |
| Corpus drift without re-index | Medium | Scheduled index rebuilds; versioning |
| Cross-encoder latency | Low | Cache rerank results; smaller reranker |
| No audit log | Medium | Log retrieval chunks + scores per query |

---

## Summary Table

| Layer | Choice | Why |
|-------|--------|-----|
| Architecture | RAG | Citations, updatable knowledge, abstention |
| Embeddings | BGE-small (local) | Quality + no API cost |
| Vector store | FAISS + JSON | Simple, local, inspectable |
| LLM | Qwen2.5-7B (Ollama) | Open-weight, instruction-tuned |
| Reranker | BGE reranker | Precision boost without cloud |
| Chunking | Heading/paragraph + tiktoken | Structure-preserving, localized citations |
| Confidence | Multi-signal heuristics | Fast, interpretable, no extra model calls |
| Eval | Deterministic golden-set metrics | Reproducible before/after comparison |
