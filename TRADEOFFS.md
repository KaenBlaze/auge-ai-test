# Design Tradeoffs

Document key architectural decisions and their rationale for the AUGE Intelligence RAG system.

## Local-Only Stack

| Decision | Choice | Rationale | Tradeoff |
|----------|--------|-----------|----------|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Fast, lightweight, runs on CPU | Lower quality vs larger models (e.g. `bge-large`) |
| Vector Store | ChromaDB (persistent) | Zero-config local persistence, cosine search | Less scalable than dedicated vector DBs (Pinecone, Weaviate) |
| LLM | Ollama | Easy local model management, no API keys | Requires separate Ollama install; model quality varies |
| Reranker | Cross-encoder (`ms-marco-MiniLM-L-6-v2`) | Significant retrieval quality boost | Adds latency (~50–200ms per query) |

## Chunking Strategy

**Current**: Character-based sliding window with overlap.

**Why**: Simple, deterministic, no tokenizer dependency at setup time.

**Tradeoff**: May split mid-sentence or mid-table; token-aware chunking (tiktoken) would align better with LLM context windows.

## Confidence & Abstention

**Current**: Heuristic score from retrieval similarity, context coverage, and abstention phrase detection.

**Why**: No additional model calls; fast and interpretable.

**Tradeoff**: Not calibrated to actual answer correctness. A golden-set calibration pass is planned (see `confidence.py` TODOs).

## Retrieval

**Current**: Dense retrieval only (bi-encoder embeddings).

**Why**: Simpler pipeline; works well for semantic queries.

**Tradeoff**: Misses exact keyword matches (SKUs, error codes). Hybrid BM25 + dense retrieval is a planned improvement.

## Evaluation

**Current**: Golden seed JSONL with exact match, substring, ROUGE-L, citation, and abstention metrics.

**Why**: Covers the core requirements (correct answers, citations, abstention).

**Tradeoff**: ROUGE-L is a weak proxy for factual correctness; LLM-as-judge or human eval would be more reliable for production.

## API Design

**Current**: Synchronous `/ask` endpoint returning full response.

**Why**: Simplest interface for a take-home demo.

**Tradeoff**: No streaming; long generations block the request. Streaming support is planned.

---

<!-- TODO: Update this document as implementation choices evolve across steps 2–11. -->
