# Evaluation Report

> **Status**: Template — populate after running the evaluation harness.

## Run Metadata

| Field | Value |
|-------|-------|
| Date | _TBD_ |
| Embedding Model | _from .env_ |
| LLM Model | _from .env_ |
| Corpus Size | _N documents, M chunks_ |
| Golden Examples | _N_ |

## Summary Metrics

| Metric | Score |
|--------|-------|
| Exact Match | _TBD_ |
| Contains Answer | _TBD_ |
| ROUGE-L | _TBD_ |
| Citation Precision | _TBD_ |
| Citation Recall | _TBD_ |
| Abstention Accuracy | _TBD_ |
| Mean Confidence | _TBD_ |

## Per-Example Analysis

<!-- TODO: Fill in after evaluation run -->

### Passing Examples

_List questions answered correctly with proper citations._

### Failing Examples

_List failures with root cause: retrieval miss, generation hallucination, abstention error, etc._

## Observations

<!-- TODO: Document qualitative findings -->

## Reproduction

```bash
python scripts/build_index.py
python eval/evaluate.py
```

Results appended to `data/historical_results.csv`.
