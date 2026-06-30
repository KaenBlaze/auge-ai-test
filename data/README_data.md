# Republic of Valdoria — Synthetic Assessment Dataset

This dataset is synthetic and created for the AUGE Intelligence AI / ML Engineer technical assessment. It is designed to test RAG, citations, abstention, confidence scoring, and evaluation harness design.

## Contents

```text
documents/
  9 Markdown documents with synthetic laws, policies, reports, and public records
historical_results.csv
  Synthetic annual regional metrics from 2021–2025
golden_seed.jsonl
  6 seed evaluation questions, including one unanswerable case
quick_test_questions.md
  Manual smoke-test prompts
README_data.md
  This file
```

## Document Types

- Electoral law
- Data protection / sovereignty law
- Campaign finance regulation
- Advertising calendar
- Budget transparency guidelines
- Budget execution report
- Turnout and service report
- Procurement and audit manual
- Public AI governance policy

## CSV Schema

`historical_results.csv`

| Column | Meaning |
|---|---|
| year | Reporting year |
| region | Synthetic Valdorian region |
| turnout_pct | Aggregate voter turnout percentage |
| education_budget_m | Education budget in synthetic millions |
| health_budget_m | Health budget in synthetic millions |
| campaign_spend_m | Campaign spend in synthetic millions |
| institutional_trust_pct | Synthetic civic survey trust score |

## Golden Seed Format

`golden_seed.jsonl` uses Spanish field names to match the candidate assessment style:

```json
{
  "id": "q001",
  "pregunta": "Question text",
  "respuesta_referencia": "Reference answer",
  "fuentes": ["source references"],
  "tipo": "factual | tabular | unanswerable",
  "respondible": true
}
```

`respondible=false` means the system should abstain because the corpus does not contain enough evidence.

## Important Notes

- Do not use external paid generation APIs.
- Treat this synthetic data as sensitive.
- Citations should refer to document IDs and sections/articles.
- Correct abstention is preferred over unsupported confidence.
