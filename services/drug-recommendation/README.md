# Service 3 — Drug Recommendation Assistant

Clinical decision support tool: suggests possible medications for a
given condition and flags basic contraindications. **Not a prescribing
system.** Complete through v1 — the only service in this project
without a v2+ progression so far.

## Run locally

```bash
cd services/drug-recommendation
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8002
```

Docs at http://localhost:8002/docs

## Dataset

**`jessicali9530/kuc-hackathon-winter-2018`** on Kaggle (UCI ML Drug
Review Dataset) — ~215,000 patient drug reviews (`drugName`, `condition`,
`review`, `rating`, `date`, `usefulCount`). v1's ranking is a weighted
aggregation formula (rating × recency × usefulness), **not a trained
model** — consistent with this being the simplest, most defensible
baseline before investing in an actual learned ranker.

A ~1,171-row scraping artifact (the literal condition value
`"N</span> users found this comment helpful."`, in 80 numeric variants)
was found and filtered during data cleaning — worth knowing if you
re-run the notebook and see a spike in short, garbled "condition" values.

## Why this service is structured differently from Services 1 & 2

- **No trained ML model in v1.** Rankings are computed once in the
  notebook, not retrained here. A real learned ranker is a natural v2+
  direction, not yet built.
- **Condition matching is exact (case-insensitive) only.** Service 1's
  predicted diseases and this dataset's conditions use different naming
  conventions (clinical vs. consumer-search style) with no guaranteed
  overlap — bridging them was deliberately deferred, not attempted.
  `GET /conditions` exists specifically so a caller always picks from
  real vocabulary rather than guessing.
- **Contraindication rules are hand-curated, not derived from data.**
  The review dataset has no chemical, allergy, or interaction
  information at all. `data/contraindication_rules.json` is a small,
  explicitly illustrative, **not exhaustive** reference table — a few
  well-known, publicly-documented pharmacology facts (e.g. penicillin-class
  cross-allergies), not a licensed clinical resource.

## Before it will actually recommend anything

Drop into `data/`:
- `drug_rankings.json` — `{condition: [{drug, score, review_count}, ...]}`, produced by the notebook
- `contraindication_rules.json` — optional; if absent, **no drug is ever flagged as contraindicated** (fails open, not closed — the API response's `contraindication_rules_loaded: false` flag exists specifically to make this visible rather than silent)

## Endpoints

- `GET /health`
- `GET /conditions` — the exact condition vocabulary this service has drug data for
- `POST /recommend` — `{condition, age?, current_medications?, allergies?}`

### Response shape

```json
{
  "condition": "depression",
  "condition_matched": "Depression",
  "recommendations": [
    {"drug": "sertraline", "score": 8.7, "review_count": 4231, "contraindicated": false, "contraindication_reasons": []}
  ],
  "contraindication_rules_loaded": true,
  "disclaimer": "Clinical decision support only — not a prescribing tool...",
  "model_version": "v1-aggregated-ranking"
}
```

Contraindicated drugs are **included but flagged**, not silently
filtered — the same transparency principle as Service 2 showing all
findings rather than hiding negatives.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DRUG_DATA_DIR` | `./data` | Where the two JSON artifacts are read from |
| `DRUG_MODEL_VERSION` | `v1-aggregated-ranking` | Reported version string |
