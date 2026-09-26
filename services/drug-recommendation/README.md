# Service 3 — Drug Recommendation Assistant

Clinical decision support tool: suggests possible medications for a given
condition and flags basic contraindications. **Not a prescribing system.**

## Run locally

```bash
cd services/drug-recommendation
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8002
```

Docs at http://localhost:8002/docs

## Why this service is structured differently from Services 1 & 2

- **No trained ML model in v1.** Rankings are a weighted aggregation
  (rating × usefulness × recency) computed once in the notebook from the
  UCI Drug Review dataset, not a model retrained here. A real learned
  ranker is a natural v2+ direction.
- **Condition matching is exact (case-insensitive) only.** Service 1's
  predicted diseases and this dataset's conditions use different naming
  conventions (clinical vs. consumer-search style) with no guaranteed
  overlap — bridging them was deliberately deferred. `GET /conditions`
  exists specifically so a caller (frontend or otherwise) always picks
  from real vocabulary rather than guessing.
- **Contraindication rules are hand-curated, not derived from data.**
  The UCI Drug Review dataset is patient reviews only — no chemical,
  allergy, or interaction data at all. `data/contraindication_rules.json`
  is a small, illustrative, NOT exhaustive reference table. Real clinical
  contraindication data is a licensed, maintained resource — this project
  does not attempt to replicate one.

## Before it will actually recommend anything

Drop into `data/`:
- `drug_rankings.json` — `{condition: [{drug, score, review_count}, ...]}`,
  produced by the notebook
- `contraindication_rules.json` — optional; if absent, no drugs are ever
  flagged as contraindicated (fails open, not closed — worth knowing)

## Endpoints

- `GET /health`
- `GET /conditions` — the exact condition vocabulary this service has data for
- `POST /recommend` — `{condition, age?, current_medications?, allergies?}`

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DRUG_DATA_DIR` | `./data` | Where the two JSON artifacts are read from |
| `DRUG_MODEL_VERSION` | `v1-aggregated-ranking` | Reported version string |
