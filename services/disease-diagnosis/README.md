# Service 1 — Disease Diagnosis Assistant

FastAPI microservice that predicts a likely disease from patient
symptoms. Called by the Go API Gateway; not exposed directly to the
frontend. **Complete through v4** (Decision Tree → Random Forest →
XGBoost → Neural Network).

## Run locally

```bash
cd services/disease-diagnosis
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Then open http://localhost:8000/docs for interactive API docs.

## Dataset

**`dhivyeshrk/diseases-and-symptoms-dataset`** on Kaggle — 246,945 rows,
773 diseases, 377 symptoms (already one-hot encoded columns).

This replaced an earlier choice (`itachi9604/disease-symptom-description-dataset`)
after discovering that dataset's ~5,000 rows were almost entirely
duplicates of a small set of unique symptom combinations — a model
trained on it would score ~100% accuracy by memorization, not genuine
generalization. Worth knowing if you see that older dataset referenced
elsewhere.

52 diseases with fewer than 5 samples were excluded from training as
statistically unreliable to evaluate — see the training notebook's
output for the full excluded list.

## Model versions

| Version | Model | Test accuracy | Notes |
|---|---|---|---|
| v1 | Decision Tree | 81.6% | Baseline |
| v2 | Random Forest | 85.2% | Best accuracy-for-simplicity tradeoff |
| v3 | XGBoost | 83.6% | **Documented regression** — underperformed v2 on this wide, sparse, binary-feature dataset. A legitimate negative result, not a bug — see the training notebook's discussion. |
| v4 | Neural Network (class-weighted) | 86.4% | Best overall; validated against train/val curves to rule out overfitting |

Only **one** model file should be present in `models/` at a time —
`model_loader.py` auto-detects it and derives the reported
`model_version` from its filename. If more than one is present, the
service returns a `503` naming the ambiguous files rather than
guessing which to serve.

## Before it will actually predict anything

Drop the trained artifact into `models/`:
- the model file (`.joblib` for v1-v3, `.keras` for v4)
- `symptom_vocab.json` — the ordered list of symptom columns the model was trained on
- `label_classes.json` — the ordered list of disease names, index-aligned to the model's output

All three **must** come from the same training run — mixing files from
different versions produces a validated, clear `503` (`ArtifactMismatchError`)
rather than silently misaligning predictions to the wrong disease names.

## Endpoints

- `GET /symptoms` — the exact symptom vocabulary the active model was trained on
- `POST /predict` — see `app/schemas.py` for the request/response shape

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_DIR` | `./models` | Where the model artifact is read from |
| `MODEL_FILENAME` | auto-detected | Force a specific file if more than one is present |
| `MODEL_VERSION` | derived from filename | Override the reported version string |

## Why age/vitals/labs/history are accepted but unused

The dataset only contains symptom presence/absence and disease labels —
no age, vitals, or lab data. The API still accepts these fields (per
the system-wide contract) so the gateway and frontend don't need to
change when a future version is trained on richer data.
