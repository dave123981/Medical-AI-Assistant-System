# Service 1 — Disease Diagnosis Assistant

FastAPI microservice that predicts a likely disease from patient
symptoms. Called by the Go API Gateway; not exposed directly to the
frontend.


## Endpoints

- `GET /health` — liveness check
- `POST /predict` — see `app/schemas.py` for the request/response shape

## Environment variables

| Variable         | Default                | Purpose |
|-------------------|------------------------|---------|
| `MODEL_DIR`       | `./models`             | Where artifacts are read from |
| `MODEL_FILENAME`  | `v1_decision_tree.joblib` | Swap this to move to v2/v3/v4 |
| `MODEL_VERSION`   | `v1-decision-tree`      | Echoed back in every response |

## Why age/vitals/labs/history are accepted but unused in v1

The Kaggle "Disease Symptom Prediction" dataset only contains
Disease + Symptom columns — there's no age, vitals, or lab data to
train on. The API still accepts these fields (per the system-wide
contract) so the gateway and frontend don't need to change when a
future version is trained on richer data; `preprocessing.py` documents
exactly what's used today.
