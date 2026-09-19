# Service 2 — Medical Image Analysis

FastAPI microservice for multi-label disease classification from medical
images. Called by the Go API Gateway.

## Run locally

```bash
cd services/medical-imaging
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

Docs at http://localhost:8001/docs

## Why this service is structured differently from Service 1

- **Multi-label, not multiclass.** A chest X-ray can show multiple
  conditions at once (or none). The model outputs one independent
  probability per condition (sigmoid), not a single softmax pick.
  `/analyze` returns all conditions with their probabilities plus a
  `positive_findings` list of whichever cleared the `threshold`.
- **One model slot per image type.** `models/chest_xray/`,
  `models/skin_lesion/`, `models/retinal/` are independent — each has
  its own model file and `condition_names.json`. Only `chest_xray/` is
  populated so far; the other two return `501` until built.
- **Always Keras.** Unlike Service 1, there's no sklearn-tree option
  for raw pixels, so `model_loader.py` here only supports `.keras`/`.h5`
  — no lazy-import branching needed.

## Before it will actually predict anything

Drop into `models/chest_xray/`:
- the trained `.keras` model
- `condition_names.json` — the ordered list of disease labels the model
  outputs, index-aligned to its sigmoid output

Until then, `/analyze` with `image_type=chest_xray` returns `503` (model
files missing/mismatched) or `501` (folder doesn't exist yet).

## Endpoints

- `GET /health`
- `GET /conditions?image_type=chest_xray` — the condition list the
  active model for that image type predicts
- `POST /analyze` — multipart form: `image` (file), `image_type`
  (default `chest_xray`), `threshold` (default `0.5`)

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `IMAGING_MODEL_DIR` | `./models` | Where model subfolders are read from |
| `CHEST_XRAY_MODEL_FILENAME` | auto-detected | Force a specific file in `models/chest_xray/` |
| `CHEST_XRAY_MODEL_VERSION` | derived from filename | Override the reported version string |

(Same pattern applies per image type, e.g. `SKIN_LESION_MODEL_FILENAME`
once that folder is populated.)
