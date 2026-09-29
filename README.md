# Medical AI Assistant System

A modular, versioned medical AI system: a Go API Gateway in front of four
independently trainable/deployable Python microservices, each covering
one part of a clinical workflow. Built as a portfolio project — models
are trained in Google Colab on public Kaggle datasets, then dropped
into their service's `models/` or `data/` folder.
```
                                 Web (vanilla JS/HTML, frontend/)
                                  │
                           API Gateway (Go, gateway/)
                                  │
        ┌──────────────┬──────────┼──────────────┐
        │              │          │              │
        ▼              ▼          ▼              ▼
 Disease Diagnosis  Medical    Drug          Medical
    Service          Imaging   Recommendation Chatbot
                     Service   Service        Service
```

## Repo layout

```
medical-ai-system/
├── gateway/ # Go API Gateway — routing, validation, CORS
├── frontend/ # Static JS/HTML client — 4 pages, shared nav
│ ├── index.html # Disease Diagnosis
│ ├── imaging.html # Image Analysis (incl. Grad-CAM display)
│ ├── drugs.html # Drug Recommendation
│ └── chatbot.html # Medical Chatbot (chat log UI)
├── services/
│ ├── disease-diagnosis/ # Service 1 — complete, v1-v4
│ ├── medical-imaging/ # Service 2 — complete, v1-v4, 2 image types
│ ├── drug-recommendation/ # Service 3 — complete, v1
│ └── medical-chatbot/ # Service 4 — complete, v1-v4
```


## Why this structure

The gateway is the **structural benchmark**: its request/response
contracts define what every service must accept and return, regardless
of which model version is behind it. Every service's model-loading code
auto-detects which trained artifact is present and reports its own
version string — swapping a Decision Tree for a Neural Network, or a
chest X-ray model for a skin lesion model, never requires touching the
routing layer or the frontend.

## Running everything locally

Each service needs its own terminal (or use `docker-compose up --build`
once each service's Dockerfile is current):

```bash
# Port forwarding:
cd frontend/ && python3 -m http.server 8081

# Gateway
cd gateway/ && go run main.go                                    

# Service 1
cd services/disease-diagnosis && uvicorn app.main:app --port 8000 --reload

# Service 2
cd services/medical-imaging && uvicorn app.main:app --port 8001 --reload

# Service 3
cd services/drug-recommendation && uvicorn app.main:app --port 8002 --reload

# Service 4
cd services/medical-chatbot && uvicorn app.main:app --port 8003 --reload
```

Then open `frontend/index.html` (served via any static file server —
not `file://` directly, to avoid CORS issues).

Each Python service returns a clean `503` with an actionable message if
its model/data artifacts aren't in place yet — check that service's own
README for exactly which files it expects.

## Versioning convention — what actually shipped

Every service versions its **model or technique**, not its API
contract. The contract stays stable across versions; only the
underlying model file (or, for Service 4, which retrieval technique is
active) changes.

| Service | v1 | v2 | v3 | v4 |
|---|---|---|---|---|
| **Disease Diagnosis** | Decision Tree (81.6% acc) | Random Forest (85.2% acc) | XGBoost (83.6% acc — documented regression) | Neural Network (86.4% acc, best) |
| **Imaging — Chest X-ray** | Baseline CNN (0.608 mean AUC) | DenseNet121 transfer learning (0.734 mean AUC) | Weighted loss (documented null result — no improvement) | Grad-CAM + per-class F1-optimal thresholds |
| **Imaging — Skin Lesion** | Baseline CNN (75.0% acc, melanoma recall 0.35) | EfficientNetB0 transfer learning (macro-F1 0.65, melanoma recall 0.42) | Class-weighted training (melanoma recall 0.58, traded accuracy for minority-class recall) | Grad-CAM (lesion-focused, no shortcut-learning found) |
| **Drug Recommendation** | Weighted aggregation ranking (not a trained model) + hand-curated contraindication rules | — | — | — |
| **Medical Chatbot** | TF-IDF retrieval | Semantic embeddings (fixed v1's false-positive matching) | Cross-encoder re-ranking (opt-in) | Local RAG generation via FLAN-T5 (opt-in) |


## Real findings worth knowing before trusting any of this

This project treats "the model works" as a claim to verify, not assume
— several things were caught along the way that are worth knowing if
you're evaluating the code rather than just running it:

- **Disease diagnosis**: the original dataset (itachi9604) turned out to
  be entirely duplicate rows padded to a round count — switched to a
  larger, more diverse dataset (dhivyeshrk) mid-project. See that
  service's README.
- **Chest X-ray v4**: Grad-CAM revealed the "Infiltration" condition's
  predictions were driven by image-marker artifacts 74% of the time,
  not actual lung tissue — a real, documented shortcut-learning finding,
  not a hypothetical concern.
- **Chest X-ray thresholds**: a flat 0.5 cutoff was unusable (the model
  is systematically under-confident); the first fix attempt (Youden's J
  statistic) was itself wrong under this class imbalance, over-flagging
  one condition by 93x its true rate. F1-optimal thresholds fixed most
  conditions.
- **Chatbot v1**: TF-IDF matched "recommend a good science fiction
  movie" to "What is (are) Good syndrome?" at 0.970 similarity — higher
  than genuine medical question matches. v2's semantic embeddings
  dropped that to 0.217.

Each service's own README has the full detail.

## Datasets

| Service | Dataset (Kaggle unless noted) |
|---|---|
| Disease Diagnosis | `dhivyeshrk/diseases-and-symptoms-dataset` |
| Imaging — Chest X-ray | `nih-chest-xrays/sample` (v1), `khanfashee/nih-chest-x-ray-14-224x224-resized` (v2+) |
| Imaging — Skin Lesion | `kmader/skin-cancer-mnist-ham10000` |
| Drug Recommendation | `jessicali9530/kuc-hackathon-winter-2018` |
| Medical Chatbot | `pythonafroz/medquad-medical-question-answer-for-ai-research` (MedQuAD) |

## Adding a new service or image type

1. Copy the shape of an existing service (FastAPI app, `models/` or
   `data/` dir, Dockerfile, README).
2. Add a client in `gateway/internal/clients/` and a handler in
   `gateway/internal/handlers/`.
3. Add its URL to `gateway/internal/config/config.go`.
4. Wire the route in `gateway/internal/router/router.go`.
5. Add a frontend page, following the shared nav pattern in the
   existing four pages.

## Disclaimer

This is an educational/portfolio project. It is not a medical device
and must not be used for real diagnosis, prescribing, or treatment
decisions.
