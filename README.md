# Medical AI Assistant System

A modular, versioned medical AI system: a Go API Gateway in front of
independently trainable/deployable Python microservices, each covering
one part of a clinical workflow. Built as a portfolio project — models
are trained in Google Colab on public Kaggle datasets, then dropped
into their service's `models/` folder.

```
                                 Web (vanilla JS/HTML, frontend/) WIP
                                  │
                           API Gateway (Go, gateway/) WIP
                                  │
        ┌──────────────┬──────────┼──────────────┐
        │              │          │              │
        ▼              ▼          ▼              ▼
 Disease Diagnosis  Medical    Drug          Medical
    Service          Imaging   Recommendation Chatbot
   (FastAPI, WIP)  Service   Service        Service
                     (TODO)    (TODO)         (TODO)
```

## Repo layout

```
medical-ai-system/
├── gateway/                    # Go API Gateway — routing, validation, CORS not built yet
├── frontend/                   # Static JS/HTML client, calls the gateway not built yet
├── services/
│   ├── disease-diagnosis/      # Service 1 — WIP
│   ├── medical-imaging/        # Service 2 — README with contract, not built yet
│   ├── drug-recommendation/    # Service 3 — README with contract, not built yet
│   └── medical-chatbot/        # Service 4 — README with contract, not built yet
└── docker-compose.yml
```

## Why this structure

The gateway is the **structural benchmark**: its request/response
contracts (`gateway/internal/models/`) define what every service must
accept and return, regardless of which model version is behind it.


## Versioning convention

Each service versions its **model**, not its API contract. The
contract (input/output shape) stays stable across versions so the
gateway and frontend don't change; only `MODEL_FILENAME` /
`MODEL_VERSION` env vars move.

| Service | Disease Diagnosis | Medical Imaging | Drug Recommendation | Medical Chatbot |
|---|---|---|---|---|
| V1 | Decision Tree | — | — | Retrieval baseline |
| V2 | Random Forest | — | — | RAG |
|  V3 | XGBoost | — | — | — |
| V4 | Neural Network |  | — | — |

## Datasets

| Service | Dataset (Kaggle unless noted) |
|---|---|
| Disease Diagnosis | Disease Symptom Prediction Dataset |
| Medical Imaging | ChestX-ray14 / CheXpert |
| Drug Recommendation | UCI ML Drug Review Dataset |
| Medical Chatbot | MedQuAD, PubMedQA |

## Adding a new service (imaging, drugs, or chatbot)

1. Copy the shape of `services/disease-diagnosis/` (FastAPI app,
   `models/` dir, Dockerfile, README).
2. Add a client in `gateway/internal/clients/` and a real handler in
   `gateway/internal/handlers/`, replacing the matching `Stub*`
   function in `stubs.go`.
3. Add its URL to `gateway/internal/config/config.go` (the env var
   already exists — e.g. `IMAGING_SERVICE_URL`).
4. Add it to `docker-compose.yml`.

## Disclaimer

This is an educational/portfolio project. It is not a medical device
and must not be used for real diagnosis, prescribing, or treatment
decisions.
