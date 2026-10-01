from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import ask
from app.model_loader import get_artifacts, DataLoadError

app = FastAPI(
    title="Medical Chatbot",
    description="Service 4 of the medical AI system. Educational Q&A only — not diagnosis or prescribing.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask.router)


@app.on_event("startup")
def preload_artifacts():
    print("=" * 70)
    print("Preloading chatbot artifacts (Downloading the required assets)...")
    try:
        artifacts = get_artifacts()
        print(f"Ready. retrieval_mode={artifacts.retrieval_mode}  model_version={artifacts.version}")
        print(f"Re-ranking enabled: {artifacts.cross_encoder is not None}")
        print(f"Generation enabled: {artifacts.generator is not None}")
    except DataLoadError as e:
        print(f"Preload failed (service will still start; every /ask call will retry): {e}")
    print("=" * 70)