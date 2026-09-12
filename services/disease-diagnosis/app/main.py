from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.routers import predict
from app import model_loader

app = FastAPI(
    title="Disease Diagnosis Assistant",
    description="Service 1 of the medical AI system. Predicts a likely disease from patient symptoms.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)


@app.on_event("startup")
def log_active_model():
    """
    Prints exactly which model file the service will try to serve, and
    whether it actually exists, the moment the server starts — so a
    discovery failure (none found, or ambiguous multiple) shows up
    immediately in the terminal instead of only surfacing as a 503 on
    the first real request.
    """
    print("=" * 70)
    try:
        model_filename = model_loader.resolve_model_filename()
        model_version = model_loader.resolve_model_version(model_filename)
        model_path = model_loader.MODEL_DIR / model_filename
        print(f"MODEL_DIR      : {model_loader.MODEL_DIR}")
        print(f"Model file     : {model_filename}  (auto-detected: {'no, MODEL_FILENAME set' if os.getenv('MODEL_FILENAME') else 'yes'})")
        print(f"Resolved path  : {model_path}")
        print(f"File exists?   : {'YES' if model_path.exists() else 'NO'}")
        print(f"Model version  : {model_version}")
    except model_loader.ModelDiscoveryError as e:
        print(f"MODEL_DIR      : {model_loader.MODEL_DIR}")
        print("Model file     : COULD NOT BE DETERMINED")
        print(f"Reason         : {e}")
    print("=" * 70)
