import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas import PatientInput, PredictionResponse, DiseasePrediction
from app.model_loader import get_artifacts, MODEL_VERSION
from app.preprocessing import encode_symptoms

router = APIRouter()

TOP_K = 3


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientInput):
    try:
        artifacts = get_artifacts()
    except FileNotFoundError as e:
        # Model not trained/dropped in yet — fail loudly and helpfully
        # rather than returning a fake prediction.
        raise HTTPException(status_code=503, detail=str(e))

    vector, unmatched = encode_symptoms(payload.symptoms, artifacts.vocab)

    if not artifacts.model:
        raise HTTPException(status_code=503, detail="Model failed to load.")

    probabilities = artifacts.model.predict_proba(vector)[0]
    top_indices = np.argsort(probabilities)[::-1][:TOP_K]

    top_candidates = [
        DiseasePrediction(disease=artifacts.label_classes[i], probability=float(probabilities[i]))
        for i in top_indices
    ]
    best = top_candidates[0]

    return PredictionResponse(
        predicted_disease=best.disease,
        confidence=best.probability,
        top_candidates=top_candidates,
        description=artifacts.descriptions.get(best.disease),
        precautions=artifacts.precautions.get(best.disease),
        model_version=MODEL_VERSION,
    )
