import numpy as np
from fastapi import APIRouter, HTTPException

from app.schemas import PatientInput, PredictionResponse, DiseasePrediction, SymptomsResponse
from app.model_loader import get_artifacts
from app.preprocessing import encode_symptoms

router = APIRouter()

TOP_K = 3


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/symptoms", response_model=SymptomsResponse)
def get_symptoms():
    """
    Returns the exact ordered symptom vocabulary the currently-loaded
    model was trained on. The frontend uses this to render a checklist
    instead of a free-text field, so symptoms sent to /predict always
    match a real vocabulary entry exactly.
    """
    try:
        artifacts = get_artifacts()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return SymptomsResponse(symptoms=artifacts.vocab, count=len(artifacts.vocab))


@router.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientInput):
    try:
        artifacts = get_artifacts()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    vector, unmatched = encode_symptoms(payload.symptoms, artifacts.vocab)

    probabilities = artifacts.predict_proba(vector)[0]
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
        model_version=artifacts.version,
    )
