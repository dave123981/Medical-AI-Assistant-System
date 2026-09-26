from fastapi import APIRouter, HTTPException

from app.schemas import RecommendationRequest, RecommendationResponse, DrugRecommendation, ConditionsResponse
from app.model_loader import get_artifacts, DataLoadError

router = APIRouter()

DISCLAIMER = (
    "Clinical decision support only — not a prescribing tool. Contraindication "
    "data is illustrative, not exhaustive, and must not be relied on for real "
    "medical decisions."
)

TOP_K = 10


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/conditions", response_model=ConditionsResponse)
def get_conditions():
    """
    Returns the exact condition vocabulary this service has drug data for.
    Mirrors /symptoms (Service 1) and /conditions (Service 2) — the frontend
    should populate a dropdown from this rather than accepting free text,
    since v1 only does exact (case-insensitive) matching.
    """
    try:
        artifacts = get_artifacts()
    except DataLoadError as e:
        raise HTTPException(status_code=503, detail=str(e))

    conditions = sorted(artifacts.condition_display_names.values())
    return ConditionsResponse(conditions=conditions, count=len(conditions))


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(payload: RecommendationRequest):
    try:
        artifacts = get_artifacts()
    except DataLoadError as e:
        raise HTTPException(status_code=503, detail=str(e))

    matched_name, drugs = artifacts.find_condition(payload.condition)

    recommendations = []
    for entry in drugs[:TOP_K]:
        reasons = artifacts.check_contraindications(entry["drug"], payload.allergies, payload.current_medications)
        recommendations.append(
            DrugRecommendation(
                drug=entry["drug"],
                score=entry["score"],
                review_count=entry["review_count"],
                contraindicated=len(reasons) > 0,
                contraindication_reasons=reasons,
            )
        )

    return RecommendationResponse(
        condition=payload.condition,
        condition_matched=matched_name,
        recommendations=recommendations,
        contraindication_rules_loaded=artifacts.rules_loaded,
        disclaimer=DISCLAIMER,
        model_version=artifacts.version,
    )
