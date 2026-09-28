from fastapi import APIRouter, HTTPException

from app.schemas import AskRequest, AskResponse
from app.model_loader import get_artifacts, DataLoadError, CONFIDENCE_THRESHOLD

router = APIRouter()

DISCLAIMER = (
    "Educational information only — not a diagnosis or medical advice. "
    "Always consult a qualified healthcare provider."
)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    try:
        artifacts = get_artifacts()
    except DataLoadError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not payload.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")

    entry, score, generated_answer = artifacts.retrieve(payload.question, payload.context_disease)

    return AskResponse(
        answer=generated_answer if generated_answer is not None else entry["answer"],
        generated=generated_answer is not None,
        source_answer=entry["answer"],
        matched_question=entry["question"],
        similarity_score=score,
        confident=score >= CONFIDENCE_THRESHOLD,
        sources=entry.get("sources", ["MedQuAD"]),
        disclaimer=DISCLAIMER,
        model_version=artifacts.version,
    )