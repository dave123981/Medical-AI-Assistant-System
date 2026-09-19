from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.schemas import ImageAnalysisResponse, ConditionProbability, ConditionsResponse
from app.model_loader import (
    get_artifacts,
    UnsupportedImageTypeError,
    ModelDiscoveryError,
    ArtifactMismatchError,
    ModelLoadError,
    SUPPORTED_IMAGE_TYPES,
)
from app.preprocessing import preprocess_image, InvalidImageError

router = APIRouter()

DEFAULT_THRESHOLD = 0.5

LOAD_ERRORS = (UnsupportedImageTypeError, ModelDiscoveryError, ArtifactMismatchError, ModelLoadError)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/conditions", response_model=ConditionsResponse)
def get_conditions(image_type: str = "chest_xray"):
    """
    Returns the ordered condition list the currently-loaded model for
    this image type predicts — mirrors Service 1's /symptoms endpoint,
    so a frontend can show users what the model actually looks for
    without hardcoding it.
    """
    try:
        artifacts = get_artifacts(image_type)
    except LOAD_ERRORS as e:
        raise HTTPException(status_code=503, detail=str(e))

    return ConditionsResponse(
        image_type=image_type,
        conditions=artifacts.condition_names,
        count=len(artifacts.condition_names),
    )


@router.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze(
    image: UploadFile = File(...),
    image_type: str = Form("chest_xray"),
    threshold: float = Form(DEFAULT_THRESHOLD),
):
    if image_type not in SUPPORTED_IMAGE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"image_type must be one of: {', '.join(sorted(SUPPORTED_IMAGE_TYPES))}",
        )

    try:
        artifacts = get_artifacts(image_type)
    except UnsupportedImageTypeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except ModelDiscoveryError as e:
        # Distinguish "not built yet" from a genuine outage: skin_lesion and
        # retinal folders simply not existing yet is an expected, documented
        # state, not a server error — 501 communicates that more accurately
        # than 503 does.
        raise HTTPException(status_code=501, detail=str(e))
    except (ArtifactMismatchError, ModelLoadError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    file_bytes = await image.read()
    try:
        batch = preprocess_image(file_bytes, artifacts.input_size)
    except InvalidImageError as e:
        raise HTTPException(status_code=422, detail=str(e))

    probabilities = artifacts.predict_proba(batch)

    findings = [
        ConditionProbability(
            condition=condition,
            probability=float(prob),
            positive=prob >= threshold,
        )
        for condition, prob in zip(artifacts.condition_names, probabilities)
    ]
    positive_findings = [f.condition for f in findings if f.positive]

    return ImageAnalysisResponse(
        image_type=image_type,
        findings=findings,
        positive_findings=positive_findings,
        threshold=threshold,
        heatmap_base64=None,  # populated starting v4
        model_version=artifacts.version,
    )
