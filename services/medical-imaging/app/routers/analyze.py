from typing import Optional

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
from app.gradcam import compute_gradcam_overlay

router = APIRouter()

LOAD_ERRORS = (UnsupportedImageTypeError, ModelDiscoveryError, ArtifactMismatchError, ModelLoadError)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/conditions", response_model=ConditionsResponse)
def get_conditions(image_type: str = "chest_xray"):
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
    # No default here on purpose: None means "use each condition's own
    # tuned threshold." A caller that WANTS a uniform cutoff (e.g. for a
    # demo, or to be stricter/looser than the tuned defaults) can still
    # pass one explicitly, which overrides every condition uniformly.
    threshold: Optional[float] = Form(None),
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
        raise HTTPException(status_code=501, detail=str(e))
    except (ArtifactMismatchError, ModelLoadError) as e:
        raise HTTPException(status_code=503, detail=str(e))

    file_bytes = await image.read()
    try:
        batch = preprocess_image(file_bytes, artifacts.input_size)
    except InvalidImageError as e:
        raise HTTPException(status_code=422, detail=str(e))

    probabilities = artifacts.predict_proba(batch)

    threshold_mode = "global_override" if threshold is not None else "per_class_tuned"

    findings = []
    for condition, prob in zip(artifacts.condition_names, probabilities):
        applied_threshold = threshold if threshold is not None else artifacts.get_threshold(condition)
        findings.append(
            ConditionProbability(
                condition=condition,
                probability=float(prob),
                positive=prob >= applied_threshold,
                threshold_used=applied_threshold,
            )
        )
    positive_findings = [f.condition for f in findings if f.positive]

    # Generate a Grad-CAM heatmap for the single highest-probability
    # condition overall — matches the exploratory notebook's approach, and
    # avoids computing 14 heatmaps (one per condition) on every request.
    # This shows what the model is "looking at" for its best guess, even
    # when that guess falls below its threshold — useful diagnostic
    # information, not just a visualization of confirmed positives.
    top_class_index = max(range(len(probabilities)), key=lambda i: probabilities[i])
    heatmap_base64 = compute_gradcam_overlay(artifacts, batch, top_class_index)

    return ImageAnalysisResponse(
        image_type=image_type,
        findings=findings,
        positive_findings=positive_findings,
        threshold_mode=threshold_mode,
        global_threshold=threshold,
        heatmap_base64=heatmap_base64,
        model_version=artifacts.version,
    )
