from typing import List, Optional
from pydantic import BaseModel


class ConditionProbability(BaseModel):
    condition: str
    probability: float
    positive: bool  # True if probability >= this condition's decision threshold
    threshold_used: float  # the actual threshold applied to THIS condition,
                            # since thresholds are now per-class, not one global value


class ImageAnalysisResponse(BaseModel):
    image_type: str
    findings: List[ConditionProbability]
    positive_findings: List[str]
    # "per_class_tuned": each condition used its own F1-optimal threshold from
    # per_class_thresholds.json (falls back to 0.5 for any condition missing
    # from that file, or for models with no thresholds file at all).
    # "global_override": the caller passed an explicit threshold, applied
    # uniformly to every condition instead.
    threshold_mode: str
    global_threshold: Optional[float] = None  # set only when threshold_mode == "global_override"
    heatmap_base64: Optional[str] = None  # None if the model doesn't support Grad-CAM (e.g. v1)
    model_version: str


class ConditionsResponse(BaseModel):
    image_type: str
    conditions: List[str]
    count: int
