from typing import List, Optional
from pydantic import BaseModel


class ConditionProbability(BaseModel):
    condition: str
    probability: float
    positive: bool  # True if probability >= the service's decision threshold


class ImageAnalysisResponse(BaseModel):
    image_type: str
    findings: List[ConditionProbability]
    # Convenience list — just the condition names where positive=True. Empty
    # means the model found nothing above threshold, i.e. "No Finding".
    positive_findings: List[str]
    threshold: float
    # None until v4 adds Grad-CAM. Keeping the field here now (rather than
    # adding it later) means the response shape never changes across
    # versions — same principle Service 1 follows.
    heatmap_base64: Optional[str] = None
    model_version: str


class ConditionsResponse(BaseModel):
    image_type: str
    conditions: List[str]
    count: int
