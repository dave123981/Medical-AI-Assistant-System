from typing import List, Optional
from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    condition: str
    age: Optional[int] = None
    current_medications: Optional[List[str]] = None
    allergies: Optional[List[str]] = None


class DrugRecommendation(BaseModel):
    drug: str
    score: float
    review_count: int
    contraindicated: bool
    contraindication_reasons: List[str]


class RecommendationResponse(BaseModel):
    condition: str  # exactly what the caller sent
    condition_matched: Optional[str] = None  # the real vocabulary entry matched, or None
    recommendations: List[DrugRecommendation]
    contraindication_rules_loaded: bool  # False means no rules file was found at all —
    # every "contraindicated: false" below means "not checked," not "confirmed safe"
    disclaimer: str
    model_version: str


class ConditionsResponse(BaseModel):
    conditions: List[str]
    count: int
