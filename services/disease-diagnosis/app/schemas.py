"""
Pydantic schemas for Service 1 — Disease Diagnosis Assistant.

These mirror gateway/internal/models/diagnosis.go field-for-field so the
Go gateway and this Python service never drift out of sync. If you add a
field on one side, add it on the other.
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class VitalSigns(BaseModel):
    temperature_c: Optional[float] = None
    heart_rate_bpm: Optional[int] = None
    respiratory_rate: Optional[int] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None


class PatientInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., pattern="^(male|female|other)$")
    symptoms: List[str] = Field(..., min_length=1)
    vital_signs: Optional[VitalSigns] = None
    lab_values: Optional[Dict[str, float]] = None
    medical_history: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "age": 34,
                "gender": "female",
                "symptoms": ["headache", "fatigue", "nausea"],
                "vital_signs": None,
                "lab_values": None,
                "medical_history": []
            }
        }


class DiseasePrediction(BaseModel):
    disease: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_disease: str
    confidence: float
    top_candidates: List[DiseasePrediction]
    description: Optional[str] = None
    precautions: Optional[List[str]] = None
    model_version: str
