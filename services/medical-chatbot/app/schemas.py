from typing import List, Optional
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    context_disease: Optional[str] = None  # soft-biases retrieval in v1; see model_loader.py
    conversation_id: Optional[str] = None  # accepted, unused in v1 (no conversation memory yet)

class AskResponse(BaseModel):
    answer: str
    matched_question: str
    similarity_score: float
    confident: bool  # False means similarity was below the confidence threshold —
    # the answer is still returned (not withheld), but the frontend should show a caveat
    sources: List[str]
    disclaimer: str
    model_version: str