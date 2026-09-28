from typing import List, Optional
from pydantic import BaseModel


class AskRequest(BaseModel):
    question: str
    context_disease: Optional[str] = None  # soft-biases retrieval in v1; see model_loader.py
    conversation_id: Optional[str] = None  # accepted, unused in v1 (no conversation memory yet)


class AskResponse(BaseModel):
    answer: str  # LLM-generated text when v4 generation is active, otherwise
    # identical to source_answer (backward-compatible with v1-v3 responses)
    generated: bool  # True if `answer` was produced by an LLM (v4), False if
    # it's the raw retrieved text verbatim (v1-v3)
    source_answer: str  # the raw retrieved answer, ALWAYS present — lets a
    # caller verify a generated answer against its actual source, even when
    # generated is True
    matched_question: str
    similarity_score: float
    confident: bool  # False means similarity was below the confidence threshold —
    # the answer is still returned (not withheld), but the frontend should show a caveat
    sources: List[str]
    disclaimer: str
    model_version: str