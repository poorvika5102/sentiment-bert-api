"""
schemas.py
Defines the request and response data shapes for the API.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="Text to analyze",
        examples=["This movie was absolutely fantastic!"]
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_whitespace(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v.strip()


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of texts to analyze (max 32)",
    )


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    positive_score: float
    negative_score: float
    label_id: int


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_path: str
    device: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_path: str
    labels: dict
    max_length: int
    device: str
    parameters: Optional[str] = None