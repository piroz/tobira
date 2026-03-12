"""Pydantic request/response models for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MAX_TEXT_LENGTH = 102_400  # 100 KB


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"text": "Buy now! Limited offer!!!"}],
        },
    )

    text: str = Field(..., max_length=MAX_TEXT_LENGTH)


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"label": "spam", "score": 0.95, "labels": {"spam": 0.95, "ham": 0.05}},
            ],
        },
    )

    label: str
    score: float
    labels: dict[str, float]


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"status": "ok"}],
        },
    )

    status: str
