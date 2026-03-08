"""Pydantic request/response models for the serving API."""

from __future__ import annotations

from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    text: str


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    label: str
    score: float
    labels: dict[str, float]


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str
