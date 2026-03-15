"""Pydantic request/response models for the serving API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

MAX_TEXT_LENGTH = 102_400  # 100 KB


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"text": "Buy now! Limited offer!!!"},
                {"text": "お買い得！今すぐ購入！", "language": "ja"},
            ],
        },
    )

    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    language: str | None = Field(
        default=None,
        description="ISO 639-1 language code (e.g. 'ja', 'en', 'ko'). "
        "When omitted, language is auto-detected.",
    )


class AIGeneratedInfo(BaseModel):
    """AI-generated text detection result embedded in prediction responses."""

    detected: bool
    confidence: float
    indicators: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "label": "spam",
                    "score": 0.95,
                    "labels": {"spam": 0.95, "ham": 0.05},
                    "language": "en",
                },
                {
                    "label": "phishing",
                    "score": 0.82,
                    "labels": {
                        "ham": 0.05,
                        "phishing": 0.82,
                        "malware": 0.03,
                        "financial_fraud": 0.04,
                        "lottery": 0.01,
                        "romance_scam": 0.01,
                        "drug": 0.02,
                        "fake_service": 0.01,
                        "tech_support": 0.01,
                    },
                },
                {
                    "label": "spam",
                    "score": 0.92,
                    "labels": {"spam": 0.92, "ham": 0.08},
                    "language": "en",
                    "ai_generated": {
                        "detected": True,
                        "confidence": 0.85,
                        "indicators": [
                            "low_entropy",
                            "uniform_vocabulary",
                        ],
                    },
                },
            ],
        },
    )

    label: str
    score: float
    labels: dict[str, float]
    language: str | None = Field(
        default=None,
        description="Detected or specified language code.",
    )
    ai_generated: AIGeneratedInfo | None = Field(
        default=None,
        description="AI-generated text detection result. "
        "Present only when ai_detection is enabled in server config.",
    )


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Buy now! Limited offer!!!",
                    "label": "spam",
                    "source": "rspamd",
                },
            ],
        },
    )

    text: str = Field(..., max_length=MAX_TEXT_LENGTH)
    label: str = Field(..., pattern=r"^(spam|ham)$")
    source: str = Field(default="unknown", max_length=256)


class FeedbackResponse(BaseModel):
    """Response body for POST /feedback."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"status": "accepted", "id": "feedback-uuid"},
            ],
        },
    )

    status: str
    id: str


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"status": "ok"}],
        },
    )

    status: str
