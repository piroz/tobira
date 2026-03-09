"""Mock tobira API server for integration testing.

Returns deterministic predictions based on keywords in the input text.
No ML model required.
"""

import hashlib
import re

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="tobira-mock")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float
    labels: dict[str, float]


class HealthResponse(BaseModel):
    status: str


SPAM_PATTERNS = re.compile(
    r"(buy now|free offer|click here|viagra|lottery|winner|"
    r"urgent|act now|limited time|casino|cheap|discount)",
    re.IGNORECASE,
)


def _classify(text: str) -> PredictResponse:
    """Simple rule-based classifier for testing."""
    matches = len(SPAM_PATTERNS.findall(text))
    if matches >= 3:
        spam_score = 0.95
    elif matches == 2:
        spam_score = 0.80
    elif matches == 1:
        spam_score = 0.60
    else:
        # Use hash for deterministic "ham" scores
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        spam_score = (h % 20) / 100.0  # 0.00 - 0.19

    ham_score = 1.0 - spam_score

    if spam_score >= 0.5:
        return PredictResponse(
            label="spam",
            score=spam_score,
            labels={"spam": spam_score, "ham": ham_score},
        )
    return PredictResponse(
        label="ham",
        score=ham_score,
        labels={"spam": spam_score, "ham": ham_score},
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest) -> PredictResponse:
    return _classify(req.text)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")
