"""Backend protocol definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class PredictionResult:
    """Result of a backend prediction.

    Attributes:
        label: The predicted label (e.g. "spam" or "ham").
        score: Confidence score for the predicted label.
        labels: Scores for all labels.
    """

    label: str
    score: float
    labels: dict[str, float]


@runtime_checkable
class BackendProtocol(Protocol):
    """Protocol that all inference backends must implement."""

    def predict(self, text: str) -> PredictionResult:
        """Run inference on the given text and return a prediction."""
        ...
