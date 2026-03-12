"""Two-stage filtering backend implementation."""

from __future__ import annotations

import logging

from tobira.backends.protocol import BackendProtocol, PredictionResult

logger = logging.getLogger(__name__)


class TwoStageBackend:
    """Two-stage filtering backend for high-traffic environments.

    Uses a fast first-stage backend (e.g. FastText) for coarse classification,
    then routes grey-zone predictions to a precise second-stage backend
    (e.g. BERT/ONNX) for refined classification.

    Args:
        first_stage: Fast backend for coarse classification.
        second_stage: Precise backend for grey-zone re-classification.
        grey_zone: Tuple of (low, high) thresholds defining the grey zone.
            Predictions with scores between low and high are sent to the
            second stage. Defaults to (0.3, 0.7).
    """

    def __init__(
        self,
        first_stage: BackendProtocol,
        second_stage: BackendProtocol,
        grey_zone: tuple[float, float] = (0.3, 0.7),
    ) -> None:
        low, high = grey_zone
        if not (0.0 <= low < high <= 1.0):
            raise ValueError(
                f"grey_zone must satisfy 0 <= low < high <= 1, got ({low}, {high})"
            )
        self._first_stage = first_stage
        self._second_stage = second_stage
        self._grey_zone = grey_zone
        self._total = 0
        self._first_stage_decided = 0

    def predict(self, text: str) -> PredictionResult:
        """Run two-stage inference on the given text."""
        low, high = self._grey_zone

        first_result = self._first_stage.predict(text)
        self._total += 1

        if first_result.score >= high or first_result.score <= low:
            self._first_stage_decided += 1
            self._log_stats()
            return first_result

        second_result = self._second_stage.predict(text)
        self._log_stats()
        return second_result

    def _log_stats(self) -> None:
        """Log statistics about first-stage vs second-stage usage."""
        total = self._total
        decided = self._first_stage_decided
        second = total - decided
        logger.info(
            "TwoStageBackend stats: total=%d, first_stage_decided=%d (%.1f%%), "
            "second_stage=%d (%.1f%%)",
            total,
            decided,
            (decided / total * 100) if total else 0,
            second,
            (second / total * 100) if total else 0,
        )
