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

    For binary classification, the grey zone is defined by score thresholds:
    predictions with confidence between *low* and *high* go to the second stage.

    For multiclass classification (more than 2 labels in the result), the grey
    zone is determined by the confidence of the top prediction: if the highest
    score is below *high*, the prediction is considered uncertain and routed to
    the second stage.

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

    def _is_confident(self, result: PredictionResult) -> bool:
        """Check whether the first-stage result is confident enough.

        For binary results (2 labels), uses the original low/high threshold
        logic on the top score.  For multiclass results (>2 labels), the
        prediction is confident when the top score is >= *high*.
        """
        low, high = self._grey_zone
        is_binary = len(result.labels) <= 2
        if is_binary:
            return result.score >= high or result.score <= low
        return result.score >= high

    def predict(self, text: str) -> PredictionResult:
        """Run two-stage inference on the given text."""
        first_result = self._first_stage.predict(text)
        self._total += 1

        if self._is_confident(first_result):
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
