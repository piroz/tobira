"""Optimal threshold search for binary classification."""

from __future__ import annotations

from dataclasses import dataclass

from tobira.evaluation.metrics import compute_metrics


@dataclass(frozen=True)
class ThresholdResult:
    """Result of threshold tuning.

    Attributes:
        threshold: The optimal threshold value.
        f1: F1 score at the optimal threshold.
        precision: Precision at the optimal threshold.
        recall: Recall at the optimal threshold.
    """

    threshold: float
    f1: float
    precision: float
    recall: float


def find_best_threshold_f1(
    y_true: list[int],
    scores: list[float],
    steps: int = 100,
) -> ThresholdResult:
    """Find the threshold that maximises F1 score.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        scores: Predicted scores (higher means more likely positive).
        steps: Number of candidate thresholds to evaluate.

    Returns:
        A ThresholdResult with the optimal threshold and metrics.

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if len(y_true) == 0:
        raise ValueError("inputs must not be empty")

    best: ThresholdResult | None = None
    lo, hi = min(scores), max(scores)

    for i in range(steps + 1):
        threshold = lo + (hi - lo) * i / steps
        y_pred = [1 if s >= threshold else 0 for s in scores]
        m = compute_metrics(y_true, y_pred)
        candidate = ThresholdResult(
            threshold=threshold,
            f1=m.f1,
            precision=m.precision,
            recall=m.recall,
        )
        if best is None or candidate.f1 > best.f1:
            best = candidate

    assert best is not None
    return best


def find_best_threshold_fpr(
    y_true: list[int],
    scores: list[float],
    max_fpr: float = 0.01,
    steps: int = 100,
) -> ThresholdResult:
    """Find the threshold that maximises F1 under a false positive rate constraint.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        scores: Predicted scores (higher means more likely positive).
        max_fpr: Maximum allowed false positive rate.
        steps: Number of candidate thresholds to evaluate.

    Returns:
        A ThresholdResult with the optimal threshold and metrics.

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if len(y_true) == 0:
        raise ValueError("inputs must not be empty")

    negatives = sum(1 for y in y_true if y == 0)
    best: ThresholdResult | None = None
    lo, hi = min(scores), max(scores)

    for i in range(steps + 1):
        threshold = lo + (hi - lo) * i / steps
        y_pred = [1 if s >= threshold else 0 for s in scores]
        m = compute_metrics(y_true, y_pred)

        fpr = m.confusion_matrix.fp / negatives if negatives > 0 else 0.0
        if fpr > max_fpr:
            continue

        candidate = ThresholdResult(
            threshold=threshold,
            f1=m.f1,
            precision=m.precision,
            recall=m.recall,
        )
        if best is None or candidate.f1 > best.f1:
            best = candidate

    if best is None:
        # All thresholds exceed FPR constraint; return highest threshold
        return ThresholdResult(threshold=hi, f1=0.0, precision=0.0, recall=0.0)

    return best
