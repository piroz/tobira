"""Evaluation metrics: Accuracy, Precision, Recall, F1, and confusion matrix."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfusionMatrix:
    """Binary confusion matrix.

    Attributes:
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        tn: True negatives.
    """

    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True)
class MetricsResult:
    """Evaluation metrics result.

    Attributes:
        accuracy: Overall accuracy.
        precision: Precision for the positive class.
        recall: Recall for the positive class.
        f1: F1 score for the positive class.
        confusion_matrix: Binary confusion matrix.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: ConfusionMatrix


def confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
) -> ConfusionMatrix:
    """Compute binary confusion matrix.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        A ConfusionMatrix instance.

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("inputs must not be empty")

    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn)


def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
) -> MetricsResult:
    """Compute accuracy, precision, recall, F1, and confusion matrix.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        A MetricsResult instance.
    """
    cm = confusion_matrix(y_true, y_pred)

    total = cm.tp + cm.fp + cm.fn + cm.tn
    accuracy = (cm.tp + cm.tn) / total

    precision = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) > 0 else 0.0
    recall = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return MetricsResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm,
    )
