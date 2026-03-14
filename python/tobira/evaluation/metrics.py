"""Evaluation metrics: Accuracy, Precision, Recall, F1, and confusion matrix."""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class PerClassMetrics:
    """Per-class precision, recall, and F1.

    Attributes:
        label: Class label name.
        precision: Precision for this class.
        recall: Recall for this class.
        f1: F1 score for this class.
        support: Number of true instances for this class.
    """

    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class MulticlassMetricsResult:
    """Multiclass evaluation metrics result.

    Attributes:
        accuracy: Overall accuracy.
        macro_precision: Macro-averaged precision across all classes.
        macro_recall: Macro-averaged recall across all classes.
        macro_f1: Macro-averaged F1 across all classes.
        per_class: Per-class metrics for each label.
        labels: Ordered list of class labels.
        confusion_matrix: N×N confusion matrix as list of lists.
            ``confusion_matrix[i][j]`` is the count of samples with true label
            ``labels[i]`` predicted as ``labels[j]``.
    """

    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: tuple[PerClassMetrics, ...]
    labels: tuple[str, ...] = field(default_factory=tuple)
    confusion_matrix: tuple[tuple[int, ...], ...] = field(default_factory=tuple)


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


def multiclass_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Compute N×N confusion matrix for multiclass classification.

    Args:
        y_true: Ground truth string labels.
        y_pred: Predicted string labels.
        labels: Ordered list of class labels. If *None*, labels are
            inferred from the union of *y_true* and *y_pred* in sorted order.

    Returns:
        N×N confusion matrix as a tuple of tuples.
        ``result[i][j]`` counts samples with true label ``labels[i]``
        predicted as ``labels[j]``.

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("inputs must not be empty")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    label_to_idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]

    for t, p in zip(y_true, y_pred):
        ti = label_to_idx.get(t)
        pi = label_to_idx.get(p)
        if ti is not None and pi is not None:
            matrix[ti][pi] += 1

    return tuple(tuple(row) for row in matrix)


def compute_multiclass_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> MulticlassMetricsResult:
    """Compute per-class and macro-averaged metrics for multiclass classification.

    Args:
        y_true: Ground truth string labels.
        y_pred: Predicted string labels.
        labels: Ordered list of class labels. If *None*, labels are
            inferred from the union of *y_true* and *y_pred* in sorted order.

    Returns:
        A MulticlassMetricsResult instance.

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("inputs must not be empty")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    cm = multiclass_confusion_matrix(y_true, y_pred, labels)

    correct = sum(cm[i][i] for i in range(len(labels)))
    total = len(y_true)
    accuracy = correct / total

    per_class_list: list[PerClassMetrics] = []
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[j][i] for j in range(len(labels))) - tp
        fn = sum(cm[i][j] for j in range(len(labels))) - tp
        support = sum(cm[i])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_list.append(
            PerClassMetrics(
                label=label,
                precision=prec,
                recall=rec,
                f1=f1,
                support=support,
            )
        )

    n_classes = len(labels)
    if n_classes > 0:
        macro_precision = (
            sum(pc.precision for pc in per_class_list) / n_classes
        )
        macro_recall = (
            sum(pc.recall for pc in per_class_list) / n_classes
        )
        macro_f1 = (
            sum(pc.f1 for pc in per_class_list) / n_classes
        )
    else:
        macro_precision = 0.0
        macro_recall = 0.0
        macro_f1 = 0.0

    return MulticlassMetricsResult(
        accuracy=accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        per_class=tuple(per_class_list),
        labels=tuple(labels),
        confusion_matrix=cm,
    )
