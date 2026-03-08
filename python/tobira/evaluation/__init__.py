"""tobira.evaluation - model evaluation metrics, reports, and threshold tuning."""

from tobira.evaluation.metrics import (
    ConfusionMatrix,
    MetricsResult,
    compute_metrics,
    confusion_matrix,
)
from tobira.evaluation.plot import plot_pr_curve, pr_curve_data
from tobira.evaluation.report import to_json, to_text
from tobira.evaluation.threshold import (
    ThresholdResult,
    find_best_threshold_f1,
    find_best_threshold_fpr,
)

__all__ = [
    "ConfusionMatrix",
    "MetricsResult",
    "ThresholdResult",
    "compute_metrics",
    "confusion_matrix",
    "find_best_threshold_f1",
    "find_best_threshold_fpr",
    "plot_pr_curve",
    "pr_curve_data",
    "to_json",
    "to_text",
]
