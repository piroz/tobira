"""tobira.evaluation - model evaluation metrics, reports, and threshold tuning."""

from tobira.evaluation.metrics import (
    ConfusionMatrix,
    MetricsResult,
    MulticlassMetricsResult,
    PerClassMetrics,
    compute_metrics,
    compute_multiclass_metrics,
    confusion_matrix,
    multiclass_confusion_matrix,
)
from tobira.evaluation.plot import plot_pr_curve, pr_curve_data
from tobira.evaluation.report import (
    multiclass_to_json,
    multiclass_to_text,
    to_json,
    to_text,
)
from tobira.evaluation.threshold import (
    ThresholdResult,
    find_best_threshold_f1,
    find_best_threshold_fpr,
)

__all__ = [
    "ConfusionMatrix",
    "MetricsResult",
    "MulticlassMetricsResult",
    "PerClassMetrics",
    "ThresholdResult",
    "compute_metrics",
    "compute_multiclass_metrics",
    "confusion_matrix",
    "find_best_threshold_f1",
    "find_best_threshold_fpr",
    "multiclass_confusion_matrix",
    "multiclass_to_json",
    "multiclass_to_text",
    "plot_pr_curve",
    "pr_curve_data",
    "to_json",
    "to_text",
]
