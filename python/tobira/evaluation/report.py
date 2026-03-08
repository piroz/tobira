"""Evaluation report generation (text and JSON)."""

from __future__ import annotations

import json

from tobira.evaluation.metrics import MetricsResult


def to_text(result: MetricsResult) -> str:
    """Format metrics as a human-readable text report.

    Args:
        result: A MetricsResult instance.

    Returns:
        A formatted text string.
    """
    cm = result.confusion_matrix
    lines = [
        "=== Evaluation Report ===",
        f"Accuracy:  {result.accuracy:.4f}",
        f"Precision: {result.precision:.4f}",
        f"Recall:    {result.recall:.4f}",
        f"F1:        {result.f1:.4f}",
        "",
        "Confusion Matrix:",
        f"  TP={cm.tp}  FP={cm.fp}",
        f"  FN={cm.fn}  TN={cm.tn}",
    ]
    return "\n".join(lines)


def to_json(result: MetricsResult) -> str:
    """Format metrics as a JSON string.

    Args:
        result: A MetricsResult instance.

    Returns:
        A JSON string.
    """
    cm = result.confusion_matrix
    data = {
        "accuracy": result.accuracy,
        "precision": result.precision,
        "recall": result.recall,
        "f1": result.f1,
        "confusion_matrix": {
            "tp": cm.tp,
            "fp": cm.fp,
            "fn": cm.fn,
            "tn": cm.tn,
        },
    }
    return json.dumps(data, indent=2)
