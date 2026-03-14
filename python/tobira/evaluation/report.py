"""Evaluation report generation (text and JSON)."""

from __future__ import annotations

import json

from tobira.evaluation.metrics import MetricsResult, MulticlassMetricsResult


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


def multiclass_to_text(result: MulticlassMetricsResult) -> str:
    """Format multiclass metrics as a human-readable text report.

    Args:
        result: A MulticlassMetricsResult instance.

    Returns:
        A formatted text string.
    """
    lines = [
        "=== Multiclass Evaluation Report ===",
        f"Accuracy:        {result.accuracy:.4f}",
        f"Macro Precision: {result.macro_precision:.4f}",
        f"Macro Recall:    {result.macro_recall:.4f}",
        f"Macro F1:        {result.macro_f1:.4f}",
        "",
        "Per-Class Metrics:",
    ]

    # Header
    label_width = max(len(pc.label) for pc in result.per_class)
    label_width = max(label_width, 5)  # minimum width
    header = (
        f"  {'Label':<{label_width}}"
        f"  {'Prec':>6}  {'Rec':>6}"
        f"  {'F1':>6}  {'Support':>7}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for pc in result.per_class:
        lines.append(
            f"  {pc.label:<{label_width}}"
            f"  {pc.precision:>6.4f}"
            f"  {pc.recall:>6.4f}"
            f"  {pc.f1:>6.4f}"
            f"  {pc.support:>7d}"
        )

    if result.confusion_matrix:
        lines.append("")
        lines.append("Confusion Matrix:")

        col_width = max(len(label) for label in result.labels)
        col_width = max(col_width, 4)

        # Header row
        header_row = " " * (col_width + 2) + "  ".join(
            f"{label:>{col_width}}" for label in result.labels
        )
        lines.append(f"  {header_row}")

        for i, label in enumerate(result.labels):
            row_values = "  ".join(
                f"{result.confusion_matrix[i][j]:>{col_width}d}"
                for j in range(len(result.labels))
            )
            lines.append(f"  {label:>{col_width}}  {row_values}")

    return "\n".join(lines)


def multiclass_to_json(result: MulticlassMetricsResult) -> str:
    """Format multiclass metrics as a JSON string.

    Args:
        result: A MulticlassMetricsResult instance.

    Returns:
        A JSON string.
    """
    data: dict[str, object] = {
        "accuracy": result.accuracy,
        "macro_precision": result.macro_precision,
        "macro_recall": result.macro_recall,
        "macro_f1": result.macro_f1,
        "per_class": [
            {
                "label": pc.label,
                "precision": pc.precision,
                "recall": pc.recall,
                "f1": pc.f1,
                "support": pc.support,
            }
            for pc in result.per_class
        ],
        "labels": list(result.labels),
        "confusion_matrix": [list(row) for row in result.confusion_matrix],
    }
    return json.dumps(data, indent=2)
