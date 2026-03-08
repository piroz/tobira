"""Precision-Recall curve plotting (requires matplotlib)."""

from __future__ import annotations

from pathlib import Path

from tobira.evaluation.metrics import compute_metrics


def _import_matplotlib():  # type: ignore[no-untyped-def]
    """Import matplotlib, raising a helpful error if not installed."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install tobira[evaluation]"
        ) from None


def pr_curve_data(
    y_true: list[int],
    scores: list[float],
    steps: int = 100,
) -> tuple[list[float], list[float], list[float]]:
    """Compute Precision-Recall curve data points.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        scores: Predicted scores (higher means more likely positive).
        steps: Number of threshold steps.

    Returns:
        A tuple of (precisions, recalls, thresholds).

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    if len(y_true) == 0:
        raise ValueError("inputs must not be empty")

    lo, hi = min(scores), max(scores)
    precisions: list[float] = []
    recalls: list[float] = []
    thresholds: list[float] = []

    for i in range(steps + 1):
        threshold = lo + (hi - lo) * i / steps
        y_pred = [1 if s >= threshold else 0 for s in scores]
        m = compute_metrics(y_true, y_pred)
        precisions.append(m.precision)
        recalls.append(m.recall)
        thresholds.append(threshold)

    return precisions, recalls, thresholds


def plot_pr_curve(
    y_true: list[int],
    scores: list[float],
    output_path: str | Path | None = None,
    steps: int = 100,
) -> None:
    """Plot and optionally save a Precision-Recall curve.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        scores: Predicted scores (higher means more likely positive).
        output_path: If provided, save the plot to this path instead of showing.
        steps: Number of threshold steps.
    """
    plt = _import_matplotlib()

    precisions, recalls, _thresholds = pr_curve_data(y_true, scores, steps)

    fig, ax = plt.subplots()
    ax.plot(recalls, precisions, marker=".")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
