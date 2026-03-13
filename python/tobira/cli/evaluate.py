"""tobira evaluate - Evaluate model performance on labelled datasets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV file and return rows as dicts."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _load_jsonl(path: Path) -> list[dict[str, str]]:
    """Load a JSONL file and return rows as dicts."""
    rows: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_dataset(path: Path) -> list[dict[str, str]]:
    """Load a dataset from CSV or JSONL based on file extension.

    Args:
        path: Path to the dataset file.

    Returns:
        List of row dicts with at least 'text' and 'label' keys.

    Raises:
        ValueError: If file extension is not .csv or .jsonl.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".jsonl":
        return _load_jsonl(path)
    raise ValueError(
        f"unsupported file format: {suffix!r} (expected .csv or .jsonl)"
    )


def register(subparsers: "argparse._SubParsersAction[Any]") -> None:
    """Register the ``evaluate`` subcommand.

    Args:
        subparsers: Subparser action from the parent parser.
    """
    parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate model on a labelled dataset",
        description="Run evaluation metrics on a labelled CSV/JSONL dataset. "
        "Supports backend inference or pre-computed scores.",
    )
    parser.add_argument(
        "dataset",
        help="Path to a labelled dataset file (.csv or .jsonl). "
        "Required columns: text, label. "
        "Optional column: score (pre-computed prediction score).",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a TOML config file for backend inference. "
        "Not required if the dataset contains a 'score' column.",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        dest="output_format",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--plot",
        default=None,
        metavar="PATH",
        help="Save a Precision-Recall curve image to PATH",
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        default=False,
        help="Find the optimal classification threshold",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    """Execute the ``evaluate`` subcommand.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from tobira.evaluation.metrics import compute_metrics
    from tobira.evaluation.report import to_json, to_text

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: dataset not found: {args.dataset}")
        return 1

    try:
        rows = _load_dataset(dataset_path)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}")
        return 1

    if not rows:
        print("Error: dataset is empty")
        return 1

    # Validate required columns
    first_row = rows[0]
    if "text" not in first_row or "label" not in first_row:
        print("Error: dataset must contain 'text' and 'label' columns")
        return 1

    y_true = [int(row["label"]) for row in rows]
    has_score = "score" in first_row

    # Determine scores: from dataset or via backend inference
    scores: list[float] | None = None
    if has_score:
        scores = [float(row["score"]) for row in rows]
    elif args.config is not None:
        from tobira.backends.factory import create_backend
        from tobira.config import load_toml

        try:
            config = load_toml(args.config)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return 1

        if "backend" not in config:
            print("Error: missing [backend] section in config")
            return 1

        backend = create_backend(config["backend"])
        scores = [backend.predict(row["text"]).score for row in rows]
    else:
        print(
            "Error: either provide a 'score' column in the dataset "
            "or specify --config for backend inference"
        )
        return 1

    # Convert scores to binary predictions at 0.5 threshold
    threshold = 0.5
    y_pred = [1 if s >= threshold else 0 for s in scores]
    result = compute_metrics(y_true, y_pred)

    # Output metrics
    if args.output_format == "json":
        print(to_json(result))
    else:
        print(to_text(result))

    # Threshold tuning
    if args.tune_threshold:
        from tobira.evaluation.threshold import find_best_threshold_f1

        best = find_best_threshold_f1(y_true, scores)
        print()
        print(f"Optimal threshold: {best.threshold:.4f}")
        print(f"  F1:        {best.f1:.4f}")
        print(f"  Precision: {best.precision:.4f}")
        print(f"  Recall:    {best.recall:.4f}")

    # PR curve plot
    if args.plot is not None:
        from tobira.evaluation.plot import plot_pr_curve

        try:
            plot_pr_curve(y_true, scores, output_path=args.plot)
            print(f"\nPR curve saved to {args.plot}")
        except ImportError as exc:
            print(f"\nWarning: {exc}")

    return 0
