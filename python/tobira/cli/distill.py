"""tobira distill - Knowledge distillation from teacher to student model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def register(subparsers: "argparse._SubParsersAction[Any]") -> None:
    """Register the ``distill`` subcommand.

    Args:
        subparsers: Subparser action from the parent parser.
    """
    parser = subparsers.add_parser(
        "distill",
        help="Run knowledge distillation from teacher to student model",
        description="Transfer knowledge from a large teacher model to a smaller "
        "student model using soft label distillation.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML config file. Must contain [teacher] and [distillation] "
        "sections.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to unlabeled data file (.csv or .jsonl). "
        "Required column: text.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save the distilled student model.",
    )
    parser.add_argument(
        "--soft-labels",
        default=None,
        metavar="PATH",
        help="Path to save/load soft labels (.jsonl). "
        "If the file exists, soft labels are loaded instead of regenerated.",
    )
    parser.set_defaults(func=_run)


def _load_texts(path: Path) -> list[str]:
    """Load texts from a CSV or JSONL file.

    Args:
        path: Path to the data file.

    Returns:
        List of text strings.

    Raises:
        ValueError: If file format is unsupported or 'text' column is missing.
    """
    import csv
    import json

    suffix = path.suffix.lower()
    texts: list[str] = []

    if suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "text" not in row:
                    raise ValueError("CSV must contain a 'text' column")
                texts.append(row["text"])
    elif suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if "text" not in record:
                        raise ValueError("JSONL records must contain a 'text' field")
                    texts.append(record["text"])
    else:
        raise ValueError(
            f"unsupported file format: {suffix!r} (expected .csv or .jsonl)"
        )

    return texts


def _run(args: argparse.Namespace) -> int:
    """Execute the ``distill`` subcommand.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from tobira.backends.factory import create_backend
    from tobira.config import load_toml
    from tobira.core.distillation import (
        DistillationConfig,
        distill,
        generate_soft_labels,
        load_soft_labels,
        save_soft_labels,
    )

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: config not found: {args.config}")
        return 1

    try:
        config = load_toml(args.config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    if "teacher" not in config:
        print("Error: missing [teacher] section in config")
        return 1

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: data file not found: {args.data}")
        return 1

    try:
        texts = _load_texts(data_path)
    except (ValueError, Exception) as exc:
        print(f"Error loading data: {exc}")
        return 1

    if not texts:
        print("Error: data file is empty")
        return 1

    # Generate or load soft labels
    soft_labels_path = Path(args.soft_labels) if args.soft_labels else None

    if soft_labels_path and soft_labels_path.exists():
        print(f"Loading existing soft labels from {soft_labels_path}")
        soft_labels = load_soft_labels(soft_labels_path)
    else:
        print(f"Generating soft labels using teacher model ({len(texts)} samples)...")
        try:
            teacher = create_backend(config["teacher"])
        except (ValueError, KeyError) as exc:
            print(f"Error creating teacher backend: {exc}")
            return 1

        soft_labels = generate_soft_labels(teacher, texts)

        if soft_labels_path:
            save_soft_labels(soft_labels, soft_labels_path)
            print(f"Soft labels saved to {soft_labels_path}")

    # Build distillation config
    dist_section: dict[str, Any] = config.get("distillation", {})
    dist_config = DistillationConfig(
        student_model_name=dist_section.get(
            "student_model_name", "distilbert-base-uncased"
        ),
        temperature=dist_section.get("temperature", 3.0),
        alpha=dist_section.get("alpha", 0.7),
        epochs=dist_section.get("epochs", 5),
        batch_size=dist_section.get("batch_size", 16),
        learning_rate=dist_section.get("learning_rate", 5e-5),
        max_length=dist_section.get("max_length", 512),
        device=dist_section.get("device"),
        label_names=dist_section.get("label_names", ["ham", "spam"]),
    )

    # Run distillation
    teacher_type = config["teacher"].get("type", "unknown")
    print(
        f"Starting distillation: {teacher_type} -> {dist_config.student_model_name}"
    )

    try:
        result = distill(
            soft_labels=soft_labels,
            output_path=args.output,
            config=dist_config,
            teacher_name=teacher_type,
        )
    except (ImportError, RuntimeError) as exc:
        print(f"Error during distillation: {exc}")
        return 1

    print()
    print("Distillation complete!")
    print(f"  Teacher:  {result.teacher_name}")
    print(f"  Student:  {result.student_model_name}")
    print(f"  Samples:  {result.num_samples}")
    print(f"  Epochs:   {result.epochs}")
    print(f"  Output:   {result.output_path}")

    return 0
