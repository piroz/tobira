"""tobira train - Fine-tuning pipeline (train → evaluate → export)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def register(subparsers: "argparse._SubParsersAction[Any]") -> None:
    """Register the ``train`` subcommand.

    Args:
        subparsers: Subparser action from the parent parser.
    """
    parser = subparsers.add_parser(
        "train",
        help="Run fine-tuning pipeline (train → evaluate → ONNX export)",
        description="Execute the full training pipeline: data preparation, "
        "fine-tuning, evaluation, and optional ONNX export.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a TOML config file. Must contain [training] section.",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to labelled data file (.csv or .jsonl). "
        "Required columns: text, label.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Directory to save the trained model and artifacts.",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        default=False,
        help="Skip ONNX export after training.",
    )
    parser.add_argument(
        "--split-ratio",
        default="0.8,0.1,0.1",
        help="Train/val/test split ratio (default: 0.8,0.1,0.1).",
    )
    parser.set_defaults(func=_run)


def _load_labelled_data(path: Path) -> list[dict[str, str]]:
    """Load labelled data from a CSV or JSONL file.

    Args:
        path: Path to the data file.

    Returns:
        List of dicts with 'text' and 'label' keys.

    Raises:
        ValueError: If file format is unsupported or required columns missing.
    """
    suffix = path.suffix.lower()
    rows: list[dict[str, str]] = []

    if suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "text" not in row or "label" not in row:
                    raise ValueError(
                        "CSV must contain 'text' and 'label' columns"
                    )
                rows.append(row)
    elif suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if "text" not in record or "label" not in record:
                        raise ValueError(
                            "JSONL records must contain 'text' and 'label' fields"
                        )
                    rows.append(record)
    else:
        raise ValueError(
            f"unsupported file format: {suffix!r} (expected .csv or .jsonl)"
        )

    return rows


def _parse_split_ratio(ratio_str: str) -> tuple[float, float, float]:
    """Parse a comma-separated split ratio string.

    Args:
        ratio_str: Comma-separated ratio (e.g. '0.8,0.1,0.1').

    Returns:
        Tuple of (train, val, test) ratios.

    Raises:
        ValueError: If the ratio is invalid.
    """
    parts = ratio_str.split(",")
    if len(parts) != 3:
        raise ValueError(
            f"split ratio must have 3 parts (train,val,test), got {len(parts)}"
        )
    ratios = tuple(float(p.strip()) for p in parts)
    total = sum(ratios)
    if abs(total - 1.0) > 0.01:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    return ratios[0], ratios[1], ratios[2]


def _split_data(
    rows: list[dict[str, str]],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Split data into train/val/test sets.

    Args:
        rows: List of data rows.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.

    Returns:
        Tuple of (train, val, test) data lists.
    """
    n = len(rows)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def _run(args: argparse.Namespace) -> int:
    """Execute the ``train`` subcommand.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    from tobira.config import load_toml

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

    if "training" not in config:
        print("Error: missing [training] section in config")
        return 1

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: data file not found: {args.data}")
        return 1

    try:
        rows = _load_labelled_data(data_path)
    except (ValueError, Exception) as exc:
        print(f"Error loading data: {exc}")
        return 1

    if not rows:
        print("Error: data file is empty")
        return 1

    # Parse split ratio
    try:
        train_ratio, val_ratio, _test_ratio = _parse_split_ratio(args.split_ratio)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    # Split data
    train_data, val_data, test_data = _split_data(rows, train_ratio, val_ratio)
    print(
        f"Data split: train={len(train_data)}, "
        f"val={len(val_data)}, test={len(test_data)}"
    )

    if not train_data:
        print("Error: training set is empty after split")
        return 1

    # Step 1: Preprocessing (optional)
    training_config: dict[str, Any] = config["training"]
    preprocess_section = config.get("preprocessing")

    if preprocess_section:
        try:
            from tobira.preprocessing.pipeline import PreprocessingPipeline
        except ImportError:
            print(
                "Error: preprocessing.pipeline module not available. "
                "Install required dependencies or check your installation."
            )
            return 1

        print("Running preprocessing pipeline...")
        try:
            pipeline = PreprocessingPipeline()
            for dataset in (train_data, val_data, test_data):
                for row in dataset:
                    pp_result = pipeline.run(row["text"])
                    row["text"] = pp_result.text
        except Exception as exc:
            print(f"Error during preprocessing: {exc}")
            return 1
        print("Preprocessing complete.")

    # Step 2: Fine-tuning
    try:
        from tobira.core.trainer import (
            TrainingConfig,
            train,
        )
    except ImportError:
        print(
            "Error: core.trainer module not available. "
            "Install required dependencies or check your installation."
        )
        return 1

    output_dir = Path(args.output)
    model_name = training_config.get("model_name", "bert-base-uncased")
    train_config = TrainingConfig(
        model_name=model_name,
        epochs=training_config.get("epochs", 3),
        batch_size=training_config.get("batch_size", 16),
        learning_rate=training_config.get("learning_rate", 5e-5),
        max_length=training_config.get("max_length", 512),
        device=training_config.get("device"),
        label_names=training_config.get("label_names", ["ham", "spam"]),
    )

    print(f"Starting fine-tuning: {model_name}")
    print(f"  Epochs:     {train_config.epochs}")
    print(f"  Batch size: {train_config.batch_size}")

    try:
        result = train(
            data=train_data,
            output_path=str(output_dir),
            config=train_config,
        )
    except (ImportError, RuntimeError) as exc:
        print(f"Error during training: {exc}")
        return 1

    print(f"Training complete: {result.output_path}")

    # Step 3: Evaluation
    if test_data:
        from tobira.evaluation.metrics import compute_metrics

        test_texts = [row["text"] for row in test_data]
        test_labels = [int(row["label"]) for row in test_data]

        print("Running evaluation on test set...")
        try:
            from tobira.backends.factory import create_backend

            backend_config = {
                "type": "bert",
                "model_path": str(result.output_path),
            }
            backend = create_backend(backend_config)
            predictions = [backend.predict(t) for t in test_texts]
            y_pred = [1 if p.score >= 0.5 else 0 for p in predictions]
        except (ImportError, RuntimeError) as exc:
            print(f"Warning: could not run evaluation inference: {exc}")
            y_pred = None

        if y_pred is not None:
            metrics = compute_metrics(test_labels, y_pred)
            print()
            print("Evaluation Results:")
            print(f"  Accuracy:  {metrics.accuracy:.4f}")
            print(f"  Precision: {metrics.precision:.4f}")
            print(f"  Recall:    {metrics.recall:.4f}")
            print(f"  F1:        {metrics.f1:.4f}")
    else:
        print("No test data available, skipping evaluation.")

    # Step 4: ONNX export
    if not args.skip_export:
        from tobira.core.export import export_onnx, quantize_dynamic

        onnx_path = output_dir / "model.onnx"
        print(f"Exporting to ONNX: {onnx_path}")

        try:
            exported = export_onnx(
                model_name=str(result.output_path),
                output_path=onnx_path,
            )
            print(f"ONNX model saved: {exported}")

            quantized_path = output_dir / "model_int8.onnx"
            quantized = quantize_dynamic(exported, quantized_path)
            print(f"Quantized model saved: {quantized}")
        except (ImportError, RuntimeError) as exc:
            print(f"Warning: ONNX export failed: {exc}")
            print("Training artifacts are still available in the output directory.")
    else:
        print("ONNX export skipped (--skip-export).")

    # Summary
    print()
    print("Training pipeline complete!")
    print(f"  Model: {model_name}")
    print(f"  Output: {output_dir}")

    return 0
