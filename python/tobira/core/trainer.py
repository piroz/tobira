"""Fine-tuning pipeline for BERT-based spam classification models.

Provides a training pipeline that wraps HuggingFace Transformers to
fine-tune sequence classification models on labeled spam/ham data.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _import_deps() -> tuple:
    """Lazily import torch and transformers."""
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except ImportError as exc:
        raise ImportError(
            "torch and transformers are required for fine-tuning. "
            "Install them with: pip install tobira[bert]"
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class TrainingResult:
    """Result of a fine-tuning run.

    Attributes:
        model_name: Base model name used for fine-tuning.
        output_path: Path where the fine-tuned model was saved.
        num_samples: Number of training samples.
        num_labels: Number of classification labels.
        epochs: Number of training epochs completed.
        final_loss: Average loss in the final epoch.
        onnx_path: Path to exported ONNX model, if export was performed.
    """

    model_name: str
    output_path: str
    num_samples: int
    num_labels: int
    epochs: int
    final_loss: float
    onnx_path: str | None = None


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning.

    Attributes:
        model_name: HuggingFace model name or path. Defaults to
            ``tohoku-nlp/bert-base-japanese-v3``.
        epochs: Number of training epochs. Defaults to 3.
        batch_size: Training batch size. Defaults to 16.
        learning_rate: Peak learning rate for AdamW. Defaults to 2e-5.
        weight_decay: Weight decay coefficient. Defaults to 0.01.
        warmup_ratio: Fraction of total steps for linear warmup.
            Defaults to 0.1.
        max_length: Maximum token sequence length. Defaults to 512.
        device: Device string (e.g. ``"cpu"``, ``"cuda"``). When *None*,
            automatically selects CUDA if available.
        label_names: Ordered list of classification label names.
        eval_split: Fraction of data held out for evaluation. Defaults to 0.1.
        seed: Random seed for reproducibility. Defaults to 42.
        export_onnx: Whether to export an ONNX model after training.
            Defaults to False.
        checkpoint_dir: Directory for saving epoch checkpoints.
            When *None*, checkpoints are not saved.
    """

    model_name: str = "tohoku-nlp/bert-base-japanese-v3"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 512
    device: str | None = None
    label_names: list[str] = field(default_factory=lambda: ["ham", "spam"])
    eval_split: float = 0.1
    seed: int = 42
    export_onnx: bool = False
    checkpoint_dir: str | None = None


def load_training_data(
    path: str | Path,
    *,
    text_column: str = "text",
    label_column: str = "label",
) -> list[dict[str, str]]:
    """Load labeled training data from CSV or JSONL.

    Each record must contain at least a text field and a label field.

    Args:
        path: Path to the data file (.csv or .jsonl).
        text_column: Name of the text column/field. Defaults to ``"text"``.
        label_column: Name of the label column/field. Defaults to ``"label"``.

    Returns:
        List of dicts with ``"text"`` and ``"label"`` keys.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the format is unsupported or required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data file not found: {path}")

    suffix = path.suffix.lower()
    records: list[dict[str, str]] = []

    if suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header: {path}")
            if text_column not in reader.fieldnames:
                raise ValueError(
                    f"CSV missing required column '{text_column}': {path}"
                )
            if label_column not in reader.fieldnames:
                raise ValueError(
                    f"CSV missing required column '{label_column}': {path}"
                )
            for row in reader:
                records.append(
                    {"text": row[text_column], "label": row[label_column]}
                )
    elif suffix == ".jsonl":
        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if text_column not in record:
                    raise ValueError(
                        f"JSONL line {line_num} missing '{text_column}': {path}"
                    )
                if label_column not in record:
                    raise ValueError(
                        f"JSONL line {line_num} missing '{label_column}': {path}"
                    )
                records.append(
                    {"text": record[text_column], "label": record[label_column]}
                )
    else:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Use .csv or .jsonl"
        )

    logger.info("Loaded %d training samples from %s", len(records), path)
    return records


def _split_data(
    records: list[dict[str, str]],
    eval_split: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split records into train and eval sets.

    Args:
        records: Full dataset.
        eval_split: Fraction for evaluation.
        seed: Random seed.

    Returns:
        Tuple of (train_records, eval_records).
    """
    import random

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n_eval = int(len(shuffled) * eval_split)
    if n_eval >= len(shuffled):
        n_eval = 0
        logger.warning(
            "Dataset too small for eval split; using all data for training"
        )
    elif eval_split > 0 and n_eval == 0 and len(shuffled) > 1:
        n_eval = 1

    if n_eval > 0:
        return shuffled[n_eval:], shuffled[:n_eval]
    return shuffled, []


def train(
    data: list[dict[str, str]],
    output_path: str | Path,
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Fine-tune a BERT model for spam classification.

    Trains a sequence classification model on the provided labeled data.
    Optionally exports the trained model to ONNX format.

    Args:
        data: List of dicts with ``"text"`` and ``"label"`` keys.
        output_path: Directory to save the fine-tuned model.
        config: Training configuration. Uses defaults if None.

    Returns:
        TrainingResult with training metadata.

    Raises:
        ValueError: If data is empty or labels are inconsistent.
        ImportError: If torch or transformers are not installed.
    """
    if not data:
        raise ValueError("data must not be empty")

    if config is None:
        config = TrainingConfig()

    torch, AutoModelForSequenceClassification, AutoTokenizer = _import_deps()

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Device selection
    if config.device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = config.device

    if device_str == "cpu":
        logger.warning(
            "Training on CPU. This will be slow for large datasets. "
            "Consider using a GPU for faster training."
        )

    device = torch.device(device_str)

    # Label mapping
    label_names = config.label_names
    num_labels = len(label_names)
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for i, name in enumerate(label_names)}

    # Validate labels in data
    data_labels = {d["label"] for d in data}
    unknown = data_labels - set(label_names)
    if unknown:
        raise ValueError(
            f"Data contains unknown labels {unknown}. "
            f"Expected one of {label_names}"
        )

    # Split data
    train_records, eval_records = _split_data(
        data, config.eval_split, config.seed
    )
    logger.info(
        "Data split: %d train, %d eval", len(train_records), len(eval_records)
    )

    # Load model and tokenizer
    logger.info("Loading model: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    # Prepare training data
    train_texts = [r["text"] for r in train_records]
    train_labels = torch.tensor(
        [label2id[r["label"]] for r in train_records], dtype=torch.long
    )

    train_encodings = tokenizer(
        train_texts,
        padding=True,
        truncation=True,
        max_length=config.max_length,
        return_tensors="pt",
    )

    # Prepare eval data
    eval_encodings = None
    eval_labels = None
    if eval_records:
        eval_texts = [r["text"] for r in eval_records]
        eval_labels = torch.tensor(
            [label2id[r["label"]] for r in eval_records], dtype=torch.long
        )
        eval_encodings = tokenizer(
            eval_texts,
            padding=True,
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
        )

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate)

    # Learning rate scheduler with warmup
    dataset_size = len(train_texts)
    batch_size = min(config.batch_size, dataset_size)
    total_steps = (dataset_size // batch_size) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1.0, float(warmup_steps))
        remaining = max(1.0, float(total_steps - warmup_steps))
        return max(0.0, float(total_steps - current_step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Set seed for reproducibility
    torch.manual_seed(config.seed)

    # Training loop
    logger.info(
        "Starting training: %d samples, %d epochs, batch_size=%d",
        dataset_size,
        config.epochs,
        batch_size,
    )

    global_step = 0
    final_loss = 0.0

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        indices = torch.randperm(dataset_size)

        for start in range(0, dataset_size, batch_size):
            end = min(start + batch_size, dataset_size)
            batch_idx = indices[start:end]

            batch_inputs: dict[str, Any] = {
                k: v[batch_idx].to(device) for k, v in train_encodings.items()
            }
            batch_labels = train_labels[batch_idx].to(device)

            outputs = model(**batch_inputs, labels=batch_labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss)
            num_batches += 1
            global_step += 1

        avg_loss = total_loss / max(num_batches, 1)

        # Evaluation
        eval_msg = ""
        if eval_encodings is not None and eval_labels is not None:
            model.eval()
            with torch.no_grad():
                eval_inputs: dict[str, Any] = {
                    k: v.to(device) for k, v in eval_encodings.items()
                }
                eval_outputs = model(**eval_inputs)
                eval_preds = torch.argmax(eval_outputs.logits, dim=-1)
                eval_correct = (
                    eval_preds == eval_labels.to(device)
                ).sum().item()
                eval_total = len(eval_records)
                eval_acc = eval_correct / max(eval_total, 1)
                eval_msg = f", eval_acc: {eval_acc:.4f}"

        logger.info(
            "Epoch %d/%d - loss: %.4f%s",
            epoch + 1,
            config.epochs,
            avg_loss,
            eval_msg,
        )

        # Checkpoint saving
        if config.checkpoint_dir is not None:
            ckpt_path = Path(config.checkpoint_dir) / f"checkpoint-epoch-{epoch + 1}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_path))
            tokenizer.save_pretrained(str(ckpt_path))
            logger.info("Checkpoint saved to %s", ckpt_path)

        final_loss = avg_loss

    # Save final model
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info("Fine-tuned model saved to %s", output_path)

    # Optional ONNX export
    onnx_path: str | None = None
    if config.export_onnx:
        try:
            from tobira.core.export import export_onnx, quantize_dynamic

            onnx_output = output_path / "model.onnx"
            export_onnx(str(output_path), str(onnx_output))
            quantized = quantize_dynamic(str(onnx_output))
            onnx_path = str(quantized)
            logger.info("ONNX model exported to %s", onnx_path)
        except ImportError:
            logger.warning(
                "ONNX export requested but onnxruntime is not installed. "
                "Install with: pip install tobira[onnx]"
            )

    return TrainingResult(
        model_name=config.model_name,
        output_path=str(output_path),
        num_samples=dataset_size,
        num_labels=num_labels,
        epochs=config.epochs,
        final_loss=final_loss,
        onnx_path=onnx_path,
    )
