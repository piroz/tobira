"""HuggingFace Hub integration for uploading and downloading tobira models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _import_hub_deps() -> tuple:
    """Lazily import huggingface_hub."""
    try:
        from huggingface_hub import HfApi, ModelCard, ModelCardData
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for Hub integration. "
            "Install it with: pip install tobira[hub]"
        ) from exc
    return HfApi, ModelCard, ModelCardData


def generate_model_card(
    *,
    repo_id: str,
    language: str = "ja",
    license_name: str = "apache-2.0",
    base_model: str = "tohoku-nlp/bert-base-japanese-v3",
    metrics: dict[str, float] | None = None,
) -> str:
    """Generate a model card (README.md) for a tobira spam classifier.

    Args:
        repo_id: HuggingFace Hub repository ID (e.g. ``velocitylabo-org/tobira-spam-bert-ja``).
        language: Language code for the model.
        license_name: License identifier.
        base_model: Base model used for fine-tuning.
        metrics: Evaluation metrics dict (e.g. ``{"f1": 0.95, "accuracy": 0.96}``).

    Returns:
        Model card content as a string.
    """
    _, ModelCard, ModelCardData = _import_hub_deps()

    card_data = ModelCardData(
        language=language,
        license=license_name,
        library_name="transformers",
        tags=["spam-detection", "email", "text-classification", "tobira"],
        pipeline_tag="text-classification",
    )

    metrics_section = ""
    if metrics:
        lines = [f"| {k} | {v:.4f} |" for k, v in metrics.items()]
        metrics_section = (
            "\n## Evaluation Results\n\n"
            "| Metric | Value |\n|--------|-------|\n"
            + "\n".join(lines)
        )

    content = f"""---
{card_data.to_yaml()}
---

# {repo_id}

Japanese email spam classifier fine-tuned from [{base_model}](https://huggingface.co/{base_model}).

## Usage

```python
from tobira.backends.factory import create_backend

backend = create_backend({{"type": "bert", "model_name": "{repo_id}"}})
result = backend.predict("メール本文をここに入力")
print(result.label, result.score)
```

## ONNX Inference

```python
backend = create_backend({{"type": "onnx", "model_path": "model.onnx"}})
result = backend.predict("メール本文をここに入力")
print(result.label, result.score)
```
{metrics_section}

## Base Model

- [{base_model}](https://huggingface.co/{base_model})

## License

{license_name}
"""
    return content


def push_to_hub(
    *,
    model_dir: str | Path,
    repo_id: str,
    token: str | None = None,
    private: bool = False,
    metrics: dict[str, float] | None = None,
    license_name: str = "apache-2.0",
    language: str = "ja",
    base_model: str = "tohoku-nlp/bert-base-japanese-v3",
) -> str:
    """Upload a trained model directory to HuggingFace Hub.

    The directory should contain model files (e.g. ``pytorch_model.bin``,
    ``config.json``, ``tokenizer.json``, and optionally ONNX files).
    A model card (``README.md``) is generated and uploaded alongside the
    model files.

    Args:
        model_dir: Local directory containing model files.
        repo_id: Target repository on HuggingFace Hub.
        token: HuggingFace API token. Falls back to ``HF_TOKEN`` env var.
        private: Whether to create a private repository.
        metrics: Evaluation metrics to include in the model card.
        license_name: License identifier for the model card.
        language: Language code for the model card.
        base_model: Base model used for fine-tuning.

    Returns:
        URL of the created/updated repository.

    Raises:
        ImportError: If huggingface_hub is not installed.
        FileNotFoundError: If model_dir does not exist.
    """
    HfApi, _, _ = _import_hub_deps()

    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    # Generate and write model card
    card_content = generate_model_card(
        repo_id=repo_id,
        language=language,
        license_name=license_name,
        base_model=base_model,
        metrics=metrics,
    )
    readme_path = model_dir / "README.md"
    readme_path.write_text(card_content, encoding="utf-8")

    # Upload entire directory
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(model_dir),
    )

    url: str = f"https://huggingface.co/{repo_id}"
    return url


def pull_from_hub(
    *,
    repo_id: str,
    local_dir: str | Path,
    token: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download a model from HuggingFace Hub to a local directory.

    Args:
        repo_id: Repository on HuggingFace Hub to download from.
        local_dir: Local directory to save the model files.
        token: HuggingFace API token. Falls back to ``HF_TOKEN`` env var.
        revision: Git revision (branch, tag, or commit hash) to download.

    Returns:
        Path to the local directory containing the downloaded model.

    Raises:
        ImportError: If huggingface_hub is not installed.
    """
    HfApi, _, _ = _import_hub_deps()

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=token)
    snapshot_path: str = api.snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        revision=revision,
    )

    return Path(snapshot_path)


def load_metrics(metrics_path: str | Path) -> dict[str, float]:
    """Load evaluation metrics from a JSON file.

    Args:
        metrics_path: Path to a JSON file with metric key-value pairs.

    Returns:
        Dictionary of metric names to values.

    Raises:
        FileNotFoundError: If the metrics file does not exist.
    """
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with open(metrics_path, encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)

    return {k: float(v) for k, v in data.items()}
