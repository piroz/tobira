"""ONNX export and INT8 dynamic quantization utilities."""

from __future__ import annotations

from pathlib import Path


def _import_export_deps() -> tuple:
    """Lazily import torch, transformers, and optimum."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "torch and transformers are required for ONNX export. "
            "Install them with: pip install tobira[onnx]"
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


def _import_quantization_deps() -> tuple:
    """Lazily import onnxruntime quantization utilities."""
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for quantization. "
            "Install it with: pip install tobira[onnx]"
        ) from exc
    return quantize_dynamic, QuantType


def export_onnx(
    model_name: str,
    output_path: str | Path,
    opset_version: int = 14,
) -> Path:
    """Export a BERT sequence-classification model to ONNX format.

    Args:
        model_name: HuggingFace model name or local path.
        output_path: File path for the exported ONNX model.
        opset_version: ONNX opset version. Defaults to 14.

    Returns:
        Path to the exported ONNX model file.

    Raises:
        ImportError: If torch or transformers are not installed.
    """
    torch, AutoModelForSequenceClassification, AutoTokenizer = _import_export_deps()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    dummy_input = tokenizer("dummy", return_tensors="pt")

    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "output": {0: "batch"},
    }

    input_names = ["input_ids", "attention_mask"]
    if "token_type_ids" in dummy_input:
        input_names.append("token_type_ids")
        dynamic_axes["token_type_ids"] = {0: "batch", 1: "seq"}

    torch.onnx.export(
        model,
        tuple(dummy_input[k] for k in input_names),
        str(output_path),
        input_names=input_names,
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
    )

    return output_path


def quantize_dynamic(
    input_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Apply INT8 dynamic quantization to an ONNX model.

    Args:
        input_path: Path to the source ONNX model.
        output_path: Path for the quantized model. When *None*, appends
            ``_quantized`` to the input filename.

    Returns:
        Path to the quantized ONNX model file.

    Raises:
        ImportError: If onnxruntime is not installed.
        FileNotFoundError: If *input_path* does not exist.
    """
    _qd, QuantType = _import_quantization_deps()

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {input_path}")

    if output_path is None:
        output_path = input_path.with_name(
            input_path.stem + "_quantized" + input_path.suffix
        )
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    _qd(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    return output_path
