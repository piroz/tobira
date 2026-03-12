"""ONNX Runtime backend implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tobira.backends.protocol import PredictionResult


def _import_deps() -> tuple:
    """Lazily import onnxruntime, numpy, and transformers."""
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for OnnxBackend. "
            "Install it with: pip install tobira[onnx]"
        ) from exc
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for OnnxBackend tokenizer. "
            "Install it with: pip install tobira[onnx]"
        ) from exc
    return ort, np, AutoConfig, AutoTokenizer


def _softmax(x: Any, np: Any) -> Any:
    """Compute softmax over the last axis."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class OnnxBackend:
    """Inference backend using ONNX Runtime.

    Args:
        model_path: Path to the ONNX model file.
        model_name: HuggingFace model name for tokenizer and label config.
            Defaults to ``tohoku-nlp/bert-base-japanese-v3``.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "tohoku-nlp/bert-base-japanese-v3",
    ) -> None:
        ort, np, AutoConfig, AutoTokenizer = _import_deps()

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model file not found: {model_path}")

        self._np = np
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self._id2label: dict[int, str] = config.id2label

    def predict(self, text: str) -> PredictionResult:
        """Run inference on the given text."""
        np = self._np

        inputs = self._tokenizer(
            text, return_tensors="np", truncation=True, max_length=512
        )

        input_names = [inp.name for inp in self._session.get_inputs()]
        feed = {k: v for k, v in inputs.items() if k in input_names}

        logits = self._session.run(None, feed)[0]
        probs = _softmax(logits, np).squeeze(0)

        labels: dict[str, float] = {}
        for idx, prob in enumerate(probs):
            labels[self._id2label[idx]] = float(prob)

        top_idx = int(np.argmax(probs))
        top_label = self._id2label[top_idx]
        top_score = float(probs[top_idx])

        return PredictionResult(label=top_label, score=top_score, labels=labels)
