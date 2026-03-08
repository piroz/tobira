"""BERT backend implementation using HuggingFace Transformers."""

from __future__ import annotations

from tobira.backends.protocol import PredictionResult


def _import_deps() -> tuple:
    """Lazily import torch and transformers."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "torch and transformers are required for BertBackend. "
            "Install them with: pip install tobira[bert]"
        ) from exc
    return torch, AutoModelForSequenceClassification, AutoTokenizer


class BertBackend:
    """Inference backend using a BERT sequence-classification model.

    Args:
        model_name: HuggingFace model name or local path.
            Defaults to ``tohoku-nlp/bert-base-japanese-v3``.
        device: Device string (e.g. ``"cpu"``, ``"cuda"``).
            When *None*, automatically selects CUDA if available.
    """

    def __init__(
        self,
        model_name: str = "tohoku-nlp/bert-base-japanese-v3",
        device: str | None = None,
    ) -> None:
        torch, AutoModelForSequenceClassification, AutoTokenizer = _import_deps()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()

    def predict(self, text: str) -> PredictionResult:
        """Run inference on the given text."""
        torch, _, _ = _import_deps()

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self._model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze(0)
        id2label = self._model.config.id2label

        labels: dict[str, float] = {}
        for idx, prob in enumerate(probs):
            labels[id2label[idx]] = float(prob)

        top_idx = int(probs.argmax())
        top_label = id2label[top_idx]
        top_score = float(probs[top_idx])

        return PredictionResult(label=top_label, score=top_score, labels=labels)
