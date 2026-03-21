"""BERT backend implementation using HuggingFace Transformers."""

from __future__ import annotations

from typing import Any

from tobira.backends.protocol import PredictionResult, TokenAttribution

_DEFAULT_MAX_EXPLANATIONS = 10


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

    def predict(
        self, text: str, *, explain: bool = False
    ) -> PredictionResult:
        """Run inference on the given text.

        Args:
            text: Input text to classify.
            explain: When True, compute token-level attention attributions.
        """
        torch, _, _ = _import_deps()

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs, output_attentions=explain)

        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0)
        id2label = self._model.config.id2label

        labels: dict[str, float] = {}
        for idx, prob in enumerate(probs):
            labels[id2label[idx]] = float(prob)

        top_idx = int(probs.argmax())
        top_label = id2label[top_idx]
        top_score = float(probs[top_idx])

        explanations: list[TokenAttribution] | None = None
        if explain and outputs.attentions:
            explanations = self._compute_attributions(
                inputs["input_ids"], outputs.attentions,
            )

        return PredictionResult(
            label=top_label, score=top_score, labels=labels,
            explanations=explanations,
        )

    def _compute_attributions(
        self,
        input_ids: Any,
        attentions: tuple[Any, ...],
    ) -> list[TokenAttribution]:
        """Compute token attributions from the last layer's attention to [CLS].

        Uses the last attention layer, averaging across all heads, and
        extracts each token's attention weight toward the [CLS] token.
        """
        # attentions[-1] shape: (batch, heads, seq_len, seq_len)
        last_layer = attentions[-1].squeeze(0)
        # Average across heads → (seq_len, seq_len)
        avg_attention = last_layer.mean(dim=0)
        # Attention from [CLS] to all tokens → (seq_len,)
        cls_attention = avg_attention[0]

        tokens: list[str] = self._tokenizer.convert_ids_to_tokens(
            input_ids.squeeze(0).tolist()
        )

        # Normalize to 0.0-1.0 range
        max_val = float(cls_attention.max())
        min_val = float(cls_attention.min())
        val_range = max_val - min_val if max_val > min_val else 1.0

        attributions: list[TokenAttribution] = []
        for i, token in enumerate(tokens):
            # Skip special tokens ([CLS], [SEP], [PAD])
            if token in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            score = (float(cls_attention[i]) - min_val) / val_range
            attributions.append(TokenAttribution(token=token, score=round(score, 4)))

        # Sort by score descending and take top N
        attributions.sort(key=lambda a: a.score, reverse=True)
        return attributions[:_DEFAULT_MAX_EXPLANATIONS]
