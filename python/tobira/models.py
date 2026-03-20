"""Recommended model definitions and metadata."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Metadata for a recommended model.

    Attributes:
        name: HuggingFace model identifier.
        description: Human-readable description.
        parameters: Approximate parameter count.
        languages: Supported languages.
    """

    name: str
    description: str
    parameters: str
    languages: list[str]


#: Recommended base model for new deployments.
#: DeBERTa-v3 uses disentangled attention and enhanced mask decoder,
#: yielding higher classification accuracy than BERT at similar size.
RECOMMENDED_MODEL = "microsoft/mdeberta-v3-base"

#: Recommended model for Japanese-only deployments.
RECOMMENDED_MODEL_JAPANESE = "ku-nlp/deberta-v3-base-japanese"

#: Legacy default model (kept for backward compatibility).
DEFAULT_MODEL = "tohoku-nlp/bert-base-japanese-v3"

#: Catalog of recommended models for different use cases.
MODEL_CATALOG: dict[str, ModelInfo] = {
    "mdeberta-v3-base": ModelInfo(
        name="microsoft/mdeberta-v3-base",
        description="Multilingual DeBERTa-v3 (recommended for new deployments)",
        parameters="86M",
        languages=["multilingual"],
    ),
    "deberta-v3-base-japanese": ModelInfo(
        name="ku-nlp/deberta-v3-base-japanese",
        description="Japanese-specialized DeBERTa-v3",
        parameters="86M",
        languages=["ja"],
    ),
    "bert-base-japanese-v3": ModelInfo(
        name="tohoku-nlp/bert-base-japanese-v3",
        description="Legacy BERT model (current default for backward compatibility)",
        parameters="111M",
        languages=["ja"],
    ),
}
