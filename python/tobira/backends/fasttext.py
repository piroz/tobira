"""FastText backend implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tobira.backends.protocol import PredictionResult


def _import_fasttext() -> Any:
    """Lazily import fasttext."""
    try:
        import fasttext
    except ImportError as exc:
        raise ImportError(
            "fasttext is required for FastTextBackend. "
            "Install it with: pip install tobira[fasttext]"
        ) from exc
    return fasttext


class FastTextBackend:
    """Inference backend using a FastText model.

    Args:
        model_path: Path to the FastText .bin model file.
    """

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(f"model file not found: {self._model_path}")
        self._model = _import_fasttext().load_model(str(self._model_path))

    def predict(self, text: str) -> PredictionResult:
        """Run inference on the given text."""
        labels, scores = self._model.predict(text, k=-1)

        # FastText labels have a "__label__" prefix
        clean: dict[str, float] = {}
        for raw_label, score in zip(labels, scores):
            name = raw_label.replace("__label__", "")
            clean[name] = float(score)

        top_label = max(clean, key=clean.__getitem__)
        top_score = clean[top_label]

        return PredictionResult(label=top_label, score=top_score, labels=clean)
