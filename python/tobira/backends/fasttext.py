"""FastText backend implementation."""

from __future__ import annotations

from pathlib import Path

import fasttext

from tobira.backends.protocol import PredictionResult


class FastTextBackend:
    """Inference backend using a FastText model.

    Args:
        model_path: Path to the FastText .bin model file.
    """

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)
        self._model = fasttext.load_model(str(self._model_path))

    def predict(self, text: str) -> PredictionResult:
        """Run inference on the given text."""
        labels, scores = self._model.predict(text, k=-1)

        # FastText labels have a "__label__" prefix
        clean: dict[str, float] = {}
        for raw_label, score in zip(labels, scores):
            name = raw_label.replace("__label__", "")
            clean[name] = float(score)

        top_label = next(iter(clean))
        top_score = clean[top_label]

        return PredictionResult(label=top_label, score=top_score, labels=clean)
