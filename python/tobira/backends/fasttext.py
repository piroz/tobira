"""FastText backend implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tobira.backends.protocol import PredictionResult

_numpy_patched = False


def _patch_numpy_copy() -> None:
    """Patch numpy to work around fasttext's use of np.array(copy=False).

    fasttext-wheel<=0.9.2 calls ``np.array(probs, copy=False)`` which raises
    ``ValueError`` on NumPy 2.x due to stricter ``copy`` semantics.  We wrap
    ``np.array`` so that a failing ``copy=False`` falls back to ``np.asarray``.
    """
    global _numpy_patched  # noqa: PLW0603
    if _numpy_patched:
        return
    _numpy_patched = True

    import numpy as np

    if np.lib.NumpyVersion(np.__version__) < "2.0.0":
        return

    _original_array = np.array

    def _array_compat(*args: Any, **kwargs: Any) -> Any:
        try:
            return _original_array(*args, **kwargs)
        except ValueError:
            if kwargs.get("copy") is False:
                kwargs.pop("copy")
                return np.asarray(*args, **kwargs)
            raise

    np.array = _array_compat


def _import_fasttext() -> Any:
    """Lazily import fasttext."""
    try:
        import fasttext
    except ImportError as exc:
        raise ImportError(
            "fasttext is required for FastTextBackend. "
            "Install it with: pip install tobira[fasttext]"
        ) from exc
    _patch_numpy_copy()
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
