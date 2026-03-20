"""Ollama backend implementation."""

from __future__ import annotations

import json
from typing import Any

from tobira.backends.prompts import (
    SPAM_CLASSIFICATION_SYSTEM,
    FewShotExample,
    build_few_shot_system_prompt,
    build_prompt,
    parse_llm_result,
)
from tobira.backends.protocol import PredictionResult


def _import_httpx() -> Any:
    """Lazily import httpx."""
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for OllamaBackend. "
            "Install it with: pip install tobira[llm]"
        ) from exc
    return httpx


class OllamaBackend:
    """Inference backend using Ollama REST API.

    Args:
        model: Ollama model name (e.g. ``gemma2:2b``).
        base_url: Ollama API base URL.
        timeout: Request timeout in seconds.
        few_shot_examples: Optional list of few-shot examples. When provided,
            examples are included in the system prompt to improve classification
            accuracy.
    """

    def __init__(
        self,
        model: str = "gemma2:2b",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        few_shot_examples: list[FewShotExample] | None = None,
    ) -> None:
        httpx = _import_httpx()
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Any = httpx.Client(timeout=timeout)
        self._few_shot_examples = few_shot_examples

    def predict(self, text: str) -> PredictionResult:
        """Run inference on the given text via Ollama API."""
        if self._few_shot_examples:
            system = build_few_shot_system_prompt(self._few_shot_examples)
        else:
            system = SPAM_CLASSIFICATION_SYSTEM

        response = self._client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self._model,
                "system": system,
                "prompt": build_prompt(text),
                "format": "json",
                "stream": False,
            },
        )
        response.raise_for_status()

        data = response.json()
        raw = json.loads(data["response"])

        return parse_llm_result(raw)
