"""OpenAI-compatible LLM API backend implementation."""

from __future__ import annotations

import json
import os
from typing import Any

from tobira.backends.prompts import (
    SPAM_CLASSIFICATION_SYSTEM,
    FewShotExample,
    build_few_shot_messages,
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
            "httpx is required for LlmApiBackend. "
            "Install it with: pip install tobira[llm]"
        ) from exc
    return httpx


class LlmApiBackend:
    """Inference backend using OpenAI-compatible chat completions API.

    Args:
        model: Model name to use (e.g. ``gpt-4o-mini``).
        base_url: API base URL (must include ``/v1`` prefix if needed).
        api_key: API key. Defaults to ``TOBIRA_LLM_API_KEY`` environment variable.
        timeout: Request timeout in seconds.
        few_shot_examples: Optional list of few-shot examples. When provided,
            examples are included as user/assistant message pairs to improve
            classification accuracy.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        timeout: float = 30.0,
        few_shot_examples: list[FewShotExample] | None = None,
    ) -> None:
        httpx = _import_httpx()
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ.get("TOBIRA_LLM_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "API key is required. Set TOBIRA_LLM_API_KEY environment variable "
                "or pass api_key parameter."
            )
        self._timeout = timeout
        self._client: Any = httpx.Client(timeout=timeout)
        self._few_shot_examples = few_shot_examples

    def predict(self, text: str) -> PredictionResult:
        """Run inference on the given text via chat completions API."""
        if self._few_shot_examples:
            messages = build_few_shot_messages(self._few_shot_examples, text)
        else:
            messages = [
                {"role": "system", "content": SPAM_CLASSIFICATION_SYSTEM},
                {"role": "user", "content": build_prompt(text)},
            ]

        response = self._client.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": messages,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]
        raw = json.loads(content)

        return parse_llm_result(raw)
