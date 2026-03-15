"""Synthetic data generation pipeline using LLM APIs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from tobira.data.categories import (
    Category,
    get_categories_for_language,
    get_category,
)

# Mapping from language code to the instruction language name used in prompts.
_LANGUAGE_NAMES: dict[str, str] = {
    "ja": "Japanese",
    "en": "English",
    "ko": "Korean",
    "zh-cn": "Simplified Chinese",
    "zh-tw": "Traditional Chinese",
}


def _import_httpx() -> Any:
    """Lazily import httpx."""
    try:
        import httpx
    except ImportError as exc:
        raise ImportError(
            "httpx is required for synthetic data generation. "
            "Install it with: pip install tobira[llm]"
        ) from exc
    return httpx


@dataclass(frozen=True)
class SyntheticSample:
    """A single synthetic data sample.

    Attributes:
        text: Generated text content.
        category: Category name the sample belongs to.
    """

    text: str
    category: str


def _build_prompt(
    category: Category, count: int, *, language: str | None = None
) -> str:
    """Build a prompt asking the LLM to generate samples."""
    lang_instruction = ""
    if language is not None:
        lang_name = _LANGUAGE_NAMES.get(language, language)
        lang_instruction = f" Write the messages in {lang_name}."

    return (
        f"Generate {count} realistic example messages for the category "
        f'"{category.label}" ({category.description}).{lang_instruction}\n'
        "Return a JSON array of strings, each being one example message.\n"
        "Return ONLY the JSON array, no other text."
    )


def generate(
    category: str,
    count: int,
    *,
    model: str = "gpt-4o-mini",
    base_url: str = "https://api.openai.com/v1",
    api_key: str = "",
    client: Any = None,
    language: str | None = None,
) -> list[SyntheticSample]:
    """Generate synthetic samples for a *category* using an LLM API.

    Args:
        category: Category name (must exist in the category catalogue).
        count: Number of samples to generate.
        model: LLM model name.
        base_url: API base URL.
        api_key: API key for the LLM service.
        client: Optional pre-configured httpx ``Client``.
        language: ISO 639-1 language code for the generated text.
            When set, the LLM is instructed to generate messages in the
            specified language and language-specific category descriptions
            are used.  Defaults to ``None`` (English / no constraint).

    Returns:
        A list of :class:`SyntheticSample` instances.

    Raises:
        KeyError: If *category* is not a known category.
        ValueError: If *count* is not positive.
    """
    if count <= 0:
        raise ValueError("count must be positive")

    # Look up category: use language-specific definitions when language is
    # specified, otherwise search binary + multiclass catalogues.
    if language:
        categories = get_categories_for_language(language)
        cat: Category | None = None
        for c in categories:
            if c.name == category:
                cat = c
                break
        if cat is None:
            raise KeyError(f"Unknown category: {category!r}")
    else:
        cat = get_category(category)

    prompt = _build_prompt(cat, count, language=language)

    httpx = _import_httpx()
    if client is None:
        client = httpx.Client(timeout=60.0)

    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data generation assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        },
    )
    response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    texts = json.loads(content)

    # Handle both bare array and {"messages": [...]} wrapper
    if isinstance(texts, dict):
        for key in ("messages", "data", "examples", "samples"):
            if key in texts:
                texts = texts[key]
                break

    if not isinstance(texts, list):
        raise ValueError(f"Expected list from LLM, got {type(texts).__name__}")

    return [SyntheticSample(text=str(t), category=category) for t in texts]
