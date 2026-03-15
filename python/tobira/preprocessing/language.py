"""Language detection for multilingual spam classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _import_langdetect() -> Any:
    """Lazily import langdetect."""
    try:
        import langdetect
    except ImportError as exc:
        raise ImportError(
            "langdetect is required for language detection. "
            "Install it with: pip install tobira[multilingual]"
        ) from exc
    return langdetect


@dataclass(frozen=True)
class LanguageResult:
    """Result of language detection.

    Attributes:
        language: Detected ISO 639-1 language code (e.g. ``ja``, ``en``, ``ko``).
        confidence: Detection confidence score (0.0 to 1.0).
    """

    language: str
    confidence: float


# Languages supported by tobira's multilingual features.
SUPPORTED_LANGUAGES: tuple[str, ...] = ("ja", "en", "ko", "zh-cn", "zh-tw")


def detect_language(text: str) -> LanguageResult:
    """Detect the language of the given text.

    Uses the ``langdetect`` library for detection. Returns the most likely
    language as an ISO 639-1 code.

    Args:
        text: Input text to analyse.

    Returns:
        A :class:`LanguageResult` with the detected language and confidence.

    Raises:
        ValueError: If the text is empty or language cannot be determined.
    """
    if not text.strip():
        raise ValueError("cannot detect language of empty text")

    langdetect = _import_langdetect()

    try:
        results = langdetect.detect_langs(text)
    except langdetect.lang_detect_exception.LangDetectException as exc:
        raise ValueError(f"language detection failed: {exc}") from exc

    if not results:
        raise ValueError("language detection returned no results")

    top = results[0]
    return LanguageResult(language=top.lang, confidence=top.prob)


def detect_languages(text: str, *, top_n: int = 3) -> list[LanguageResult]:
    """Detect multiple candidate languages for the given text.

    Args:
        text: Input text to analyse.
        top_n: Maximum number of results to return.

    Returns:
        A list of :class:`LanguageResult` sorted by confidence (descending).

    Raises:
        ValueError: If the text is empty or detection fails.
    """
    if not text.strip():
        raise ValueError("cannot detect language of empty text")

    langdetect = _import_langdetect()

    try:
        results = langdetect.detect_langs(text)
    except langdetect.lang_detect_exception.LangDetectException as exc:
        raise ValueError(f"language detection failed: {exc}") from exc

    return [
        LanguageResult(language=r.lang, confidence=r.prob)
        for r in results[:top_n]
    ]


def is_supported_language(language: str) -> bool:
    """Check whether a language code is in the supported set.

    Args:
        language: ISO 639-1 language code.

    Returns:
        ``True`` if the language is supported.
    """
    return language in SUPPORTED_LANGUAGES
