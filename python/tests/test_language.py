"""Tests for tobira.preprocessing.language — language detection."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tobira.preprocessing.language import (
    SUPPORTED_LANGUAGES,
    LanguageResult,
    detect_language,
    detect_languages,
    is_supported_language,
)


class TestLanguageResult:
    def test_frozen(self) -> None:
        result = LanguageResult(language="en", confidence=0.99)
        with pytest.raises(AttributeError):
            result.language = "ja"  # type: ignore[misc]

    def test_fields(self) -> None:
        result = LanguageResult(language="ja", confidence=0.85)
        assert result.language == "ja"
        assert result.confidence == 0.85


class TestSupportedLanguages:
    def test_contains_core_languages(self) -> None:
        assert "ja" in SUPPORTED_LANGUAGES
        assert "en" in SUPPORTED_LANGUAGES
        assert "ko" in SUPPORTED_LANGUAGES

    def test_is_tuple(self) -> None:
        assert isinstance(SUPPORTED_LANGUAGES, tuple)


class TestIsSupportedLanguage:
    def test_supported(self) -> None:
        assert is_supported_language("ja") is True
        assert is_supported_language("en") is True
        assert is_supported_language("ko") is True

    def test_unsupported(self) -> None:
        assert is_supported_language("xx") is False
        assert is_supported_language("") is False


class TestDetectLanguage:
    def test_detects_english(self) -> None:
        result = detect_language(
            "This is a test email about a business proposal"
        )
        assert isinstance(result, LanguageResult)
        assert result.language == "en"
        assert 0.0 <= result.confidence <= 1.0

    def test_detects_japanese(self) -> None:
        result = detect_language("これはテストメールです。ビジネスの提案について")
        assert result.language == "ja"

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="empty text"):
            detect_language("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty text"):
            detect_language("   ")

    def test_import_error_without_langdetect(self) -> None:
        from tobira.preprocessing.language import _import_langdetect

        with patch.dict("sys.modules", {"langdetect": None}):
            with pytest.raises(ImportError, match="langdetect is required"):
                _import_langdetect()


class TestDetectLanguages:
    def test_returns_list(self) -> None:
        results = detect_languages(
            "This is a test message in English"
        )
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(r, LanguageResult) for r in results)

    def test_top_n_limits_results(self) -> None:
        results = detect_languages(
            "This is a test message in English", top_n=1
        )
        assert len(results) <= 1

    def test_sorted_by_confidence(self) -> None:
        results = detect_languages(
            "This is a test message in English"
        )
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValueError, match="empty text"):
            detect_languages("")


class TestPreprocessingLanguagePublicAPI:
    def test_imports_from_package(self) -> None:
        from tobira.preprocessing import (
            SUPPORTED_LANGUAGES,
            LanguageResult,
            detect_language,
            detect_languages,
            is_supported_language,
        )

        assert SUPPORTED_LANGUAGES is not None
        assert LanguageResult is not None
        assert detect_language is not None
        assert detect_languages is not None
        assert is_supported_language is not None
