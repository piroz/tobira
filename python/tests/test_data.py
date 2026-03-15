"""Tests for tobira.data — category definitions and synthetic data generation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tobira.data.categories import (
    LANGUAGE_CATEGORIES,
    MULTICLASS_CATEGORIES,
    SPAM_CATEGORIES,
    SPAM_SUBCATEGORIES,
    Category,
    get_categories_for_language,
    get_category,
)
from tobira.data.generator import SyntheticSample, generate


class TestCategory:
    def test_frozen(self) -> None:
        cat = Category(name="spam", label="Spam", description="desc")
        with pytest.raises(AttributeError):
            cat.name = "other"  # type: ignore[misc]

    def test_fields(self) -> None:
        cat = Category(name="test", label="Test", description="A test category")
        assert cat.name == "test"
        assert cat.label == "Test"
        assert cat.description == "A test category"


class TestSPAMCategories:
    def test_has_spam_and_ham(self) -> None:
        names = {c.name for c in SPAM_CATEGORIES}
        assert "spam" in names
        assert "ham" in names

    def test_all_have_required_fields(self) -> None:
        for cat in SPAM_CATEGORIES:
            assert cat.name
            assert cat.label
            assert cat.description


class TestSpamSubcategories:
    def test_has_8_subcategories(self) -> None:
        assert len(SPAM_SUBCATEGORIES) == 8

    def test_expected_names(self) -> None:
        names = {c.name for c in SPAM_SUBCATEGORIES}
        expected = {
            "phishing", "malware", "financial_fraud", "lottery",
            "romance_scam", "drug", "fake_service", "tech_support",
        }
        assert names == expected

    def test_all_have_required_fields(self) -> None:
        for cat in SPAM_SUBCATEGORIES:
            assert cat.name
            assert cat.label
            assert cat.description


class TestMulticlassCategories:
    def test_includes_ham(self) -> None:
        names = {c.name for c in MULTICLASS_CATEGORIES}
        assert "ham" in names

    def test_includes_all_subcategories(self) -> None:
        mc_names = {c.name for c in MULTICLASS_CATEGORIES}
        sub_names = {c.name for c in SPAM_SUBCATEGORIES}
        assert sub_names.issubset(mc_names)

    def test_does_not_include_generic_spam(self) -> None:
        names = {c.name for c in MULTICLASS_CATEGORIES}
        assert "spam" not in names

    def test_count(self) -> None:
        assert len(MULTICLASS_CATEGORIES) == 9  # ham + 8 subcategories


class TestLanguageCategories:
    def test_has_core_languages(self) -> None:
        assert "ja" in LANGUAGE_CATEGORIES
        assert "en" in LANGUAGE_CATEGORIES
        assert "ko" in LANGUAGE_CATEGORIES

    def test_all_languages_have_spam_and_ham(self) -> None:
        for lang, cats in LANGUAGE_CATEGORIES.items():
            names = {c.name for c in cats}
            assert "spam" in names, f"{lang} missing spam"
            assert "ham" in names, f"{lang} missing ham"

    def test_en_is_default(self) -> None:
        assert LANGUAGE_CATEGORIES["en"] is SPAM_CATEGORIES


class TestGetCategoriesForLanguage:
    def test_known_language(self) -> None:
        cats = get_categories_for_language("ja")
        names = {c.name for c in cats}
        assert "spam" in names
        assert "ham" in names

    def test_unknown_language_falls_back(self) -> None:
        cats = get_categories_for_language("xx")
        assert cats is SPAM_CATEGORIES

    def test_en_returns_default(self) -> None:
        cats = get_categories_for_language("en")
        assert cats is SPAM_CATEGORIES


class TestGetCategory:
    def test_get_spam(self) -> None:
        cat = get_category("spam")
        assert cat.name == "spam"

    def test_get_ham(self) -> None:
        cat = get_category("ham")
        assert cat.name == "ham"

    def test_get_subcategory(self) -> None:
        cat = get_category("phishing")
        assert cat.name == "phishing"
        assert cat.label == "Phishing"

    def test_get_all_subcategories(self) -> None:
        for sub in SPAM_SUBCATEGORIES:
            cat = get_category(sub.name)
            assert cat.name == sub.name

    def test_unknown_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown category"):
            get_category("nonexistent")


class TestSyntheticSample:
    def test_frozen(self) -> None:
        sample = SyntheticSample(text="hello", category="spam")
        with pytest.raises(AttributeError):
            sample.text = "modified"  # type: ignore[misc]

    def test_fields(self) -> None:
        sample = SyntheticSample(text="hello", category="spam")
        assert sample.text == "hello"
        assert sample.category == "spam"


class TestGenerate:
    def _mock_response(self, texts: list[str]) -> MagicMock:
        """Create a mock httpx response returning *texts* as JSON."""
        content = json.dumps(texts)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": content}}],
        }
        return mock_resp

    def test_generates_samples(self) -> None:
        mock_client = MagicMock()
        mock_client.post.return_value = self._mock_response(
            ["spam message 1", "spam message 2"]
        )

        samples = generate("spam", 2, api_key="test-key", client=mock_client)
        assert len(samples) == 2
        assert all(isinstance(s, SyntheticSample) for s in samples)
        assert all(s.category == "spam" for s in samples)
        assert samples[0].text == "spam message 1"

    def test_sends_correct_request(self) -> None:
        mock_client = MagicMock()
        mock_client.post.return_value = self._mock_response(["msg"])

        generate(
            "ham",
            1,
            model="test-model",
            base_url="https://api.test.com/v1",
            api_key="key123",
            client=mock_client,
        )

        call_args = mock_client.post.call_args
        assert "chat/completions" in call_args[0][0]
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer key123"
        body = call_args[1]["json"]
        assert body["model"] == "test-model"

    def test_generates_subcategory_samples(self) -> None:
        mock_client = MagicMock()
        mock_client.post.return_value = self._mock_response(
            ["phishing message"]
        )

        samples = generate(
            "phishing", 1, api_key="test-key", client=mock_client
        )
        assert len(samples) == 1
        assert samples[0].category == "phishing"

        # Verify prompt includes subcategory label/description
        call_args = mock_client.post.call_args
        body = call_args[1]["json"]
        prompt = body["messages"][1]["content"]
        assert "Phishing" in prompt

    def test_unknown_category_raises(self) -> None:
        mock_client = MagicMock()
        with pytest.raises(KeyError, match="Unknown category"):
            generate("nonexistent", 1, api_key="key", client=mock_client)

    def test_zero_count_raises(self) -> None:
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="count must be positive"):
            generate("spam", 0, api_key="key", client=mock_client)

    def test_negative_count_raises(self) -> None:
        mock_client = MagicMock()
        with pytest.raises(ValueError, match="count must be positive"):
            generate("spam", -1, api_key="key", client=mock_client)

    def test_handles_dict_wrapper_response(self) -> None:
        """LLM might return {"messages": [...]} instead of bare array."""
        mock_client = MagicMock()
        content = json.dumps({"messages": ["wrapped msg"]})
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": content}}],
        }
        mock_client.post.return_value = mock_resp

        samples = generate("spam", 1, api_key="key", client=mock_client)
        assert len(samples) == 1
        assert samples[0].text == "wrapped msg"

    def test_generate_with_language(self) -> None:
        mock_client = MagicMock()
        mock_client.post.return_value = self._mock_response(
            ["スパムメッセージ"]
        )

        samples = generate(
            "spam", 1, api_key="key", client=mock_client, language="ja"
        )
        assert len(samples) == 1
        assert samples[0].category == "spam"

        # Verify prompt includes language instruction
        call_args = mock_client.post.call_args
        body = call_args[1]["json"]
        user_msg = body["messages"][1]["content"]
        assert "Japanese" in user_msg

    def test_generate_with_unknown_language_uses_default(self) -> None:
        mock_client = MagicMock()
        mock_client.post.return_value = self._mock_response(["msg"])

        samples = generate(
            "spam", 1, api_key="key", client=mock_client, language="xx"
        )
        assert len(samples) == 1

    def test_import_error_without_httpx(self) -> None:
        from tobira.data.generator import _import_httpx

        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx is required"):
                _import_httpx()


class TestDataPublicAPI:
    def test_imports_from_package(self) -> None:
        from tobira.data import (
            LANGUAGE_CATEGORIES,
            MULTICLASS_CATEGORIES,
            SPAM_CATEGORIES,
            SPAM_SUBCATEGORIES,
            Category,
            SyntheticSample,
            generate,
            get_categories_for_language,
            get_category,
        )

        assert LANGUAGE_CATEGORIES is not None
        assert MULTICLASS_CATEGORIES is not None
        assert SPAM_CATEGORIES is not None
        assert SPAM_SUBCATEGORIES is not None
        assert Category is not None
        assert SyntheticSample is not None
        assert generate is not None
        assert get_categories_for_language is not None
        assert get_category is not None
