"""Tests for tobira.data — category definitions and synthetic data generation."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tobira.data.categories import (
    SPAM_CATEGORIES,
    Category,
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


class TestGetCategory:
    def test_get_spam(self) -> None:
        cat = get_category("spam")
        assert cat.name == "spam"

    def test_get_ham(self) -> None:
        cat = get_category("ham")
        assert cat.name == "ham"

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

    def test_import_error_without_httpx(self) -> None:
        from tobira.data.generator import _import_httpx

        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises(ImportError, match="httpx is required"):
                _import_httpx()


class TestDataPublicAPI:
    def test_imports_from_package(self) -> None:
        from tobira.data import (
            SPAM_CATEGORIES,
            Category,
            SyntheticSample,
            generate,
            get_category,
        )

        assert SPAM_CATEGORIES is not None
        assert Category is not None
        assert SyntheticSample is not None
        assert generate is not None
        assert get_category is not None
