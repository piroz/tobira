"""Tests for tobira.models."""

from __future__ import annotations

from tobira.models import (
    DEFAULT_MODEL,
    MODEL_CATALOG,
    RECOMMENDED_MODEL,
    RECOMMENDED_MODEL_JAPANESE,
    ModelInfo,
)


class TestModelConstants:
    def test_recommended_model(self) -> None:
        assert RECOMMENDED_MODEL == "microsoft/mdeberta-v3-base"

    def test_recommended_model_japanese(self) -> None:
        assert RECOMMENDED_MODEL_JAPANESE == "ku-nlp/deberta-v3-base-japanese"

    def test_default_model(self) -> None:
        assert DEFAULT_MODEL == "tohoku-nlp/bert-base-japanese-v3"


class TestModelCatalog:
    def test_catalog_contains_all_recommended(self) -> None:
        names = {info.name for info in MODEL_CATALOG.values()}
        assert RECOMMENDED_MODEL in names
        assert RECOMMENDED_MODEL_JAPANESE in names
        assert DEFAULT_MODEL in names

    def test_model_info_frozen(self) -> None:
        info = MODEL_CATALOG["mdeberta-v3-base"]
        assert isinstance(info, ModelInfo)
        assert info.name == RECOMMENDED_MODEL

    def test_catalog_entries_have_required_fields(self) -> None:
        for key, info in MODEL_CATALOG.items():
            assert info.name, f"{key} missing name"
            assert info.description, f"{key} missing description"
            assert info.parameters, f"{key} missing parameters"
            assert info.languages, f"{key} missing languages"
