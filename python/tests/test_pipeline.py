"""Tests for tobira.preprocessing.pipeline — preprocessing pipeline."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tobira.preprocessing.pipeline import (
    PipelineConfig,
    PreprocessingPipeline,
    PreprocessingResult,
    load_pipeline_config,
)


class TestPreprocessingResult:
    def test_immutable(self) -> None:
        result = PreprocessingResult(text="hello")
        with pytest.raises(AttributeError):
            result.text = "world"  # type: ignore[misc]

    def test_defaults(self) -> None:
        result = PreprocessingResult(text="hello")
        assert result.language is None
        assert result.anonymize_result is None
        assert result.header_features is None
        assert result.header_score is None
        assert result.metadata == {}


class TestPipelineConfig:
    def test_defaults(self) -> None:
        config = PipelineConfig()
        assert config.language_detection is True
        assert config.anonymization is True
        assert config.header_analysis is True
        assert config.anonymization_use_ner is False
        assert config.anonymization_pii_types is None
        assert config.header_weights is None

    def test_immutable(self) -> None:
        config = PipelineConfig()
        with pytest.raises(AttributeError):
            config.language_detection = False  # type: ignore[misc]


class TestLoadPipelineConfig:
    def test_empty_config(self) -> None:
        config = load_pipeline_config({})
        assert config.language_detection is True
        assert config.anonymization is True
        assert config.header_analysis is True

    def test_disable_steps(self) -> None:
        config = load_pipeline_config(
            {
                "steps": {
                    "language_detection": False,
                    "anonymization": False,
                    "header_analysis": False,
                }
            }
        )
        assert config.language_detection is False
        assert config.anonymization is False
        assert config.header_analysis is False

    def test_anonymization_options(self) -> None:
        config = load_pipeline_config(
            {
                "anonymization": {
                    "use_ner": True,
                    "pii_types": ["email", "phone"],
                }
            }
        )
        assert config.anonymization_use_ner is True
        assert list(config.anonymization_pii_types or []) == ["email", "phone"]

    def test_header_weights(self) -> None:
        config = load_pipeline_config(
            {"headers": {"weights": {"spf": 0.5, "dkim": 0.5}}}
        )
        assert config.header_weights == {"spf": 0.5, "dkim": 0.5}


class TestPreprocessingPipelineDefaults:
    def test_default_config(self) -> None:
        pipeline = PreprocessingPipeline()
        assert pipeline.config == PipelineConfig()

    def test_custom_config(self) -> None:
        config = PipelineConfig(anonymization=False)
        pipeline = PreprocessingPipeline(config)
        assert pipeline.config.anonymization is False


class TestPipelineRunAllSteps:
    def test_basic_run(self) -> None:
        pipeline = PreprocessingPipeline()
        result = pipeline.run("Hello world from user@example.com")

        assert isinstance(result, PreprocessingResult)
        # Anonymization should replace the email
        assert "user@example.com" not in result.text
        assert "<EMAIL>" in result.text
        assert result.anonymize_result is not None
        assert len(result.anonymize_result.entities) >= 1
        # Language should be detected
        assert result.language is not None
        assert result.language.language in ("en", "so")  # short text may vary
        # No headers provided so header analysis is skipped
        assert result.header_features is None
        assert result.header_score is None

    def test_with_headers(self) -> None:
        headers: dict[str, object] = {
            "spf": "pass",
            "dkim": "pass",
            "dmarc": "pass",
        }
        pipeline = PreprocessingPipeline()
        result = pipeline.run("Test email body", headers=headers)

        assert result.header_features is not None
        assert result.header_score is not None
        assert result.header_score == pytest.approx(0.0)
        assert result.metadata.get("header_score") is not None

    def test_pii_metadata(self) -> None:
        pipeline = PreprocessingPipeline()
        result = pipeline.run(
            "Contact user@example.com or call 090-1234-5678"
        )
        assert result.metadata.get("pii_count", 0) >= 2
        kinds = result.metadata.get("pii_kinds", {})
        assert isinstance(kinds, dict)
        assert "email" in kinds


class TestPipelineDisabledSteps:
    def test_disable_language_detection(self) -> None:
        config = PipelineConfig(language_detection=False)
        pipeline = PreprocessingPipeline(config)
        result = pipeline.run("Hello world")

        assert result.language is None
        assert "language" not in result.metadata

    def test_disable_anonymization(self) -> None:
        config = PipelineConfig(anonymization=False)
        pipeline = PreprocessingPipeline(config)
        result = pipeline.run("Email: user@example.com")

        assert result.anonymize_result is None
        assert result.text == "Email: user@example.com"

    def test_disable_header_analysis(self) -> None:
        headers: dict[str, object] = {"spf": "fail"}
        config = PipelineConfig(header_analysis=False)
        pipeline = PreprocessingPipeline(config)
        result = pipeline.run("Test", headers=headers)

        assert result.header_features is None
        assert result.header_score is None

    def test_all_disabled(self) -> None:
        config = PipelineConfig(
            language_detection=False,
            anonymization=False,
            header_analysis=False,
        )
        pipeline = PreprocessingPipeline(config)
        result = pipeline.run("Hello user@example.com")

        assert result.text == "Hello user@example.com"
        assert result.language is None
        assert result.anonymize_result is None
        assert result.header_features is None


class TestPipelineFailOpen:
    def test_language_detection_failure_skips(self) -> None:
        pipeline = PreprocessingPipeline()
        # Empty text causes ValueError in language detection
        result = pipeline.run("")

        assert result.language is None
        assert result.metadata.get("language_error") is True
        # Other steps should still run
        assert result.anonymize_result is not None

    def test_anonymization_failure_skips(self) -> None:
        config = PipelineConfig(anonymization=True)
        pipeline = PreprocessingPipeline(config)
        with patch(
            "tobira.preprocessing.pipeline.anonymize",
            side_effect=RuntimeError("boom"),
        ):
            result = pipeline.run("some text")

        assert result.anonymize_result is None
        assert result.metadata.get("anonymization_error") is True
        assert result.text == "some text"

    def test_header_analysis_failure_skips(self) -> None:
        pipeline = PreprocessingPipeline()
        with patch(
            "tobira.preprocessing.pipeline.extract_features",
            side_effect=RuntimeError("boom"),
        ):
            result = pipeline.run("test", headers={"spf": "pass"})

        assert result.header_features is None
        assert result.header_score is None
        assert result.metadata.get("header_analysis_error") is True


class TestPipelineAnonymizationOptions:
    def test_pii_type_filter(self) -> None:
        config = PipelineConfig(
            language_detection=False,
            anonymization_pii_types=["email"],
        )
        pipeline = PreprocessingPipeline(config)
        result = pipeline.run("user@example.com 090-1234-5678")

        assert result.anonymize_result is not None
        # Email should be anonymized but phone kept
        assert "<EMAIL>" in result.text
        kinds = result.metadata.get("pii_kinds", {})
        assert isinstance(kinds, dict)
        assert "email" in kinds
        assert "phone" not in kinds


class TestPipelineExports:
    def test_public_exports(self) -> None:
        from tobira.preprocessing import (
            PipelineConfig,
            PreprocessingPipeline,
            PreprocessingResult,
            load_pipeline_config,
        )

        assert PipelineConfig is not None
        assert PreprocessingPipeline is not None
        assert PreprocessingResult is not None
        assert load_pipeline_config is not None
