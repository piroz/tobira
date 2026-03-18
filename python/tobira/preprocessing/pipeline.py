"""Preprocessing pipeline for email text.

Orchestrates PII anonymization, header analysis, and language detection
into a single configurable pipeline.  Each step can be enabled or disabled
via configuration; the execution order is fixed to ensure correctness
(language detection → anonymization → header analysis).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from tobira.preprocessing.anonymizer import AnonymizeResult, anonymize
from tobira.preprocessing.headers import (
    HeaderFeatures,
    analyze_headers,
    extract_features,
)
from tobira.preprocessing.language import LanguageResult, detect_language

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingResult:
    """Result of the full preprocessing pipeline.

    Attributes:
        text: The processed text (after anonymization if enabled).
        language: Detected language, or ``None`` if language detection is
            disabled or failed.
        anonymize_result: Anonymization result, or ``None`` if anonymization
            is disabled.
        header_features: Extracted header features, or ``None`` if header
            analysis is disabled or no headers were provided.
        header_score: Header-based spam risk score (0.0–1.0), or ``None``
            if header analysis is disabled.
        metadata: Additional metadata collected during processing.
    """

    text: str
    language: LanguageResult | None = None
    anonymize_result: AnonymizeResult | None = None
    header_features: HeaderFeatures | None = None
    header_score: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the preprocessing pipeline.

    Attributes:
        language_detection: Enable language detection step.
        anonymization: Enable PII anonymization step.
        header_analysis: Enable header risk analysis step.
        anonymization_use_ner: Use GiNZA NER for person-name detection
            during anonymization.
        anonymization_pii_types: Restrict anonymization to these PII types.
            ``None`` means anonymize all detected types.
        header_weights: Custom weights for header risk scoring.
            ``None`` uses the default weights.
    """

    language_detection: bool = True
    anonymization: bool = True
    header_analysis: bool = True
    anonymization_use_ner: bool = False
    anonymization_pii_types: Sequence[str] | None = None
    header_weights: dict[str, float] | None = None


def load_pipeline_config(config: dict[str, Any]) -> PipelineConfig:
    """Build a :class:`PipelineConfig` from a parsed TOML dict.

    Expects the ``[preprocessing]`` section of a ``tobira.toml`` file.
    Missing keys fall back to defaults.

    Args:
        config: The ``preprocessing`` section of the TOML configuration.

    Returns:
        A :class:`PipelineConfig` instance.
    """
    steps = config.get("steps", {})
    anon_cfg = config.get("anonymization", {})
    header_cfg = config.get("headers", {})

    pii_types_raw = anon_cfg.get("pii_types")
    pii_types: Sequence[str] | None = (
        list(pii_types_raw) if pii_types_raw is not None else None
    )

    weights_raw = header_cfg.get("weights")
    weights: dict[str, float] | None = (
        dict(weights_raw) if weights_raw is not None else None
    )

    return PipelineConfig(
        language_detection=bool(steps.get("language_detection", True)),
        anonymization=bool(steps.get("anonymization", True)),
        header_analysis=bool(steps.get("header_analysis", True)),
        anonymization_use_ner=bool(anon_cfg.get("use_ner", False)),
        anonymization_pii_types=pii_types,
        header_weights=weights,
    )


class PreprocessingPipeline:
    """Configurable preprocessing pipeline for email text.

    Steps are executed in a fixed order:

    1. **Language detection** — detect the primary language of the text.
    2. **Anonymization** — replace PII with placeholders.
    3. **Header analysis** — extract header features and compute risk score.

    Each step can be individually enabled or disabled.  If a step fails at
    runtime it is skipped and a warning is logged (fail-open behaviour so
    that mail delivery is never blocked by a preprocessing error).

    Example::

        pipeline = PreprocessingPipeline()
        result = pipeline.run("お世話になっております。")
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        nlp: Any = None,
    ) -> None:
        self._config = config or PipelineConfig()
        self._nlp = nlp

    @property
    def config(self) -> PipelineConfig:
        """Return the current pipeline configuration."""
        return self._config

    def run(
        self,
        text: str,
        *,
        headers: dict[str, object] | None = None,
    ) -> PreprocessingResult:
        """Execute the pipeline on *text*.

        Args:
            text: The email body text (subject + body concatenated).
            headers: Optional raw email headers for header analysis.

        Returns:
            A :class:`PreprocessingResult` aggregating all step outputs.
        """
        language: LanguageResult | None = None
        anon_result: AnonymizeResult | None = None
        hdr_features: HeaderFeatures | None = None
        hdr_score: float | None = None
        metadata: dict[str, object] = {}
        processed_text = text

        # Step 1: Language detection
        if self._config.language_detection:
            language = self._detect_language(text, metadata)

        # Step 2: Anonymization
        if self._config.anonymization:
            anon_result = self._anonymize(processed_text, metadata)
            if anon_result is not None:
                processed_text = anon_result.text

        # Step 3: Header analysis
        if self._config.header_analysis and headers is not None:
            hdr_features, hdr_score = self._analyze_headers(
                headers, metadata
            )

        return PreprocessingResult(
            text=processed_text,
            language=language,
            anonymize_result=anon_result,
            header_features=hdr_features,
            header_score=hdr_score,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private step methods
    # ------------------------------------------------------------------

    def _detect_language(
        self, text: str, metadata: dict[str, object]
    ) -> LanguageResult | None:
        try:
            result = detect_language(text)
            metadata["language"] = result.language
            metadata["language_confidence"] = result.confidence
            return result
        except (ValueError, ImportError):
            logger.warning("language detection failed, skipping step")
            metadata["language_error"] = True
            return None

    def _anonymize(
        self, text: str, metadata: dict[str, object]
    ) -> AnonymizeResult | None:
        try:
            result = anonymize(
                text,
                pii_types=self._config.anonymization_pii_types,
                use_ner=self._config.anonymization_use_ner,
                nlp=self._nlp,
            )
            metadata["pii_count"] = len(result.entities)
            pii_kinds: dict[str, int] = {}
            for entity in result.entities:
                pii_kinds[entity.kind] = pii_kinds.get(entity.kind, 0) + 1
            metadata["pii_kinds"] = pii_kinds
            return result
        except Exception:
            logger.warning(
                "anonymization failed, skipping step", exc_info=True
            )
            metadata["anonymization_error"] = True
            return None

    def _analyze_headers(
        self,
        headers: dict[str, object],
        metadata: dict[str, object],
    ) -> tuple[HeaderFeatures | None, float | None]:
        try:
            features = extract_features(headers)
            score = analyze_headers(headers, self._config.header_weights)
            metadata["header_score"] = score
            return features, score
        except Exception:
            logger.warning(
                "header analysis failed, skipping step", exc_info=True
            )
            metadata["header_analysis_error"] = True
            return None, None
