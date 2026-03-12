"""Shared prompt templates and utilities for LLM-based backends."""

from __future__ import annotations

from typing import Any

from tobira.backends.protocol import PredictionResult

SPAM_CLASSIFICATION_SYSTEM = (
    "You are an email spam classifier. "
    "Classify the given email text as spam or ham. "
    "Respond ONLY with a JSON object in this exact format: "
    '{"label": "spam", "score": 0.95} '
    "where label is either \"spam\" or \"ham\" and score is your confidence "
    "between 0.0 and 1.0."
)

SPAM_CLASSIFICATION_USER = "Classify this email:\n\n{text}"


def build_prompt(text: str) -> str:
    """Build the user prompt for spam classification.

    Args:
        text: The email text to classify.

    Returns:
        The formatted user prompt string.
    """
    return SPAM_CLASSIFICATION_USER.format(text=text)


def parse_llm_result(raw: dict[str, Any]) -> PredictionResult:
    """Validate LLM JSON output and build a PredictionResult.

    Args:
        raw: Parsed JSON dict with ``label`` and ``score`` keys.

    Returns:
        A validated PredictionResult.

    Raises:
        ValueError: If the label is not ``spam``/``ham`` or score is out of range.
    """
    label = str(raw["label"]).lower()
    score = float(raw["score"])

    if label not in ("spam", "ham"):
        raise ValueError(f"unexpected label from LLM: {label!r}")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"score out of range: {score}")

    other = "ham" if label == "spam" else "spam"
    labels = {label: score, other: 1.0 - score}

    return PredictionResult(label=label, score=score, labels=labels)
