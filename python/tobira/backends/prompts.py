"""Shared prompt templates and utilities for LLM-based backends."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class FewShotExample:
    """A single few-shot example for LLM-based spam classification.

    Attributes:
        text: The email text (subject + body summary).
        label: The expected classification label ("spam" or "ham").
        score: The expected confidence score.
    """

    text: str
    label: str
    score: float


DEFAULT_FEW_SHOT_EXAMPLES_JA: list[FewShotExample] = [
    FewShotExample(
        text=(
            "500\u4e07\u5186\u5f53\u9078\uff01"
            "\u4eca\u3059\u3050 [URL] \u304b\u3089\u53d7\u3051\u53d6\u308a\u3002"
            "\u671f\u9650\u306f24\u6642\u9593\u4ee5\u5185\u3002"
        ),
        label="spam",
        score=0.97,
    ),
    FewShotExample(
        text=(
            "\u30a2\u30ab\u30a6\u30f3\u30c8\u306b\u4e0d\u6b63\u30a2\u30af\u30bb\u30b9\u691c\u51fa\u3002"
            "[URL] \u3088\u308a\u30d1\u30b9\u30ef\u30fc\u30c9\u5909\u66f4\u8981\u3002"
        ),
        label="spam",
        score=0.95,
    ),
    FewShotExample(
        text=(
            "\u4f1a\u8b70\u306e\u8b70\u4e8b\u9332\u3092\u6dfb\u4ed8\u3057\u307e\u3059\u3002"
            "\u6b21\u56de\u306f\u6765\u9031\u706b\u66dc 14:00\u3002"
        ),
        label="ham",
        score=0.96,
    ),
    FewShotExample(
        text=(
            "\u30d7\u30ed\u30b8\u30a7\u30af\u30c8\u306e\u9032\u6357\u5831\u544a\u3067\u3059\u3002"
            "\u30b9\u30b1\u30b8\u30e5\u30fc\u30eb\u901a\u308a\u9032\u884c\u4e2d\u3067\u3059\u3002"
        ),
        label="ham",
        score=0.94,
    ),
]

DEFAULT_FEW_SHOT_EXAMPLES_EN: list[FewShotExample] = [
    FewShotExample(
        text=(
            "Congratulations! You've won $1,000,000! "
            "Click [URL] to claim your prize now."
        ),
        label="spam",
        score=0.97,
    ),
    FewShotExample(
        text=(
            "URGENT: Your account has been compromised. "
            "Verify your identity at [URL] to avoid suspension."
        ),
        label="spam",
        score=0.95,
    ),
    FewShotExample(
        text=(
            "Hi team, please find the meeting notes attached. "
            "Next sync is Thursday at 2pm."
        ),
        label="ham",
        score=0.96,
    ),
    FewShotExample(
        text=(
            "Reminder: quarterly report is due by end of week. "
            "Let me know if you need any data."
        ),
        label="ham",
        score=0.94,
    ),
]

DEFAULT_FEW_SHOT_EXAMPLES: dict[str, list[FewShotExample]] = {
    "ja": DEFAULT_FEW_SHOT_EXAMPLES_JA,
    "en": DEFAULT_FEW_SHOT_EXAMPLES_EN,
}


def build_prompt(text: str) -> str:
    """Build the user prompt for spam classification.

    Args:
        text: The email text to classify.

    Returns:
        The formatted user prompt string.
    """
    return SPAM_CLASSIFICATION_USER.format(text=text)


def build_few_shot_system_prompt(
    examples: list[FewShotExample],
) -> str:
    """Build a system prompt with few-shot examples for Ollama API.

    The Ollama ``/api/generate`` endpoint uses a single ``system`` string,
    so examples are inlined into the system prompt.

    Args:
        examples: List of few-shot examples to include.

    Returns:
        A system prompt string containing classification instructions
        and few-shot examples.
    """
    parts = [SPAM_CLASSIFICATION_SYSTEM, "\n\nHere are some examples:\n"]
    for ex in examples:
        parts.append(
            f'\nEmail: {ex.text}\n'
            f'Answer: {{"label": "{ex.label}", "score": {ex.score}}}\n'
        )
    return "".join(parts)


def build_few_shot_messages(
    examples: list[FewShotExample],
    text: str,
) -> list[dict[str, str]]:
    """Build chat messages with few-shot examples for OpenAI-compatible API.

    Returns a list of message dicts suitable for the ``messages`` parameter
    of the chat completions API.

    Args:
        examples: List of few-shot examples to include.
        text: The email text to classify.

    Returns:
        A list of message dicts with system, example user/assistant pairs,
        and the final user message.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SPAM_CLASSIFICATION_SYSTEM},
    ]
    for ex in examples:
        messages.append({"role": "user", "content": build_prompt(ex.text)})
        messages.append(
            {
                "role": "assistant",
                "content": f'{{"label": "{ex.label}", "score": {ex.score}}}',
            }
        )
    messages.append({"role": "user", "content": build_prompt(text)})
    return messages


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
