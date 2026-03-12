"""PII anonymization using regex-based detection and optional GiNZA NER."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

# PII patterns: (name, compiled regex)
_PII_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "email",
        re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    ),
    (
        "url",
        re.compile(r"https?://[^\s<>\"']+"),
    ),
    (
        "credit_card",
        re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b"),
    ),
    (
        "my_number",
        re.compile(r"\b\d{4} ?\d{4} ?\d{4}\b"),
    ),
    (
        "phone",
        re.compile(r"0\d{1,4}[-(]?\d{1,4}[-)]?\d{3,4}"),
    ),
    (
        "postal_code",
        re.compile(r"\b\d{3}-\d{4}\b"),
    ),
    (
        "ip_address",
        re.compile(
            r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
            r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
        ),
    ),
]


@dataclass(frozen=True)
class PIIEntity:
    """A detected PII entity.

    Attributes:
        kind: PII type (e.g. ``email``, ``phone``).
        start: Start offset in the original text.
        end: End offset in the original text.
        text: The matched substring.
    """

    kind: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class AnonymizeResult:
    """Result of anonymization.

    Attributes:
        text: Text with PII replaced by placeholders.
        entities: Detected PII entities before replacement.
    """

    text: str
    entities: tuple[PIIEntity, ...] = field(default_factory=tuple)


def _import_ginza() -> Any:
    """Lazily import spacy with GiNZA model."""
    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spacy and ginza are required for NER-based anonymization. "
            "Install them with: pip install spacy ginza ja-ginza"
        ) from exc
    return spacy


def detect_pii(text: str) -> list[PIIEntity]:
    """Detect PII entities in *text* using regex patterns.

    Args:
        text: Input text to scan.

    Returns:
        A list of :class:`PIIEntity` sorted by start offset.
    """
    entities: list[PIIEntity] = []
    for kind, pattern in _PII_PATTERNS:
        for m in pattern.finditer(text):
            entities.append(
                PIIEntity(kind=kind, start=m.start(), end=m.end(), text=m.group())
            )
    # Sort by start position and remove overlapping matches (keep first)
    entities.sort(key=lambda e: (e.start, -e.end))
    merged: list[PIIEntity] = []
    for entity in entities:
        if merged and entity.start < merged[-1].end:
            continue
        merged.append(entity)
    return merged


def detect_pii_with_ner(text: str, nlp: Any = None) -> list[PIIEntity]:
    """Detect PII entities using both regex and GiNZA NER.

    Args:
        text: Input text to scan.
        nlp: A pre-loaded spaCy ``Language`` instance.  If *None* the default
            ``ja_ginza`` model is loaded.

    Returns:
        A list of :class:`PIIEntity` sorted by start offset.
    """
    entities = detect_pii(text)
    if nlp is None:
        spacy = _import_ginza()
        nlp = spacy.load("ja_ginza")
    doc = nlp(text)
    person_labels = {"PERSON", "Person", "人名"}
    for ent in doc.ents:
        if ent.label_ in person_labels:
            entities.append(
                PIIEntity(
                    kind="person",
                    start=ent.start_char,
                    end=ent.end_char,
                    text=ent.text,
                )
            )
    entities.sort(key=lambda e: (e.start, -e.end))
    merged: list[PIIEntity] = []
    for entity in entities:
        if merged and entity.start < merged[-1].end:
            continue
        merged.append(entity)
    return merged


def anonymize(
    text: str,
    *,
    pii_types: Sequence[str] | None = None,
    use_ner: bool = False,
    nlp: Any = None,
) -> AnonymizeResult:
    """Replace PII in *text* with placeholders.

    Args:
        text: Input text.
        pii_types: If given, only anonymize these PII types.
        use_ner: If ``True``, also use GiNZA NER for person-name detection.
        nlp: Pre-loaded spaCy model (used when *use_ner* is True).

    Returns:
        An :class:`AnonymizeResult` with the scrubbed text and detected entities.
    """
    if use_ner:
        entities = detect_pii_with_ner(text, nlp=nlp)
    else:
        entities = detect_pii(text)

    if pii_types is not None:
        allowed = set(pii_types)
        entities = [e for e in entities if e.kind in allowed]

    # Replace from end to preserve offsets
    result = text
    for entity in reversed(entities):
        placeholder = f"<{entity.kind.upper()}>"
        result = result[: entity.start] + placeholder + result[entity.end :]

    return AnonymizeResult(text=result, entities=tuple(entities))
