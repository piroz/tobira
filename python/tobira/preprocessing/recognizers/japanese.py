"""Japanese-specific PII recognizers.

Provides high-accuracy detection of Japanese PII types that require
context-aware matching beyond simple regex patterns:

- My Number (マイナンバー) with keyword context validation
- Japanese addresses (都道府県 level)
- Bank account numbers with context keywords
- Driver's license numbers with context keywords
- Japanese person names via GiNZA NER with organization filtering

This module produces :class:`~tobira.preprocessing.anonymizer.PIIEntity`
instances compatible with the existing anonymization pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from tobira.preprocessing.anonymizer import PIIEntity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PREFECTURES: tuple[str, ...] = (
    "北海道",
    "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県",
    "岐阜県", "静岡県", "愛知県", "三重県",
    "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
    "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県",
    "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県",
    "沖縄県",
)

# My Number context keywords (must appear within _CONTEXT_WINDOW chars)
_MY_NUMBER_KEYWORDS: tuple[str, ...] = (
    "マイナンバー",
    "個人番号",
    "my number",
    "mynumber",
)

_BANK_ACCOUNT_KEYWORDS: tuple[str, ...] = (
    "口座番号",
    "口座",
    "振込先",
    "銀行口座",
    "account",
)

_LICENSE_KEYWORDS: tuple[str, ...] = (
    "免許証",
    "運転免許",
    "免許番号",
    "license",
)

# How many characters before/after the match to search for context keywords.
_CONTEXT_WINDOW = 30

# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

# 12-digit My Number (with optional spaces/hyphens)
# Use (?<!\d) / (?!\d) instead of \b to handle Japanese text boundaries.
_MY_NUMBER_RE = re.compile(r"(?<!\d)(\d{4})[- ]?(\d{4})[- ]?(\d{4})(?!\d)")

# Japanese address: prefecture name (都道府県 level)
# Match prefecture + following city/ward/town characters.
# Use a non-greedy quantifier and stop before another prefecture name.
_PREFECTURE_PATTERN = "|".join(re.escape(p) for p in _PREFECTURES)
_ADDRESS_RE = re.compile(
    r"(?:" + _PREFECTURE_PATTERN + r")"
    r"(?:(?!" + _PREFECTURE_PATTERN + r")[^\s、。,.\n]){0,30}"
)

# Bank account: 7-digit number with context
_BANK_ACCOUNT_RE = re.compile(r"\b\d{7}\b")

# Driver's license: 12-digit number
_LICENSE_RE = re.compile(r"\b\d{12}\b")

# Organization labels from GiNZA NER to exclude from person detection
_ORG_LABELS = {"ORG", "Organization", "組織"}


# ---------------------------------------------------------------------------
# Helper: context check
# ---------------------------------------------------------------------------

def _has_context(
    text: str,
    start: int,
    end: int,
    keywords: Sequence[str],
    window: int = _CONTEXT_WINDOW,
) -> bool:
    """Return True if any *keyword* appears near the match in *text*."""
    region_start = max(0, start - window)
    region_end = min(len(text), end + window)
    region = text[region_start:region_end].lower()
    return any(kw.lower() in region for kw in keywords)


# ---------------------------------------------------------------------------
# My Number check-digit (modulus 11)
# ---------------------------------------------------------------------------

_MY_NUMBER_WEIGHTS = (6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)


def _valid_my_number_check_digit(digits: str) -> bool:
    """Validate the My Number check digit (12th digit).

    Uses the official modulus-11 algorithm defined by the Japanese
    government specification.
    """
    if len(digits) != 12 or not digits.isdigit():
        return False
    total = sum(int(digits[i]) * _MY_NUMBER_WEIGHTS[i] for i in range(11))
    remainder = total % 11
    if remainder <= 1:
        expected = 0
    else:
        expected = 11 - remainder
    return int(digits[11]) == expected


# ---------------------------------------------------------------------------
# JapanesePIIRecognizer
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JapanesePIIRecognizer:
    """Detects Japanese-specific PII entities.

    Attributes:
        use_ner: Whether to use GiNZA NER for person/organization names.
        nlp: Pre-loaded spaCy Language instance. Loaded lazily if *None*.
    """

    use_ner: bool = False
    nlp: Any = None

    def detect(self, text: str) -> list[PIIEntity]:
        """Detect Japanese-specific PII in *text*.

        Returns a list of :class:`PIIEntity` sorted by start offset,
        with overlapping matches removed (first match wins).
        """
        entities: list[PIIEntity] = []
        entities.extend(self._detect_my_number(text))
        entities.extend(self._detect_address(text))
        entities.extend(self._detect_bank_account(text))
        entities.extend(self._detect_license(text))
        if self.use_ner:
            entities.extend(self._detect_names_ner(text))

        # Sort and de-duplicate overlapping matches
        entities.sort(key=lambda e: (e.start, -e.end))
        merged: list[PIIEntity] = []
        for entity in entities:
            if merged and entity.start < merged[-1].end:
                continue
            merged.append(entity)
        return merged

    # -- My Number --------------------------------------------------------

    def _detect_my_number(self, text: str) -> list[PIIEntity]:
        results: list[PIIEntity] = []
        for m in _MY_NUMBER_RE.finditer(text):
            digits = m.group(1) + m.group(2) + m.group(3)
            if not _has_context(text, m.start(), m.end(), _MY_NUMBER_KEYWORDS):
                continue
            if not _valid_my_number_check_digit(digits):
                continue
            results.append(
                PIIEntity(
                    kind="my_number",
                    start=m.start(),
                    end=m.end(),
                    text=m.group(),
                )
            )
        return results

    # -- Address ----------------------------------------------------------

    def _detect_address(self, text: str) -> list[PIIEntity]:
        results: list[PIIEntity] = []
        for m in _ADDRESS_RE.finditer(text):
            results.append(
                PIIEntity(
                    kind="address",
                    start=m.start(),
                    end=m.end(),
                    text=m.group(),
                )
            )
        return results

    # -- Bank account -----------------------------------------------------

    def _detect_bank_account(self, text: str) -> list[PIIEntity]:
        results: list[PIIEntity] = []
        for m in _BANK_ACCOUNT_RE.finditer(text):
            if not _has_context(text, m.start(), m.end(), _BANK_ACCOUNT_KEYWORDS):
                continue
            results.append(
                PIIEntity(
                    kind="bank_account",
                    start=m.start(),
                    end=m.end(),
                    text=m.group(),
                )
            )
        return results

    # -- Driver's license -------------------------------------------------

    def _detect_license(self, text: str) -> list[PIIEntity]:
        results: list[PIIEntity] = []
        for m in _LICENSE_RE.finditer(text):
            if not _has_context(text, m.start(), m.end(), _LICENSE_KEYWORDS):
                continue
            results.append(
                PIIEntity(
                    kind="drivers_license",
                    start=m.start(),
                    end=m.end(),
                    text=m.group(),
                )
            )
        return results

    # -- Person names (NER) -----------------------------------------------

    def _detect_names_ner(self, text: str) -> list[PIIEntity]:
        nlp = self.nlp
        if nlp is None:
            from tobira.preprocessing.anonymizer import _import_ginza

            spacy = _import_ginza()
            nlp = spacy.load("ja_ginza")
        doc = nlp(text)
        results: list[PIIEntity] = []
        person_labels = {"PERSON", "Person", "人名"}
        for ent in doc.ents:
            if ent.label_ in person_labels:
                results.append(
                    PIIEntity(
                        kind="person",
                        start=ent.start_char,
                        end=ent.end_char,
                        text=ent.text,
                    )
                )
            elif ent.label_ in _ORG_LABELS:
                results.append(
                    PIIEntity(
                        kind="organization",
                        start=ent.start_char,
                        end=ent.end_char,
                        text=ent.text,
                    )
                )
        return results


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_japanese_pii(
    text: str,
    *,
    use_ner: bool = False,
    nlp: Any = None,
) -> list[PIIEntity]:
    """Detect Japanese-specific PII in *text*.

    This is a convenience wrapper around :class:`JapanesePIIRecognizer`.

    Args:
        text: Input text to scan.
        use_ner: If *True*, also detect person/organization names via GiNZA.
        nlp: Pre-loaded spaCy ``Language`` instance.

    Returns:
        A list of :class:`PIIEntity` sorted by start offset.
    """
    recognizer = JapanesePIIRecognizer(use_ner=use_ner, nlp=nlp)
    return recognizer.detect(text)
