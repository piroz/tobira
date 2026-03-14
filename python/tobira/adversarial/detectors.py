"""Adversarial text manipulation detectors for spam evasion techniques.

Detects and normalizes homoglyph substitutions (e.g. Cyrillic 'а' → Latin 'a')
and invisible Unicode characters (e.g. zero-width spaces, direction overrides)
commonly used in spam to evade text-based classifiers.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Homoglyph mapping: visually similar non-Latin characters → ASCII equivalent
# ---------------------------------------------------------------------------

# Cyrillic → Latin
_CYRILLIC_MAP: dict[str, str] = {
    "\u0410": "A",  # А
    "\u0412": "B",  # В
    "\u0421": "C",  # С
    "\u0415": "E",  # Е
    "\u041d": "H",  # Н
    "\u041a": "K",  # К
    "\u041c": "M",  # М
    "\u041e": "O",  # О
    "\u0420": "P",  # Р
    "\u0422": "T",  # Т
    "\u0425": "X",  # Х
    "\u0430": "a",  # а
    "\u0435": "e",  # е
    "\u043e": "o",  # о
    "\u0440": "p",  # р
    "\u0441": "c",  # с
    "\u0443": "y",  # у
    "\u0445": "x",  # х
    "\u0455": "s",  # ѕ (Macedonian)
    "\u0456": "i",  # і (Ukrainian/Belarusian)
    "\u0458": "j",  # ј (Serbian)
    "\u04bb": "h",  # һ (Bashkir/Chuvash)
}

# Greek → Latin
_GREEK_MAP: dict[str, str] = {
    "\u0391": "A",  # Α
    "\u0392": "B",  # Β
    "\u0395": "E",  # Ε
    "\u0396": "Z",  # Ζ
    "\u0397": "H",  # Η
    "\u0399": "I",  # Ι
    "\u039a": "K",  # Κ
    "\u039c": "M",  # Μ
    "\u039d": "N",  # Ν
    "\u039f": "O",  # Ο
    "\u03a1": "P",  # Ρ
    "\u03a4": "T",  # Τ
    "\u03a5": "Y",  # Υ
    "\u03a7": "X",  # Χ
    "\u03bf": "o",  # ο
    "\u03b1": "a",  # α (visually similar in some fonts)
}

# Fullwidth → ASCII
_FULLWIDTH_MAP: dict[str, str] = {}
for _code in range(0xFF01, 0xFF5F):
    _FULLWIDTH_MAP[chr(_code)] = chr(_code - 0xFEE0)

# Combined homoglyph mapping
HOMOGLYPH_MAP: dict[str, str] = {
    **_CYRILLIC_MAP,
    **_GREEK_MAP,
    **_FULLWIDTH_MAP,
}

# Build a character class pattern matching any homoglyph character
_HOMOGLYPH_CHARS = re.escape("".join(HOMOGLYPH_MAP.keys()))
_HOMOGLYPH_PATTERN = re.compile(f"[{_HOMOGLYPH_CHARS}]")

# ---------------------------------------------------------------------------
# Invisible Unicode characters
# ---------------------------------------------------------------------------

_INVISIBLE_CODEPOINTS: set[int] = {
    0x200B,  # Zero-width space (ZWSP)
    0x200C,  # Zero-width non-joiner (ZWNJ)
    0x200D,  # Zero-width joiner (ZWJ)
    0x200E,  # Left-to-right mark
    0x200F,  # Right-to-left mark
    0x202A,  # Left-to-right embedding
    0x202B,  # Right-to-left embedding
    0x202C,  # Pop directional formatting
    0x202D,  # Left-to-right override
    0x202E,  # Right-to-left override
    0x2060,  # Word joiner
    0x2061,  # Function application
    0x2062,  # Invisible times
    0x2063,  # Invisible separator
    0x2064,  # Invisible plus
    0x2066,  # Left-to-right isolate
    0x2067,  # Right-to-left isolate
    0x2068,  # First strong isolate
    0x2069,  # Pop directional isolate
    0xFEFF,  # Byte order mark / zero-width no-break space
    0x00AD,  # Soft hyphen
    0x034F,  # Combining grapheme joiner
    0x061C,  # Arabic letter mark
    0x180E,  # Mongolian vowel separator
}

# Variation selectors: U+FE00..U+FE0F and U+E0100..U+E01EF
for _vs in range(0xFE00, 0xFE10):
    _INVISIBLE_CODEPOINTS.add(_vs)
for _vs in range(0xE0100, 0xE01F0):
    _INVISIBLE_CODEPOINTS.add(_vs)

_INVISIBLE_CHARS = "".join(chr(cp) for cp in sorted(_INVISIBLE_CODEPOINTS))
_INVISIBLE_PATTERN = re.compile(f"[{re.escape(_INVISIBLE_CHARS)}]")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdversarialDetection:
    """Result of adversarial text analysis.

    Attributes:
        original: The original input text.
        normalized: Text after applying all normalizations.
        homoglyph_count: Number of homoglyph characters detected and replaced.
        invisible_count: Number of invisible characters detected and removed.
    """

    original: str
    normalized: str
    homoglyph_count: int = 0
    invisible_count: int = 0

    @property
    def has_adversarial(self) -> bool:
        """Return ``True`` if any adversarial manipulation was detected."""
        return self.homoglyph_count > 0 or self.invisible_count > 0


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------


class HomoglyphDetector:
    """Detect and normalize homoglyph characters.

    Replaces visually similar characters from Cyrillic, Greek, and fullwidth
    Unicode ranges with their ASCII Latin equivalents.
    """

    def __init__(
        self,
        *,
        extra_mappings: dict[str, str] | None = None,
    ) -> None:
        self._map = dict(HOMOGLYPH_MAP)
        if extra_mappings:
            self._map.update(extra_mappings)
        chars = re.escape("".join(self._map.keys()))
        self._pattern = re.compile(f"[{chars}]")

    def detect(self, text: str) -> list[tuple[int, str, str]]:
        """Find homoglyph characters in *text*.

        Args:
            text: Input text to scan.

        Returns:
            A list of ``(position, original_char, replacement_char)`` tuples.
        """
        results: list[tuple[int, str, str]] = []
        for m in self._pattern.finditer(text):
            ch = m.group()
            replacement = self._map.get(ch, ch)
            results.append((m.start(), ch, replacement))
        return results

    def normalize(self, text: str) -> tuple[str, int]:
        """Replace homoglyph characters with their ASCII equivalents.

        Args:
            text: Input text.

        Returns:
            A tuple of ``(normalized_text, replacement_count)``.
        """
        count = 0

        def _replace(m: re.Match[str]) -> str:
            nonlocal count
            ch = m.group()
            replacement = self._map.get(ch, ch)
            if replacement != ch:
                count += 1
            return replacement

        normalized = self._pattern.sub(_replace, text)
        return normalized, count


class InvisibleUnicodeDetector:
    """Detect and remove invisible Unicode characters.

    Targets zero-width characters, directional overrides, variation selectors,
    and other invisible formatting characters commonly inserted to evade
    text-based spam filters.
    """

    def __init__(
        self,
        *,
        extra_codepoints: set[int] | None = None,
    ) -> None:
        codepoints = set(_INVISIBLE_CODEPOINTS)
        if extra_codepoints:
            codepoints.update(extra_codepoints)
        chars = "".join(chr(cp) for cp in sorted(codepoints))
        self._pattern = re.compile(f"[{re.escape(chars)}]")

    def detect(self, text: str) -> list[tuple[int, str]]:
        """Find invisible Unicode characters in *text*.

        Args:
            text: Input text to scan.

        Returns:
            A list of ``(position, character)`` tuples.
        """
        return [(m.start(), m.group()) for m in self._pattern.finditer(text)]

    def normalize(self, text: str) -> tuple[str, int]:
        """Remove invisible Unicode characters from *text*.

        Args:
            text: Input text.

        Returns:
            A tuple of ``(cleaned_text, removal_count)``.
        """
        cleaned, count = self._pattern.subn("", text)
        return cleaned, count


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

_DEFAULT_HOMOGLYPH = HomoglyphDetector()
_DEFAULT_INVISIBLE = InvisibleUnicodeDetector()


def normalize(
    text: str,
    *,
    homoglyphs: bool = True,
    invisible: bool = True,
    nfkc: bool = True,
) -> AdversarialDetection:
    """Normalize adversarial text manipulations.

    Applies homoglyph replacement, invisible character removal, and Unicode
    NFKC normalization in sequence.

    Args:
        text: Input text to normalize.
        homoglyphs: If ``True``, replace homoglyph characters with ASCII
            equivalents.
        invisible: If ``True``, remove invisible Unicode characters.
        nfkc: If ``True``, apply Unicode NFKC normalization after other steps.

    Returns:
        An :class:`AdversarialDetection` with the original and normalized text.
    """
    original = text
    homoglyph_count = 0
    invisible_count = 0

    if invisible:
        text, invisible_count = _DEFAULT_INVISIBLE.normalize(text)

    if homoglyphs:
        text, homoglyph_count = _DEFAULT_HOMOGLYPH.normalize(text)

    if nfkc:
        text = unicodedata.normalize("NFKC", text)

    return AdversarialDetection(
        original=original,
        normalized=text,
        homoglyph_count=homoglyph_count,
        invisible_count=invisible_count,
    )
