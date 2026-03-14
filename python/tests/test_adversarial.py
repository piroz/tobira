"""Tests for tobira.adversarial — homoglyph and invisible Unicode detection."""

from __future__ import annotations

import pytest

from tobira.adversarial.detectors import (
    AdversarialDetection,
    HomoglyphDetector,
    InvisibleUnicodeDetector,
    normalize,
)


class TestHomoglyphDetectorDetect:
    def test_cyrillic_a(self) -> None:
        detector = HomoglyphDetector()
        results = detector.detect("sp\u0430m")  # Cyrillic а
        assert len(results) == 1
        assert results[0] == (2, "\u0430", "a")

    def test_multiple_cyrillic(self) -> None:
        detector = HomoglyphDetector()
        # "Неllo" with Cyrillic Н and е
        results = detector.detect("\u041d\u0435llo")
        assert len(results) == 2
        assert results[0][2] == "H"
        assert results[1][2] == "e"

    def test_greek_omicron(self) -> None:
        detector = HomoglyphDetector()
        results = detector.detect("g\u03bfod")  # Greek ο
        assert len(results) == 1
        assert results[0] == (1, "\u03bf", "o")

    def test_fullwidth_characters(self) -> None:
        detector = HomoglyphDetector()
        results = detector.detect("\uff33\uff50\uff41\uff4d")  # Ｓｐａｍ
        assert len(results) == 4

    def test_no_homoglyphs(self) -> None:
        detector = HomoglyphDetector()
        results = detector.detect("normal text")
        assert results == []

    def test_empty_string(self) -> None:
        detector = HomoglyphDetector()
        results = detector.detect("")
        assert results == []

    def test_extra_mappings(self) -> None:
        detector = HomoglyphDetector(extra_mappings={"\u00d8": "O"})
        results = detector.detect("\u00d8K")
        assert len(results) == 1
        assert results[0][2] == "O"


class TestHomoglyphDetectorNormalize:
    def test_cyrillic_to_latin(self) -> None:
        detector = HomoglyphDetector()
        text, count = detector.normalize("sp\u0430m")  # Cyrillic а → a
        assert text == "spam"
        assert count == 1

    def test_mixed_scripts(self) -> None:
        detector = HomoglyphDetector()
        # "Неllo wоrld" with Cyrillic Н, е, о
        text, count = detector.normalize("\u041d\u0435llo w\u043erld")
        assert text == "Hello world"
        assert count == 3

    def test_fullwidth_normalization(self) -> None:
        detector = HomoglyphDetector()
        text, count = detector.normalize("\uff33\uff50\uff41\uff4d")
        assert text == "Spam"
        assert count == 4

    def test_no_changes(self) -> None:
        detector = HomoglyphDetector()
        text, count = detector.normalize("hello world")
        assert text == "hello world"
        assert count == 0

    def test_japanese_text_preserved(self) -> None:
        detector = HomoglyphDetector()
        text, count = detector.normalize("こんにちは世界")
        assert text == "こんにちは世界"
        assert count == 0


class TestInvisibleUnicodeDetectorDetect:
    def test_zwsp(self) -> None:
        detector = InvisibleUnicodeDetector()
        results = detector.detect("sp\u200Bam")  # Zero-width space
        assert len(results) == 1
        assert results[0] == (2, "\u200B")

    def test_multiple_invisible(self) -> None:
        detector = InvisibleUnicodeDetector()
        text = "s\u200Bp\u200Ca\u200Dm"  # ZWSP, ZWNJ, ZWJ
        results = detector.detect(text)
        assert len(results) == 3

    def test_direction_overrides(self) -> None:
        detector = InvisibleUnicodeDetector()
        text = "\u202Ehello\u202C"  # RTL override + pop
        results = detector.detect(text)
        assert len(results) == 2

    def test_variation_selectors(self) -> None:
        detector = InvisibleUnicodeDetector()
        results = detector.detect("A\uFE0F")  # Variation selector
        assert len(results) == 1

    def test_soft_hyphen(self) -> None:
        detector = InvisibleUnicodeDetector()
        results = detector.detect("vi\u00ADagra")  # Soft hyphen
        assert len(results) == 1
        assert results[0] == (2, "\u00AD")

    def test_bom(self) -> None:
        detector = InvisibleUnicodeDetector()
        results = detector.detect("\uFEFFhello")
        assert len(results) == 1

    def test_no_invisible(self) -> None:
        detector = InvisibleUnicodeDetector()
        results = detector.detect("normal text")
        assert results == []

    def test_empty_string(self) -> None:
        detector = InvisibleUnicodeDetector()
        results = detector.detect("")
        assert results == []

    def test_extra_codepoints(self) -> None:
        detector = InvisibleUnicodeDetector(extra_codepoints={0x2028})
        results = detector.detect("line\u2028break")
        assert len(results) == 1


class TestInvisibleUnicodeDetectorNormalize:
    def test_remove_zwsp(self) -> None:
        detector = InvisibleUnicodeDetector()
        text, count = detector.normalize("sp\u200Bam")
        assert text == "spam"
        assert count == 1

    def test_remove_multiple(self) -> None:
        detector = InvisibleUnicodeDetector()
        text, count = detector.normalize("s\u200Bp\u200Ca\u200Dm")
        assert text == "spam"
        assert count == 3

    def test_remove_direction_overrides(self) -> None:
        detector = InvisibleUnicodeDetector()
        text, count = detector.normalize("\u202Ehello\u202C")
        assert text == "hello"
        assert count == 2

    def test_no_changes(self) -> None:
        detector = InvisibleUnicodeDetector()
        text, count = detector.normalize("hello world")
        assert text == "hello world"
        assert count == 0


class TestNormalize:
    def test_combined_normalization(self) -> None:
        # Cyrillic а + zero-width space
        result = normalize("sp\u0430\u200Bm")
        assert result.normalized == "spam"
        assert result.homoglyph_count == 1
        assert result.invisible_count == 1
        assert result.has_adversarial is True

    def test_no_adversarial(self) -> None:
        result = normalize("normal text")
        assert result.normalized == "normal text"
        assert result.homoglyph_count == 0
        assert result.invisible_count == 0
        assert result.has_adversarial is False

    def test_original_preserved(self) -> None:
        original = "sp\u0430\u200Bm"
        result = normalize(original)
        assert result.original == original
        assert result.normalized != result.original

    def test_homoglyphs_only(self) -> None:
        result = normalize("sp\u0430m", homoglyphs=True, invisible=False)
        assert result.normalized == "spam"
        assert result.homoglyph_count == 1
        assert result.invisible_count == 0

    def test_invisible_only(self) -> None:
        result = normalize("sp\u200Bam", homoglyphs=False, invisible=True)
        assert result.normalized == "spam"
        assert result.homoglyph_count == 0
        assert result.invisible_count == 1

    def test_nfkc_normalization(self) -> None:
        # NFKC normalizes ﬁ ligature to "fi"
        result = normalize("\uFB01le", homoglyphs=False, invisible=False, nfkc=True)
        assert result.normalized == "file"

    def test_disable_nfkc(self) -> None:
        result = normalize("\uFB01le", homoglyphs=False, invisible=False, nfkc=False)
        assert result.normalized == "\uFB01le"

    def test_empty_string(self) -> None:
        result = normalize("")
        assert result.normalized == ""
        assert result.has_adversarial is False

    def test_real_spam_evasion(self) -> None:
        # Simulated spam: "Free Vi​аgra" with Cyrillic а and ZWSP
        text = "Free Vi\u200B\u0430gra"
        result = normalize(text)
        assert result.normalized == "Free Viagra"
        assert result.has_adversarial is True


class TestAdversarialDetection:
    def test_frozen(self) -> None:
        detection = AdversarialDetection(
            original="test", normalized="test"
        )
        with pytest.raises(AttributeError):
            detection.normalized = "changed"  # type: ignore[misc]

    def test_has_adversarial_true(self) -> None:
        detection = AdversarialDetection(
            original="test",
            normalized="test",
            homoglyph_count=1,
        )
        assert detection.has_adversarial is True

    def test_has_adversarial_false(self) -> None:
        detection = AdversarialDetection(
            original="test",
            normalized="test",
        )
        assert detection.has_adversarial is False
