"""Tests for tobira.preprocessing.recognizers.japanese — Japanese PII detection."""

from __future__ import annotations

from unittest.mock import MagicMock

from tobira.preprocessing.recognizers.japanese import (
    JapanesePIIRecognizer,
    _valid_my_number_check_digit,
    detect_japanese_pii,
)

# ---------------------------------------------------------------------------
# My Number check-digit validation
# ---------------------------------------------------------------------------


class TestMyNumberCheckDigit:
    def test_valid_check_digit(self) -> None:
        # Use a known valid My Number: 123456789018
        # Weights: 6,5,4,3,2,7,6,5,4,3,2
        # Sum = 1*6+2*5+3*4+4*3+5*2+6*7+7*6+8*5+9*4+0*3+1*2
        #     = 6+10+12+12+10+42+42+40+36+0+2 = 212
        # 212 % 11 = 3, expected = 11 - 3 = 8
        assert _valid_my_number_check_digit("123456789018") is True

    def test_invalid_check_digit(self) -> None:
        assert _valid_my_number_check_digit("123456789019") is False

    def test_non_digit_string(self) -> None:
        assert _valid_my_number_check_digit("12345678901a") is False

    def test_wrong_length(self) -> None:
        assert _valid_my_number_check_digit("12345678901") is False
        assert _valid_my_number_check_digit("1234567890123") is False

    def test_remainder_zero(self) -> None:
        # When remainder is 0, check digit should be 0
        assert _valid_my_number_check_digit("000000000000") is True

    def test_remainder_one(self) -> None:
        # When remainder is 1, check digit should be 0
        # 100000000000: 1*6=6, rest=0 -> sum=6, 6%11=6, expected=11-6=5 ≠ 0
        # Find a number where remainder is 1:
        # We need sum%11 == 1 -> check digit = 0
        # sum = 1: d0=0,d1=0,...d9=0, d10=0 => sum=0, but d0*6=...
        # Try: digits 00000000010 -> sum = 0*6+...+1*3+0*2 = 3, 3%11=3, exp=8
        # Try: 20000000000 -> sum=2*6=12, 12%11=1, expected=0
        assert _valid_my_number_check_digit("200000000000") is True


# ---------------------------------------------------------------------------
# My Number detection (context-aware)
# ---------------------------------------------------------------------------


class TestDetectMyNumber:
    def test_with_keyword_context(self) -> None:
        # 123456789018 has valid check digit
        text = "マイナンバー: 1234 5678 9018"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "my_number"]
        assert len(found) == 1
        assert found[0].text == "1234 5678 9018"

    def test_without_keyword_context(self) -> None:
        text = "番号は 1234 5678 9018 です"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "my_number"]
        assert len(found) == 0

    def test_invalid_check_digit_rejected(self) -> None:
        text = "マイナンバー: 1234 5678 9019"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "my_number"]
        assert len(found) == 0

    def test_kojin_bango_keyword(self) -> None:
        text = "個人番号は1234 5678 9018です"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "my_number"]
        assert len(found) == 1

    def test_no_spaces(self) -> None:
        text = "マイナンバー: 123456789018"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "my_number"]
        assert len(found) == 1

    def test_with_hyphens(self) -> None:
        text = "マイナンバー: 1234-5678-9018"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "my_number"]
        assert len(found) == 1


# ---------------------------------------------------------------------------
# Japanese address detection
# ---------------------------------------------------------------------------


class TestDetectAddress:
    def test_tokyo(self) -> None:
        text = "住所: 東京都千代田区丸の内1丁目"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "address"]
        assert len(found) == 1
        assert "東京都" in found[0].text

    def test_osaka(self) -> None:
        text = "大阪府大阪市北区梅田"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "address"]
        assert len(found) == 1
        assert "大阪府" in found[0].text

    def test_hokkaido(self) -> None:
        text = "北海道札幌市中央区"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "address"]
        assert len(found) == 1
        assert "北海道" in found[0].text

    def test_kyoto(self) -> None:
        text = "京都府京都市左京区"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "address"]
        assert len(found) == 1
        assert "京都府" in found[0].text

    def test_no_address(self) -> None:
        text = "これは普通のテキストです。"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "address"]
        assert len(found) == 0

    def test_multiple_addresses(self) -> None:
        text = "東京都千代田区と大阪府大阪市"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "address"]
        assert len(found) == 2


# ---------------------------------------------------------------------------
# Bank account detection
# ---------------------------------------------------------------------------


class TestDetectBankAccount:
    def test_with_keyword(self) -> None:
        text = "口座番号: 1234567"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "bank_account"]
        assert len(found) == 1
        assert found[0].text == "1234567"

    def test_without_keyword(self) -> None:
        text = "番号: 1234567"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "bank_account"]
        assert len(found) == 0

    def test_furikomi_keyword(self) -> None:
        text = "振込先 1234567"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "bank_account"]
        assert len(found) == 1


# ---------------------------------------------------------------------------
# Driver's license detection
# ---------------------------------------------------------------------------


class TestDetectDriversLicense:
    def test_with_keyword(self) -> None:
        text = "運転免許証番号: 123456789012"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "drivers_license"]
        assert len(found) == 1
        assert found[0].text == "123456789012"

    def test_without_keyword(self) -> None:
        text = "番号: 123456789012"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "drivers_license"]
        assert len(found) == 0

    def test_license_keyword(self) -> None:
        text = "免許番号 123456789012"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "drivers_license"]
        assert len(found) == 1


# ---------------------------------------------------------------------------
# NER-based name detection
# ---------------------------------------------------------------------------


class TestDetectNamesNER:
    def test_person_detection(self) -> None:
        mock_nlp = MagicMock()
        mock_ent = MagicMock()
        mock_ent.label_ = "Person"
        mock_ent.start_char = 0
        mock_ent.end_char = 4
        mock_ent.text = "田中太郎"
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc

        entities = detect_japanese_pii(
            "田中太郎さんのメール", use_ner=True, nlp=mock_nlp
        )
        found = [e for e in entities if e.kind == "person"]
        assert len(found) == 1
        assert found[0].text == "田中太郎"

    def test_organization_detection(self) -> None:
        mock_nlp = MagicMock()
        mock_ent = MagicMock()
        mock_ent.label_ = "ORG"
        mock_ent.start_char = 0
        mock_ent.end_char = 6
        mock_ent.text = "株式会社テスト"
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc

        entities = detect_japanese_pii(
            "株式会社テストの住所", use_ner=True, nlp=mock_nlp
        )
        found = [e for e in entities if e.kind == "organization"]
        assert len(found) == 1
        assert found[0].text == "株式会社テスト"

    def test_ner_disabled_by_default(self) -> None:
        # Without use_ner, NER-based entities should not appear
        text = "田中太郎さん"
        entities = detect_japanese_pii(text)
        found = [e for e in entities if e.kind == "person"]
        assert len(found) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_string(self) -> None:
        entities = detect_japanese_pii("")
        assert entities == []

    def test_no_japanese_pii(self) -> None:
        entities = detect_japanese_pii("Hello, this is a test.")
        assert entities == []

    def test_entities_sorted_by_offset(self) -> None:
        text = "北海道札幌市中央区の口座番号: 1234567"
        entities = detect_japanese_pii(text)
        offsets = [e.start for e in entities]
        assert offsets == sorted(offsets)

    def test_no_overlapping_entities(self) -> None:
        text = "東京都千代田区で口座番号: 1234567 を使用"
        entities = detect_japanese_pii(text)
        for i in range(1, len(entities)):
            assert entities[i].start >= entities[i - 1].end

    def test_recognizer_class_usage(self) -> None:
        recognizer = JapanesePIIRecognizer()
        entities = recognizer.detect("東京都千代田区")
        found = [e for e in entities if e.kind == "address"]
        assert len(found) == 1


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_imports_from_package(self) -> None:
        from tobira.preprocessing.recognizers import (
            JapanesePIIRecognizer,
            detect_japanese_pii,
        )

        assert callable(detect_japanese_pii)
        assert JapanesePIIRecognizer is not None
