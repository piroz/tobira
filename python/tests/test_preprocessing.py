"""Tests for tobira.preprocessing — PII detection and anonymization."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tobira.preprocessing.anonymizer import (
    PIIEntity,
    anonymize,
    detect_pii,
    detect_pii_with_ner,
)


class TestDetectPIIEmail:
    def test_simple_email(self) -> None:
        entities = detect_pii("連絡先: user@example.com まで")
        assert len(entities) == 1
        assert entities[0].kind == "email"
        assert entities[0].text == "user@example.com"

    def test_email_with_plus(self) -> None:
        entities = detect_pii("user+tag@example.co.jp")
        assert len(entities) == 1
        assert entities[0].kind == "email"
        assert entities[0].text == "user+tag@example.co.jp"


class TestDetectPIIUrl:
    def test_https_url(self) -> None:
        entities = detect_pii("詳細は https://example.com/page を参照")
        assert len(entities) == 1
        assert entities[0].kind == "url"
        assert entities[0].text == "https://example.com/page"

    def test_http_url(self) -> None:
        entities = detect_pii("http://example.com")
        assert len(entities) == 1
        assert entities[0].kind == "url"


class TestDetectPIIPhone:
    def test_mobile_number(self) -> None:
        entities = detect_pii("電話: 090-1234-5678")
        found = [e for e in entities if e.kind == "phone"]
        assert len(found) == 1
        assert "090" in found[0].text

    def test_landline_number(self) -> None:
        entities = detect_pii("TEL: 03-1234-5678")
        found = [e for e in entities if e.kind == "phone"]
        assert len(found) == 1


class TestDetectPIIPostalCode:
    def test_postal_code(self) -> None:
        entities = detect_pii("〒100-0001 東京都千代田区")
        found = [e for e in entities if e.kind == "postal_code"]
        assert len(found) == 1
        assert found[0].text == "100-0001"


class TestDetectPIICreditCard:
    def test_credit_card_with_spaces(self) -> None:
        entities = detect_pii("カード番号: 4111 1111 1111 1111")
        found = [e for e in entities if e.kind == "credit_card"]
        assert len(found) == 1

    def test_credit_card_with_dashes(self) -> None:
        entities = detect_pii("4111-1111-1111-1111")
        found = [e for e in entities if e.kind == "credit_card"]
        assert len(found) == 1


class TestDetectPIIMyNumber:
    def test_my_number_with_spaces(self) -> None:
        entities = detect_pii("マイナンバー: 1234 5678 9012")
        found = [e for e in entities if e.kind == "my_number"]
        assert len(found) >= 1

    def test_my_number_no_spaces(self) -> None:
        entities = detect_pii("番号: 123456789012")
        found = [e for e in entities if e.kind == "my_number"]
        assert len(found) >= 1


class TestDetectPIIIPAddress:
    def test_ipv4(self) -> None:
        entities = detect_pii("サーバー: 192.168.1.1 に接続")
        found = [e for e in entities if e.kind == "ip_address"]
        assert len(found) == 1
        assert found[0].text == "192.168.1.1"

    def test_invalid_ip_out_of_range(self) -> None:
        entities = detect_pii("999.999.999.999")
        found = [e for e in entities if e.kind == "ip_address"]
        assert len(found) == 0


class TestDetectPIIEdgeCases:
    def test_no_pii(self) -> None:
        entities = detect_pii("これは普通のテキストです。PIIは含まれません。")
        assert entities == []

    def test_empty_string(self) -> None:
        entities = detect_pii("")
        assert entities == []

    def test_multiple_pii_types(self) -> None:
        text = "メール: foo@bar.com 電話: 090-1111-2222 IP: 10.0.0.1"
        entities = detect_pii(text)
        kinds = {e.kind for e in entities}
        assert "email" in kinds
        assert "phone" in kinds
        assert "ip_address" in kinds

    def test_entities_sorted_by_offset(self) -> None:
        text = "IP: 10.0.0.1 メール: a@b.com"
        entities = detect_pii(text)
        offsets = [e.start for e in entities]
        assert offsets == sorted(offsets)

    def test_overlapping_matches_resolved(self) -> None:
        # A credit card pattern might overlap with my_number;
        # the first match (by start) should win.
        text = "4111 1111 1111 1111"
        entities = detect_pii(text)
        # Should not have overlapping entities
        for i in range(1, len(entities)):
            assert entities[i].start >= entities[i - 1].end


class TestAnonymize:
    def test_replaces_email(self) -> None:
        result = anonymize("連絡先: user@example.com まで")
        assert "<EMAIL>" in result.text
        assert "user@example.com" not in result.text
        assert len(result.entities) == 1
        assert result.entities[0].kind == "email"

    def test_replaces_multiple_types(self) -> None:
        text = "メール: a@b.com 電話: 090-0000-1111"
        result = anonymize(text)
        assert "<EMAIL>" in result.text
        assert "<PHONE>" in result.text
        assert "a@b.com" not in result.text

    def test_no_pii_returns_unchanged(self) -> None:
        text = "PIIなしのテキスト"
        result = anonymize(text)
        assert result.text == text
        assert result.entities == ()

    def test_pii_types_filter(self) -> None:
        text = "メール: a@b.com IP: 10.0.0.1"
        result = anonymize(text, pii_types=["email"])
        assert "<EMAIL>" in result.text
        # IP should remain
        assert "10.0.0.1" in result.text

    def test_frozen_result(self) -> None:
        result = anonymize("test user@example.com")
        with pytest.raises(AttributeError):
            result.text = "modified"  # type: ignore[misc]

    def test_frozen_entity(self) -> None:
        entity = PIIEntity(kind="email", start=0, end=5, text="a@b.c")
        with pytest.raises(AttributeError):
            entity.kind = "other"  # type: ignore[misc]


class TestDetectPIIWithNER:
    def test_uses_mock_nlp(self) -> None:
        mock_nlp = MagicMock()
        mock_ent = MagicMock()
        mock_ent.label_ = "Person"
        mock_ent.start_char = 0
        mock_ent.end_char = 2
        mock_ent.text = "太郎"
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc

        entities = detect_pii_with_ner("太郎 user@example.com", nlp=mock_nlp)
        kinds = {e.kind for e in entities}
        assert "person" in kinds
        assert "email" in kinds

    def test_import_error_without_spacy(self) -> None:
        from tobira.preprocessing.anonymizer import _import_ginza

        with patch.dict("sys.modules", {"spacy": None}):
            with pytest.raises(ImportError, match="spacy and ginza"):
                _import_ginza()

    def test_anonymize_with_ner(self) -> None:
        mock_nlp = MagicMock()
        mock_ent = MagicMock()
        mock_ent.label_ = "PERSON"
        mock_ent.start_char = 0
        mock_ent.end_char = 2
        mock_ent.text = "太郎"
        mock_doc = MagicMock()
        mock_doc.ents = [mock_ent]
        mock_nlp.return_value = mock_doc

        result = anonymize("太郎のメールは a@b.com", use_ner=True, nlp=mock_nlp)
        assert "<PERSON>" in result.text
        assert "<EMAIL>" in result.text


class TestPreprocessingPublicAPI:
    def test_imports_from_package(self) -> None:
        from tobira import preprocessing

        assert hasattr(preprocessing, "AnonymizeResult")
        assert hasattr(preprocessing, "PIIEntity")
        assert hasattr(preprocessing, "anonymize")
        assert hasattr(preprocessing, "detect_pii")
        assert hasattr(preprocessing, "detect_pii_with_ner")
