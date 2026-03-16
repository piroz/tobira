"""Tests for tobira.preprocessing.headers."""

from __future__ import annotations

import pytest

from tobira.preprocessing.headers import (
    HeaderFeatures,
    analyze_headers,
    compute_header_score,
    extract_features,
)


class TestExtractFeatures:
    def test_empty_headers(self) -> None:
        features = extract_features({})
        assert features.spf == 0.5  # none → 0.5
        assert features.dkim == 0.5
        assert features.dmarc == 0.5
        assert features.reply_to_mismatch == 0.0
        assert features.hop_count == 0.0
        assert features.content_type_anomaly == 0.0

    def test_spf_pass(self) -> None:
        features = extract_features({"spf": "pass"})
        assert features.spf == 0.0

    def test_spf_fail(self) -> None:
        features = extract_features({"spf": "fail"})
        assert features.spf == 1.0

    def test_spf_softfail(self) -> None:
        features = extract_features({"spf": "softfail"})
        assert features.spf == pytest.approx(0.7)

    def test_dkim_pass(self) -> None:
        features = extract_features({"dkim": "pass"})
        assert features.dkim == 0.0

    def test_dkim_fail(self) -> None:
        features = extract_features({"dkim": "fail"})
        assert features.dkim == 1.0

    def test_dmarc_pass(self) -> None:
        features = extract_features({"dmarc": "pass"})
        assert features.dmarc == 0.0

    def test_reply_to_mismatch(self) -> None:
        features = extract_features({
            "from": "user@example.com",
            "reply_to": "other@different.com",
        })
        assert features.reply_to_mismatch == 1.0

    def test_reply_to_match(self) -> None:
        features = extract_features({
            "from": "user@example.com",
            "reply_to": "support@example.com",
        })
        assert features.reply_to_mismatch == 0.0

    def test_reply_to_absent(self) -> None:
        features = extract_features({"from": "user@example.com"})
        assert features.reply_to_mismatch == 0.0

    def test_from_with_display_name(self) -> None:
        features = extract_features({
            "from": "User Name <user@example.com>",
            "reply_to": "Other <other@different.com>",
        })
        assert features.reply_to_mismatch == 1.0

    def test_hop_count_normal(self) -> None:
        received = [f"hop{i}" for i in range(5)]
        features = extract_features({"received": received})
        assert features.hop_count == 0.0

    def test_hop_count_high(self) -> None:
        received = [f"hop{i}" for i in range(12)]
        features = extract_features({"received": received})
        assert features.hop_count == pytest.approx(0.3)

    def test_hop_count_very_high(self) -> None:
        received = [f"hop{i}" for i in range(20)]
        features = extract_features({"received": received})
        assert features.hop_count == pytest.approx(0.6)

    def test_content_type_normal(self) -> None:
        features = extract_features({"content_type": "text/plain; charset=utf-8"})
        assert features.content_type_anomaly == 0.0

    def test_content_type_nested_multipart(self) -> None:
        features = extract_features({
            "content_type": "multipart/mixed; multipart/alternative"
        })
        assert features.content_type_anomaly == pytest.approx(0.7)

    def test_content_type_suspicious_charset(self) -> None:
        features = extract_features({
            "content_type": "text/plain; charset=gb2312"
        })
        assert features.content_type_anomaly == pytest.approx(0.3)


class TestComputeHeaderScore:
    def test_all_pass(self) -> None:
        features = HeaderFeatures(
            spf=0.0, dkim=0.0, dmarc=0.0,
            reply_to_mismatch=0.0, hop_count=0.0, content_type_anomaly=0.0,
        )
        score = compute_header_score(features)
        assert score == pytest.approx(0.0)

    def test_all_fail(self) -> None:
        features = HeaderFeatures(
            spf=1.0, dkim=1.0, dmarc=1.0,
            reply_to_mismatch=1.0, hop_count=1.0, content_type_anomaly=1.0,
        )
        score = compute_header_score(features)
        assert score == pytest.approx(1.0)

    def test_custom_weights(self) -> None:
        features = HeaderFeatures(spf=1.0)
        score = compute_header_score(
            features, weights={"spf": 1.0, "dkim": 0.0, "dmarc": 0.0,
                               "reply_to_mismatch": 0.0, "hop_count": 0.0,
                               "content_type_anomaly": 0.0}
        )
        assert score == pytest.approx(1.0)

    def test_zero_total_weight(self) -> None:
        features = HeaderFeatures(spf=1.0)
        score = compute_header_score(
            features, weights={"spf": 0.0, "dkim": 0.0, "dmarc": 0.0,
                               "reply_to_mismatch": 0.0, "hop_count": 0.0,
                               "content_type_anomaly": 0.0}
        )
        assert score == 0.0

    def test_score_clamped_to_unit_range(self) -> None:
        features = HeaderFeatures(spf=1.0, dkim=1.0, dmarc=1.0,
                                  reply_to_mismatch=1.0, hop_count=1.0,
                                  content_type_anomaly=1.0)
        score = compute_header_score(features)
        assert 0.0 <= score <= 1.0


class TestAnalyzeHeaders:
    def test_convenience_function(self) -> None:
        headers = {"spf": "fail", "dkim": "fail", "dmarc": "fail"}
        score = analyze_headers(headers)
        assert score > 0.0

    def test_clean_headers(self) -> None:
        headers = {"spf": "pass", "dkim": "pass", "dmarc": "pass"}
        score = analyze_headers(headers)
        # spf/dkim/dmarc all pass (0.0), but empty other fields default to 0.0
        # except the missing auth results which default to 0.5 in extract_features
        # But here we provide all three auth results as pass
        assert score < 0.5

    def test_suspicious_headers(self) -> None:
        headers = {
            "spf": "fail",
            "dkim": "fail",
            "dmarc": "fail",
            "from": "user@legit.com",
            "reply_to": "scam@phishing.com",
        }
        score = analyze_headers(headers)
        assert score > 0.5
