"""Tests for doctor feature-configuration checks."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import patch

from tobira.cli.doctor import (
    _check_ab_test,
    _check_active_learning,
    _check_ai_detection_threshold,
    _check_header_analysis_weight,
    _check_model_path,
    _check_notification,
    _check_retrain,
    _check_telemetry_storage,
)

# ── Telemetry storage_dir check ───────────────────────────────


class TestCheckTelemetryStorage:
    def test_disabled(self) -> None:
        ok, msg = _check_telemetry_storage({})
        assert ok is True
        assert "disabled" in msg

    def test_enabled_dir_exists_writable(self, tmp_path: Path) -> None:
        config = {"telemetry": {"enabled": True, "storage_dir": str(tmp_path)}}
        ok, msg = _check_telemetry_storage(config)
        assert ok is True
        assert "writable" in msg

    def test_enabled_dir_not_exists(self, tmp_path: Path) -> None:
        config = {
            "telemetry": {
                "enabled": True,
                "storage_dir": str(tmp_path / "nonexistent"),
            },
        }
        ok, msg = _check_telemetry_storage(config)
        assert ok is False
        assert "does not exist" in msg

    def test_enabled_dir_not_writable(self, tmp_path: Path) -> None:
        read_only = tmp_path / "readonly"
        read_only.mkdir()
        read_only.chmod(0o444)
        try:
            config = {
                "telemetry": {"enabled": True, "storage_dir": str(read_only)},
            }
            ok, msg = _check_telemetry_storage(config)
            assert ok is False
            assert "not writable" in msg
        finally:
            read_only.chmod(0o755)


# ── Header analysis weight check ─────────────────────────────


class TestCheckHeaderAnalysisWeight:
    def test_disabled(self) -> None:
        ok, msg = _check_header_analysis_weight({})
        assert ok is True
        assert "disabled" in msg

    def test_valid_weight(self) -> None:
        config = {"header_analysis": {"enabled": True, "weight": 0.5}}
        ok, msg = _check_header_analysis_weight(config)
        assert ok is True

    def test_default_weight(self) -> None:
        config = {"header_analysis": {"enabled": True}}
        ok, msg = _check_header_analysis_weight(config)
        assert ok is True
        assert "0.3" in msg

    def test_boundary_zero(self) -> None:
        config = {"header_analysis": {"enabled": True, "weight": 0.0}}
        ok, msg = _check_header_analysis_weight(config)
        assert ok is True

    def test_boundary_one(self) -> None:
        config = {"header_analysis": {"enabled": True, "weight": 1.0}}
        ok, msg = _check_header_analysis_weight(config)
        assert ok is True

    def test_negative_weight(self) -> None:
        config = {"header_analysis": {"enabled": True, "weight": -0.1}}
        ok, msg = _check_header_analysis_weight(config)
        assert ok is False
        assert "must be between" in msg

    def test_weight_above_one(self) -> None:
        config = {"header_analysis": {"enabled": True, "weight": 1.5}}
        ok, msg = _check_header_analysis_weight(config)
        assert ok is False
        assert "must be between" in msg

    def test_non_numeric_weight(self) -> None:
        config = {"header_analysis": {"enabled": True, "weight": "high"}}
        ok, msg = _check_header_analysis_weight(config)
        assert ok is False
        assert "must be between" in msg


# ── AI detection threshold check ─────────────────────────────


class TestCheckAiDetectionThreshold:
    def test_disabled(self) -> None:
        ok, msg = _check_ai_detection_threshold({})
        assert ok is True
        assert "disabled" in msg

    def test_valid_threshold(self) -> None:
        config = {"ai_detection": {"enabled": True, "threshold": 0.7}}
        ok, msg = _check_ai_detection_threshold(config)
        assert ok is True

    def test_default_threshold(self) -> None:
        config = {"ai_detection": {"enabled": True}}
        ok, msg = _check_ai_detection_threshold(config)
        assert ok is True
        assert "0.65" in msg

    def test_boundary_zero(self) -> None:
        config = {"ai_detection": {"enabled": True, "threshold": 0.0}}
        ok, msg = _check_ai_detection_threshold(config)
        assert ok is True

    def test_boundary_one(self) -> None:
        config = {"ai_detection": {"enabled": True, "threshold": 1.0}}
        ok, msg = _check_ai_detection_threshold(config)
        assert ok is True

    def test_negative_threshold(self) -> None:
        config = {"ai_detection": {"enabled": True, "threshold": -0.5}}
        ok, msg = _check_ai_detection_threshold(config)
        assert ok is False
        assert "must be between" in msg

    def test_threshold_above_one(self) -> None:
        config = {"ai_detection": {"enabled": True, "threshold": 2.0}}
        ok, msg = _check_ai_detection_threshold(config)
        assert ok is False
        assert "must be between" in msg


# ── Model path check ─────────────────────────────────────────


class TestCheckModelPath:
    def test_no_backend(self) -> None:
        results = _check_model_path({})
        assert results == []

    def test_unknown_backend(self) -> None:
        config = {"backend": {"backend": "dummy"}}
        results = _check_model_path(config)
        assert results == []

    def test_fasttext_exists(self, tmp_path: Path) -> None:
        model = tmp_path / "model.bin"
        model.write_bytes(b"fake")
        config = {"backend": {"backend": "fasttext", "model_path": str(model)}}
        results = _check_model_path(config)
        assert len(results) == 1
        assert results[0][0] is True
        assert "fasttext model file exists" in results[0][1]

    def test_fasttext_missing(self) -> None:
        config = {
            "backend": {
                "backend": "fasttext",
                "model_path": "/nonexistent/model.bin",
            },
        }
        results = _check_model_path(config)
        assert len(results) == 1
        assert results[0][0] is False
        assert "fasttext model file not found" in results[0][1]

    def test_onnx_exists(self, tmp_path: Path) -> None:
        model = tmp_path / "model.onnx"
        model.write_bytes(b"fake")
        config = {"backend": {"backend": "onnx", "model_path": str(model)}}
        results = _check_model_path(config)
        assert len(results) == 1
        assert results[0][0] is True
        assert "ONNX model file exists" in results[0][1]

    def test_onnx_missing(self) -> None:
        config = {
            "backend": {
                "backend": "onnx",
                "model_path": "/nonexistent/model.onnx",
            },
        }
        results = _check_model_path(config)
        assert len(results) == 1
        assert results[0][0] is False
        assert "ONNX model file not found" in results[0][1]

    def test_no_model_path(self) -> None:
        config = {"backend": {"backend": "fasttext"}}
        results = _check_model_path(config)
        assert results == []


# ── A/B test checks ──────────────────────────────────────────


class TestCheckABTest:
    def test_disabled_returns_empty(self) -> None:
        config: dict = {"ab_test": {"enabled": False}}
        assert _check_ab_test(config) == []

    def test_missing_section_returns_empty(self) -> None:
        assert _check_ab_test({}) == []

    def test_no_variants(self) -> None:
        config = {"ab_test": {"enabled": True}}
        results = _check_ab_test(config)
        assert len(results) == 1
        assert results[0][0] is False
        assert "no variants" in results[0][1]

    def test_empty_variants_list(self) -> None:
        config = {"ab_test": {"enabled": True, "variants": []}}
        results = _check_ab_test(config)
        assert results[0][0] is False

    def test_duplicate_names(self) -> None:
        config = {
            "ab_test": {
                "enabled": True,
                "variants": [
                    {"name": "a", "weight": 1.0, "backend": {"type": "dummy"}},
                    {"name": "a", "weight": 1.0, "backend": {"type": "dummy"}},
                ],
            },
        }
        results = _check_ab_test(config)
        name_check = results[0]
        assert name_check[0] is False
        assert "not unique" in name_check[1]

    def test_zero_total_weight(self) -> None:
        config = {
            "ab_test": {
                "enabled": True,
                "variants": [
                    {"name": "a", "weight": 0, "backend": {"type": "dummy"}},
                    {"name": "b", "weight": 0, "backend": {"type": "dummy"}},
                ],
            },
        }
        results = _check_ab_test(config)
        weight_checks = [r for r in results if "weight" in r[1]]
        assert any(r[0] is False for r in weight_checks)

    def test_valid_config_with_mock_backend(self) -> None:
        config = {
            "ab_test": {
                "enabled": True,
                "variants": [
                    {"name": "control", "weight": 80, "backend": {"type": "dummy"}},
                    {"name": "candidate", "weight": 20, "backend": {"type": "dummy"}},
                ],
            },
        }
        with patch("tobira.backends.factory.create_backend") as mock_create:
            mock_create.return_value = None
            results = _check_ab_test(config)
            passed = [r for r in results if r[0] is True]
            assert len(passed) >= 3  # names unique, weight ok, 2 backends ok

    def test_backend_init_failure(self) -> None:
        config = {
            "ab_test": {
                "enabled": True,
                "variants": [
                    {"name": "bad", "weight": 1, "backend": {"type": "broken"}},
                ],
            },
        }
        with patch(
            "tobira.backends.factory.create_backend",
            side_effect=ValueError("bad config"),
        ):
            results = _check_ab_test(config)
            backend_checks = [r for r in results if "backend" in r[1] and "bad" in r[1]]
            assert any(r[0] is False for r in backend_checks)

    def test_variant_missing_backend(self) -> None:
        config = {
            "ab_test": {
                "enabled": True,
                "variants": [
                    {"name": "no-backend", "weight": 1},
                ],
            },
        }
        results = _check_ab_test(config)
        assert any("no backend config" in r[1] for r in results)


# ── Active Learning checks ───────────────────────────────────


class TestCheckActiveLearning:
    def test_disabled_returns_empty(self) -> None:
        config: dict = {"active_learning": {"enabled": False}}
        assert _check_active_learning(config) == []

    def test_missing_section_returns_empty(self) -> None:
        assert _check_active_learning({}) == []

    def test_invalid_strategy(self) -> None:
        config = {"active_learning": {"enabled": True, "strategy": "unknown"}}
        results = _check_active_learning(config)
        strategy_checks = [r for r in results if "strategy" in r[1]]
        assert any(r[0] is False for r in strategy_checks)

    def test_valid_strategies(self) -> None:
        for strat in ("entropy", "margin", "least_confidence"):
            config = {"active_learning": {"enabled": True, "strategy": strat}}
            results = _check_active_learning(config)
            strategy_checks = [r for r in results if "strategy" in r[1]]
            assert all(r[0] is True for r in strategy_checks)

    def test_invalid_threshold(self) -> None:
        config = {
            "active_learning": {
                "enabled": True,
                "uncertainty_threshold": 1.5,
            },
        }
        results = _check_active_learning(config)
        threshold_checks = [r for r in results if "threshold" in r[1]]
        assert any(r[0] is False for r in threshold_checks)

    def test_negative_threshold(self) -> None:
        config = {
            "active_learning": {
                "enabled": True,
                "uncertainty_threshold": -0.1,
            },
        }
        results = _check_active_learning(config)
        threshold_checks = [r for r in results if "threshold" in r[1]]
        assert any(r[0] is False for r in threshold_checks)

    def test_invalid_max_queue_size(self) -> None:
        config = {
            "active_learning": {
                "enabled": True,
                "max_queue_size": 0,
            },
        }
        results = _check_active_learning(config)
        size_checks = [r for r in results if "max_queue_size" in r[1]]
        assert any(r[0] is False for r in size_checks)

    def test_valid_config(self, tmp_path: Path) -> None:
        queue_path = str(tmp_path / "queue.jsonl")
        config = {
            "active_learning": {
                "enabled": True,
                "strategy": "margin",
                "uncertainty_threshold": 0.5,
                "max_queue_size": 500,
                "queue_path": queue_path,
            },
        }
        results = _check_active_learning(config)
        assert all(r[0] is True for r in results)

    def test_queue_path_parent_missing(self) -> None:
        config = {
            "active_learning": {
                "enabled": True,
                "queue_path": "/nonexistent/path/queue.jsonl",
            },
        }
        results = _check_active_learning(config)
        path_checks = [r for r in results if "queue_path" in r[1]]
        assert any(r[0] is False for r in path_checks)


# ── Notification checks ──────────────────────────────────────


class TestCheckNotification:
    def test_disabled_returns_empty(self) -> None:
        config: dict = {"notification": {"enabled": False}}
        assert _check_notification(config) == []

    def test_missing_section_returns_empty(self) -> None:
        assert _check_notification({}) == []

    def test_enabled_no_channels(self) -> None:
        config = {"notification": {"enabled": True, "channels": []}}
        results = _check_notification(config)
        assert results[0][0] is False
        assert "no channels" in results[0][1]

    def test_slack_empty_webhook(self) -> None:
        config = {
            "notification": {
                "enabled": True,
                "channels": [{"type": "slack", "webhook_url": ""}],
            },
        }
        results = _check_notification(config)
        assert any(r[0] is False and "webhook_url is empty" in r[1] for r in results)

    def test_slack_valid_webhook(self) -> None:
        config = {
            "notification": {
                "enabled": True,
                "channels": [
                    {"type": "slack", "webhook_url": "https://hooks.slack.com/x"},
                ],
            },
        }
        results = _check_notification(config)
        assert all(r[0] is True for r in results)

    def test_teams_valid_webhook(self) -> None:
        config = {
            "notification": {
                "enabled": True,
                "channels": [
                    {"type": "teams", "webhook_url": "https://outlook.office.com/x"},
                ],
            },
        }
        results = _check_notification(config)
        assert all(r[0] is True for r in results)

    def test_email_missing_fields(self) -> None:
        config = {
            "notification": {
                "enabled": True,
                "channels": [{"type": "email"}],
            },
        }
        results = _check_notification(config)
        assert any(r[0] is False and "missing required fields" in r[1] for r in results)

    def test_email_valid(self) -> None:
        config = {
            "notification": {
                "enabled": True,
                "channels": [
                    {
                        "type": "email",
                        "smtp_host": "smtp.example.com",
                        "from_addr": "noreply@example.com",
                        "to_addrs": ["admin@example.com"],
                    },
                ],
            },
        }
        results = _check_notification(config)
        config_checks = [r for r in results if "config OK" in r[1]]
        assert len(config_checks) == 1

    def test_env_var_resolved(self) -> None:
        config = {
            "notification": {
                "enabled": True,
                "channels": [
                    {"type": "slack", "webhook_url": "${SLACK_WEBHOOK}"},
                ],
            },
        }
        with patch.dict(os.environ, {"SLACK_WEBHOOK": "https://hooks.slack.com/x"}):
            results = _check_notification(config)
            assert all(r[0] is True for r in results)

    def test_env_var_not_set(self) -> None:
        config = {
            "notification": {
                "enabled": True,
                "channels": [
                    {"type": "slack", "webhook_url": "${MISSING_WEBHOOK}"},
                ],
            },
        }
        with patch.dict(os.environ, {}, clear=True):
            results = _check_notification(config)
            assert any(r[0] is False and "MISSING_WEBHOOK" in r[1] for r in results)

    def test_email_env_var_not_set(self) -> None:
        config = {
            "notification": {
                "enabled": True,
                "channels": [
                    {
                        "type": "email",
                        "smtp_host": "smtp.example.com",
                        "from_addr": "noreply@example.com",
                        "to_addrs": ["admin@example.com"],
                        "username": "${SMTP_USER}",
                    },
                ],
            },
        }
        with patch.dict(os.environ, {}, clear=True):
            results = _check_notification(config)
            assert any(r[0] is False and "SMTP_USER" in r[1] for r in results)


# ── Retrain checks ───────────────────────────────────────────


class TestCheckRetrain:
    def test_no_retrain_section_returns_empty(self) -> None:
        assert _check_retrain({}) == []

    def test_disabled_returns_empty(self) -> None:
        config = {"retrain": {"enabled": False}}
        assert _check_retrain(config) == []

    def test_invalid_action(self) -> None:
        config = {"retrain": {"action": "invalid"}}
        results = _check_retrain(config)
        assert results[0][0] is False
        assert "invalid action" in results[0][1]

    def test_log_action_valid(self) -> None:
        config = {"retrain": {"action": "log"}}
        results = _check_retrain(config)
        assert results[0][0] is True
        assert "log" in results[0][1]

    def test_script_no_path(self) -> None:
        config = {"retrain": {"action": "script"}}
        results = _check_retrain(config)
        assert any(r[0] is False and "script_path is empty" in r[1] for r in results)

    def test_script_path_not_found(self) -> None:
        config = {"retrain": {"action": "script", "script_path": "/no/such/script.sh"}}
        results = _check_retrain(config)
        assert any(r[0] is False and "not found" in r[1] for r in results)

    def test_script_path_not_executable(self, tmp_path: Path) -> None:
        script = tmp_path / "retrain.sh"
        script.write_text("#!/bin/bash\necho ok")
        script.chmod(stat.S_IRUSR | stat.S_IWUSR)  # not executable
        config = {"retrain": {"action": "script", "script_path": str(script)}}
        results = _check_retrain(config)
        assert any(r[0] is False and "not executable" in r[1] for r in results)

    def test_script_path_valid(self, tmp_path: Path) -> None:
        script = tmp_path / "retrain.sh"
        script.write_text("#!/bin/bash\necho ok")
        script.chmod(stat.S_IRWXU)
        config = {"retrain": {"action": "script", "script_path": str(script)}}
        results = _check_retrain(config)
        assert any(r[0] is True and "script_path OK" in r[1] for r in results)

    def test_webhook_no_url(self) -> None:
        config = {"retrain": {"action": "webhook"}}
        results = _check_retrain(config)
        assert any(r[0] is False and "webhook_url is empty" in r[1] for r in results)

    def test_webhook_valid(self) -> None:
        config = {
            "retrain": {
                "action": "webhook",
                "webhook_url": "https://example.com/retrain",
            },
        }
        results = _check_retrain(config)
        assert all(r[0] is True for r in results)
