"""Tests for tobira.monitoring.retrain."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tobira.monitoring.retrain import (
    RetrainConfig,
    RetrainEvent,
    _event_to_dict,
    _last_retrain_timestamp,
    check_retrain_needed,
    load_retrain_config,
    trigger_retrain,
)

# ── RetrainConfig tests ────────────────────────────────────────


class TestRetrainConfig:
    def test_defaults(self) -> None:
        config = RetrainConfig()
        assert config.enabled is True
        assert config.drift_threshold == 0.1
        assert config.min_feedback_count == 100
        assert config.max_interval_days == 30
        assert config.action == "log"

    def test_frozen(self) -> None:
        config = RetrainConfig()
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore[misc]


class TestLoadRetrainConfig:
    def test_from_empty_dict(self) -> None:
        config = load_retrain_config({})
        assert config.enabled is True
        assert config.action == "log"

    def test_from_full_dict(self) -> None:
        raw = {
            "enabled": False,
            "drift_threshold": 0.2,
            "min_feedback_count": 50,
            "max_interval_days": 14,
            "action": "webhook",
            "webhook_url": "https://example.com/hook",
            "retrain_log_path": "/tmp/retrain.jsonl",
        }
        config = load_retrain_config(raw)
        assert config.enabled is False
        assert config.drift_threshold == 0.2
        assert config.min_feedback_count == 50
        assert config.max_interval_days == 14
        assert config.action == "webhook"
        assert config.webhook_url == "https://example.com/hook"

    def test_invalid_action_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid retrain action"):
            load_retrain_config({"action": "invalid"})

    def test_unknown_keys_ignored(self) -> None:
        config = load_retrain_config({"unknown_key": 42})
        assert config.enabled is True


# ── RetrainEvent tests ─────────────────────────────────────────


class TestRetrainEvent:
    def test_frozen(self) -> None:
        event = RetrainEvent(
            timestamp="2025-01-01T00:00:00+00:00",
            reason="test", action="log", success=True,
        )
        with pytest.raises(AttributeError):
            event.success = False  # type: ignore[misc]

    def test_event_to_dict(self) -> None:
        event = RetrainEvent(
            timestamp="2025-01-01T00:00:00+00:00",
            reason="drift", action="log", success=True, detail="ok",
        )
        d = _event_to_dict(event)
        assert d["timestamp"] == "2025-01-01T00:00:00+00:00"
        assert d["reason"] == "drift"
        assert d["action"] == "log"
        assert d["success"] is True
        assert d["detail"] == "ok"


# ── _last_retrain_timestamp tests ──────────────────────────────


class TestLastRetrainTimestamp:
    def test_returns_none_when_file_missing(self) -> None:
        result = _last_retrain_timestamp("/nonexistent/retrain.jsonl")
        assert result is None

    def test_returns_none_when_no_successful_events(
        self, tmp_path: Path,
    ) -> None:
        log_file = tmp_path / "retrain.jsonl"
        log_file.write_text(
            json.dumps({
                "timestamp": "2025-01-01T00:00:00+00:00",
                "success": False,
            }) + "\n"
        )
        result = _last_retrain_timestamp(str(log_file))
        assert result is None

    def test_returns_latest_successful_timestamp(
        self, tmp_path: Path,
    ) -> None:
        log_file = tmp_path / "retrain.jsonl"
        lines = [
            json.dumps({
                "timestamp": "2025-01-01T00:00:00+00:00",
                "success": True,
            }),
            json.dumps({
                "timestamp": "2025-02-01T00:00:00+00:00",
                "success": True,
            }),
            json.dumps({
                "timestamp": "2025-03-01T00:00:00+00:00",
                "success": False,
            }),
        ]
        log_file.write_text("\n".join(lines) + "\n")
        result = _last_retrain_timestamp(str(log_file))
        assert result == "2025-02-01T00:00:00+00:00"


# ── check_retrain_needed tests ─────────────────────────────────


class TestCheckRetrainNeeded:
    def test_disabled_returns_none(self) -> None:
        config = RetrainConfig(enabled=False)
        assert check_retrain_needed(config, current_psi=1.0) is None

    def test_drift_triggers(self) -> None:
        config = RetrainConfig(drift_threshold=0.1)
        reason = check_retrain_needed(config, current_psi=0.15)
        assert reason is not None
        assert "Drift detected" in reason
        assert "0.15" in reason

    def test_drift_below_threshold(self, tmp_path: Path) -> None:
        log_file = tmp_path / "retrain.jsonl"
        # Write a recent successful retrain event to avoid time trigger
        now = datetime.now(timezone.utc).isoformat()
        log_file.write_text(
            json.dumps({"timestamp": now, "success": True}) + "\n"
        )
        config = RetrainConfig(
            drift_threshold=0.2,
            retrain_log_path=str(log_file),
        )
        reason = check_retrain_needed(config, current_psi=0.05)
        assert reason is None

    def test_feedback_count_triggers(self, tmp_path: Path) -> None:
        log_file = tmp_path / "retrain.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        log_file.write_text(
            json.dumps({"timestamp": now, "success": True}) + "\n"
        )
        config = RetrainConfig(
            min_feedback_count=100,
            retrain_log_path=str(log_file),
        )
        reason = check_retrain_needed(config, feedback_count=150)
        assert reason is not None
        assert "Feedback count" in reason

    def test_feedback_below_threshold(self, tmp_path: Path) -> None:
        log_file = tmp_path / "retrain.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        log_file.write_text(
            json.dumps({"timestamp": now, "success": True}) + "\n"
        )
        config = RetrainConfig(
            min_feedback_count=100,
            retrain_log_path=str(log_file),
        )
        reason = check_retrain_needed(config, feedback_count=50)
        assert reason is None

    def test_time_interval_triggers(self, tmp_path: Path) -> None:
        log_file = tmp_path / "retrain.jsonl"
        old_ts = (
            datetime.now(timezone.utc) - timedelta(days=40)
        ).isoformat()
        log_file.write_text(
            json.dumps({"timestamp": old_ts, "success": True}) + "\n"
        )
        config = RetrainConfig(
            max_interval_days=30,
            retrain_log_path=str(log_file),
        )
        reason = check_retrain_needed(config)
        assert reason is not None
        assert "Time since last retrain" in reason

    def test_no_previous_retrain_triggers(self) -> None:
        config = RetrainConfig(
            retrain_log_path="/nonexistent/retrain.jsonl",
        )
        reason = check_retrain_needed(config)
        assert reason is not None
        assert "No previous retrain" in reason

    def test_drift_checked_first(self, tmp_path: Path) -> None:
        """Drift is checked before feedback and time."""
        log_file = tmp_path / "retrain.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        log_file.write_text(
            json.dumps({"timestamp": now, "success": True}) + "\n"
        )
        config = RetrainConfig(
            drift_threshold=0.1,
            min_feedback_count=10,
            retrain_log_path=str(log_file),
        )
        reason = check_retrain_needed(
            config, current_psi=0.5, feedback_count=100,
        )
        assert reason is not None
        assert "Drift" in reason


# ── trigger_retrain tests ──────────────────────────────────────


class TestTriggerRetrain:
    def test_no_trigger_when_not_needed(self, tmp_path: Path) -> None:
        log_file = tmp_path / "retrain.jsonl"
        now = datetime.now(timezone.utc).isoformat()
        log_file.write_text(
            json.dumps({"timestamp": now, "success": True}) + "\n"
        )
        config = RetrainConfig(retrain_log_path=str(log_file))
        event = trigger_retrain(config, current_psi=0.01, feedback_count=0)
        assert event is None

    def test_log_action(self, tmp_path: Path) -> None:
        log_file = tmp_path / "retrain.jsonl"
        config = RetrainConfig(
            action="log",
            retrain_log_path=str(log_file),
        )
        event = trigger_retrain(config, current_psi=0.5)
        assert event is not None
        assert event.action == "log"
        assert event.success is True
        # Event should be recorded
        records = json.loads(log_file.read_text().strip().split("\n")[-1])
        assert records["success"] is True
        assert "Drift" in records["reason"]

    def test_script_action_success(self, tmp_path: Path) -> None:
        log_file = tmp_path / "retrain.jsonl"
        script = tmp_path / "retrain.sh"
        script.write_text("#!/bin/sh\necho 'done'")
        script.chmod(0o755)

        config = RetrainConfig(
            action="script",
            script_path=str(script),
            retrain_log_path=str(log_file),
        )
        event = trigger_retrain(config, current_psi=0.5)
        assert event is not None
        assert event.action == "script"
        assert event.success is True
        assert "done" in event.detail

    def test_script_action_failure(self, tmp_path: Path) -> None:
        log_file = tmp_path / "retrain.jsonl"
        script = tmp_path / "fail.sh"
        script.write_text("#!/bin/sh\nexit 1")
        script.chmod(0o755)

        config = RetrainConfig(
            action="script",
            script_path=str(script),
            retrain_log_path=str(log_file),
        )
        event = trigger_retrain(config, current_psi=0.5)
        assert event is not None
        assert event.success is False

    def test_script_action_no_path(self, tmp_path: Path) -> None:
        log_file = tmp_path / "retrain.jsonl"
        config = RetrainConfig(
            action="script",
            script_path="",
            retrain_log_path=str(log_file),
        )
        event = trigger_retrain(config, current_psi=0.5)
        assert event is not None
        assert event.success is False
        assert "No script_path" in event.detail

    def test_webhook_action_no_url(self, tmp_path: Path) -> None:
        log_file = tmp_path / "retrain.jsonl"
        config = RetrainConfig(
            action="webhook",
            webhook_url="",
            retrain_log_path=str(log_file),
        )
        event = trigger_retrain(config, current_psi=0.5)
        assert event is not None
        assert event.success is False
        assert "No webhook_url" in event.detail

    @patch("tobira.monitoring.retrain.urllib.request.urlopen")
    def test_webhook_action_success(
        self, mock_urlopen: MagicMock, tmp_path: Path,
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        log_file = tmp_path / "retrain.jsonl"
        config = RetrainConfig(
            action="webhook",
            webhook_url="https://example.com/hook",
            retrain_log_path=str(log_file),
        )
        event = trigger_retrain(config, current_psi=0.5)
        assert event is not None
        assert event.success is True
        assert "HTTP 200" in event.detail
        mock_urlopen.assert_called_once()

    @patch("tobira.monitoring.retrain.urllib.request.urlopen")
    def test_webhook_action_failure(
        self, mock_urlopen: MagicMock, tmp_path: Path,
    ) -> None:
        mock_urlopen.side_effect = Exception("Connection refused")

        log_file = tmp_path / "retrain.jsonl"
        config = RetrainConfig(
            action="webhook",
            webhook_url="https://example.com/hook",
            retrain_log_path=str(log_file),
        )
        event = trigger_retrain(config, current_psi=0.5)
        assert event is not None
        assert event.success is False
        assert "Connection refused" in event.detail

    def test_event_recorded_to_log(self, tmp_path: Path) -> None:
        log_file = tmp_path / "retrain.jsonl"
        config = RetrainConfig(
            action="log",
            retrain_log_path=str(log_file),
        )
        trigger_retrain(config, current_psi=0.5)
        trigger_retrain(config, current_psi=0.6)

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert "timestamp" in record
            assert "reason" in record
            assert record["action"] == "log"
            assert record["success"] is True


# ── lazy import tests ──────────────────────────────────────────


class TestRetrainLazyImport:
    def test_lazy_retrain_config_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "RetrainConfig")

    def test_lazy_retrain_event_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "RetrainEvent")

    def test_lazy_check_retrain_needed_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "check_retrain_needed")

    def test_lazy_load_retrain_config_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "load_retrain_config")

    def test_lazy_trigger_retrain_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "trigger_retrain")
