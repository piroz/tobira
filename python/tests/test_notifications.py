"""Tests for tobira.monitoring notification modules."""

from __future__ import annotations

import smtplib
from unittest.mock import MagicMock, patch

import pytest

from tobira.monitoring.analyzer import (
    AnalysisReport,
    PSIResult,
    notify_analysis_results,
)
from tobira.monitoring.email_notifier import EmailNotifier
from tobira.monitoring.notifier import (
    NotificationConfig,
    NotificationDispatcher,
    _resolve_env,
    create_dispatcher,
    format_alert,
    load_notification_config,
)
from tobira.monitoring.slack import SlackNotifier
from tobira.monitoring.teams import TeamsNotifier

# ── NotificationConfig tests ──────────────────────────────────


class TestNotificationConfig:
    def test_defaults(self) -> None:
        config = NotificationConfig()
        assert config.enabled is False
        assert config.cooldown_seconds == 300
        assert config.channels == ()

    def test_frozen(self) -> None:
        config = NotificationConfig()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore[misc]


class TestLoadNotificationConfig:
    def test_from_empty_dict(self) -> None:
        config = load_notification_config({})
        assert config.enabled is False
        assert config.cooldown_seconds == 300
        assert config.channels == ()

    def test_from_full_dict(self) -> None:
        raw = {
            "enabled": True,
            "cooldown_seconds": 600,
            "channels": [
                {"type": "slack", "webhook_url": "https://hooks.slack.com/x"},
                {"type": "email", "smtp_host": "smtp.example.com"},
            ],
        }
        config = load_notification_config(raw)
        assert config.enabled is True
        assert config.cooldown_seconds == 600
        assert len(config.channels) == 2
        assert config.channels[0]["type"] == "slack"

    def test_non_list_channels_ignored(self) -> None:
        config = load_notification_config({"channels": "invalid"})
        assert config.channels == ()

    def test_non_dict_channel_items_filtered(self) -> None:
        config = load_notification_config({"channels": ["invalid", {"type": "slack"}]})
        assert len(config.channels) == 1

    def test_unknown_keys_ignored(self) -> None:
        config = load_notification_config({"unknown_key": 42})
        assert config.enabled is False


# ── _resolve_env tests ────────────────────────────────────────


class TestResolveEnv:
    def test_plain_string(self) -> None:
        assert _resolve_env("hello") == "hello"

    def test_env_var_found(self) -> None:
        with patch.dict("os.environ", {"MY_VAR": "secret123"}):
            assert _resolve_env("${MY_VAR}") == "secret123"

    def test_env_var_not_found(self) -> None:
        assert _resolve_env("${NONEXISTENT_VAR_XYZ}") == ""


# ── format_alert tests ────────────────────────────────────────


class TestFormatAlert:
    def test_info(self) -> None:
        result = format_alert("Title", "Body", severity="info")
        assert result["title"] == "Title"
        assert result["body"] == "Body"
        assert result["severity"] == "info"

    def test_warning(self) -> None:
        result = format_alert("T", "B", severity="warning")
        assert result["severity"] == "warning"

    def test_critical(self) -> None:
        result = format_alert("T", "B", severity="critical")
        assert result["severity"] == "critical"

    def test_unknown_severity_uses_info_icon(self) -> None:
        result = format_alert("T", "B", severity="unknown")
        assert result["icon"] == "\u2139\ufe0f"


# ── NotificationDispatcher tests ──────────────────────────────


class TestNotificationDispatcher:
    def test_disabled_returns_empty(self) -> None:
        config = NotificationConfig(enabled=False)
        dispatcher = NotificationDispatcher(config=config)
        results = dispatcher.notify("title", "body")
        assert results == []

    def test_sends_to_all_notifiers(self) -> None:
        config = NotificationConfig(enabled=True)
        mock1 = MagicMock()
        mock1.send.return_value = True
        mock2 = MagicMock()
        mock2.send.return_value = True

        dispatcher = NotificationDispatcher(
            config=config, notifiers=[mock1, mock2],
        )
        results = dispatcher.notify("title", "body", severity="warning")
        assert results == [True, True]
        mock1.send.assert_called_once_with("title", "body", severity="warning")
        mock2.send.assert_called_once_with("title", "body", severity="warning")

    def test_cooldown_suppresses_duplicate(self) -> None:
        config = NotificationConfig(enabled=True, cooldown_seconds=300)
        mock_notifier = MagicMock()
        mock_notifier.send.return_value = True

        dispatcher = NotificationDispatcher(
            config=config, notifiers=[mock_notifier],
        )
        results1 = dispatcher.notify("same title", "body")
        results2 = dispatcher.notify("same title", "body")
        assert results1 == [True]
        assert results2 == []
        assert mock_notifier.send.call_count == 1

    def test_different_titles_not_suppressed(self) -> None:
        config = NotificationConfig(enabled=True, cooldown_seconds=300)
        mock_notifier = MagicMock()
        mock_notifier.send.return_value = True

        dispatcher = NotificationDispatcher(
            config=config, notifiers=[mock_notifier],
        )
        dispatcher.notify("title A", "body")
        dispatcher.notify("title B", "body")
        assert mock_notifier.send.call_count == 2

    def test_notifier_exception_caught(self) -> None:
        config = NotificationConfig(enabled=True)
        mock_notifier = MagicMock()
        mock_notifier.send.side_effect = RuntimeError("boom")

        dispatcher = NotificationDispatcher(
            config=config, notifiers=[mock_notifier],
        )
        results = dispatcher.notify("title", "body")
        assert results == [False]


# ── create_dispatcher tests ───────────────────────────────────


class TestCreateDispatcher:
    def test_creates_slack_notifier(self) -> None:
        config = NotificationConfig(
            enabled=True,
            channels=({"type": "slack", "webhook_url": "https://hooks.slack.com/x"},),
        )
        dispatcher = create_dispatcher(config)
        assert len(dispatcher.notifiers) == 1
        assert isinstance(dispatcher.notifiers[0], SlackNotifier)

    def test_creates_teams_notifier(self) -> None:
        config = NotificationConfig(
            enabled=True,
            channels=({"type": "teams", "webhook_url": "https://outlook.office.com/x"},),
        )
        dispatcher = create_dispatcher(config)
        assert len(dispatcher.notifiers) == 1
        assert isinstance(dispatcher.notifiers[0], TeamsNotifier)

    def test_creates_email_notifier(self) -> None:
        config = NotificationConfig(
            enabled=True,
            channels=({"type": "email", "smtp_host": "smtp.example.com",
                        "from_addr": "a@b.com", "to_addrs": ["c@d.com"]},),
        )
        dispatcher = create_dispatcher(config)
        assert len(dispatcher.notifiers) == 1
        assert isinstance(dispatcher.notifiers[0], EmailNotifier)

    def test_unknown_type_skipped(self) -> None:
        config = NotificationConfig(
            enabled=True,
            channels=({"type": "unknown"},),
        )
        dispatcher = create_dispatcher(config)
        assert len(dispatcher.notifiers) == 0

    def test_env_var_resolution(self) -> None:
        with patch.dict("os.environ", {"SLACK_URL": "https://resolved.url"}):
            config = NotificationConfig(
                enabled=True,
                channels=({"type": "slack", "webhook_url": "${SLACK_URL}"},),
            )
            dispatcher = create_dispatcher(config)
            assert isinstance(dispatcher.notifiers[0], SlackNotifier)
            assert dispatcher.notifiers[0].webhook_url == "https://resolved.url"

    def test_multiple_channels(self) -> None:
        config = NotificationConfig(
            enabled=True,
            channels=(
                {"type": "slack", "webhook_url": "https://hooks.slack.com/x"},
                {"type": "teams", "webhook_url": "https://outlook.office.com/x"},
            ),
        )
        dispatcher = create_dispatcher(config)
        assert len(dispatcher.notifiers) == 2


# ── SlackNotifier tests ───────────────────────────────────────


class TestSlackNotifier:
    def test_empty_url_returns_false(self) -> None:
        notifier = SlackNotifier(webhook_url="")
        assert notifier.send("title", "body") is False

    @patch("tobira.monitoring.slack.urllib.request.urlopen")
    def test_success(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        assert notifier.send("title", "body", severity="warning") is True
        mock_urlopen.assert_called_once()

    @patch("tobira.monitoring.slack.urllib.request.urlopen")
    def test_failure(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = Exception("Connection error")
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        assert notifier.send("title", "body") is False


# ── TeamsNotifier tests ───────────────────────────────────────


class TestTeamsNotifier:
    def test_empty_url_returns_false(self) -> None:
        notifier = TeamsNotifier(webhook_url="")
        assert notifier.send("title", "body") is False

    @patch("tobira.monitoring.teams.urllib.request.urlopen")
    def test_success(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        notifier = TeamsNotifier(webhook_url="https://outlook.office.com/test")
        assert notifier.send("title", "body", severity="critical") is True
        mock_urlopen.assert_called_once()

    @patch("tobira.monitoring.teams.urllib.request.urlopen")
    def test_failure(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = Exception("Timeout")
        notifier = TeamsNotifier(webhook_url="https://outlook.office.com/test")
        assert notifier.send("title", "body") is False


# ── EmailNotifier tests ──────────────────────────────────────


class TestEmailNotifier:
    def test_missing_from_addr_returns_false(self) -> None:
        notifier = EmailNotifier(to_addrs=["a@b.com"])
        assert notifier.send("title", "body") is False

    def test_missing_to_addrs_returns_false(self) -> None:
        notifier = EmailNotifier(from_addr="a@b.com")
        assert notifier.send("title", "body") is False

    @patch("tobira.monitoring.email_notifier.smtplib.SMTP")
    def test_success(self, mock_smtp_cls: MagicMock) -> None:
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        mock_smtp_cls.return_value = mock_smtp

        notifier = EmailNotifier(
            smtp_host="smtp.example.com",
            smtp_port=587,
            from_addr="sender@example.com",
            to_addrs=["recipient@example.com"],
            use_tls=True,
            username="user",
            password="pass",
        )
        assert notifier.send("title", "body", severity="warning") is True
        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once_with("user", "pass")
        mock_smtp.send_message.assert_called_once()

    @patch("tobira.monitoring.email_notifier.smtplib.SMTP")
    def test_no_tls_no_auth(self, mock_smtp_cls: MagicMock) -> None:
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        mock_smtp_cls.return_value = mock_smtp

        notifier = EmailNotifier(
            from_addr="sender@example.com",
            to_addrs=["recipient@example.com"],
        )
        assert notifier.send("title", "body") is True
        mock_smtp.starttls.assert_not_called()
        mock_smtp.login.assert_not_called()

    @patch("tobira.monitoring.email_notifier.smtplib.SMTP")
    def test_failure(self, mock_smtp_cls: MagicMock) -> None:
        mock_smtp_cls.side_effect = smtplib.SMTPException("Connection failed")
        notifier = EmailNotifier(
            from_addr="sender@example.com",
            to_addrs=["recipient@example.com"],
        )
        assert notifier.send("title", "body") is False

    def test_frozen(self) -> None:
        notifier = EmailNotifier(from_addr="a@b.com")
        with pytest.raises(AttributeError):
            notifier.from_addr = "x@y.com"  # type: ignore[misc]

    def test_to_addrs_stored_as_tuple(self) -> None:
        notifier = EmailNotifier(to_addrs=["a@b.com", "c@d.com"])
        assert isinstance(notifier.to_addrs, tuple)
        assert len(notifier.to_addrs) == 2


# ── notify_analysis_results tests ─────────────────────────────


class TestNotifyAnalysisResults:
    def test_drift_alert_sends_critical(self) -> None:
        report = AnalysisReport(
            total_records=100,
            period_start="2025-01-01",
            period_end="2025-01-31",
            psi=PSIResult(psi=0.3, level="alert"),
        )
        config = NotificationConfig(enabled=True, cooldown_seconds=0)
        mock_notifier = MagicMock()
        mock_notifier.send.return_value = True
        dispatcher = NotificationDispatcher(
            config=config, notifiers=[mock_notifier],
        )
        results = notify_analysis_results(report, dispatcher)
        assert len(results) == 1
        assert results[0] is True
        call_args = mock_notifier.send.call_args
        assert call_args.kwargs["severity"] == "critical"

    def test_drift_warning_sends_warning(self) -> None:
        report = AnalysisReport(
            total_records=100,
            period_start="2025-01-01",
            period_end="2025-01-31",
            psi=PSIResult(psi=0.15, level="warning"),
        )
        config = NotificationConfig(enabled=True, cooldown_seconds=0)
        mock_notifier = MagicMock()
        mock_notifier.send.return_value = True
        dispatcher = NotificationDispatcher(
            config=config, notifiers=[mock_notifier],
        )
        results = notify_analysis_results(report, dispatcher)
        assert len(results) == 1
        call_args = mock_notifier.send.call_args
        assert call_args.kwargs["severity"] == "warning"

    def test_no_issues_sends_nothing(self) -> None:
        report = AnalysisReport(
            total_records=100,
            period_start="2025-01-01",
            period_end="2025-01-31",
            psi=PSIResult(psi=0.01, level="ok"),
        )
        config = NotificationConfig(enabled=True)
        mock_notifier = MagicMock()
        dispatcher = NotificationDispatcher(
            config=config, notifiers=[mock_notifier],
        )
        results = notify_analysis_results(report, dispatcher)
        assert results == []

    def test_invalid_dispatcher_returns_empty(self) -> None:
        report = AnalysisReport(
            total_records=100,
            period_start="2025-01-01",
            period_end="2025-01-31",
        )
        results = notify_analysis_results(report, "not a dispatcher")
        assert results == []


# ── lazy import tests ─────────────────────────────────────────


class TestNotificationLazyImport:
    def test_lazy_notification_config_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "NotificationConfig")

    def test_lazy_notification_dispatcher_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "NotificationDispatcher")

    def test_lazy_create_dispatcher_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "create_dispatcher")

    def test_lazy_load_notification_config_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "load_notification_config")

    def test_lazy_slack_notifier_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "SlackNotifier")

    def test_lazy_teams_notifier_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "TeamsNotifier")

    def test_lazy_email_notifier_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "EmailNotifier")

    def test_lazy_notify_analysis_results_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "notify_analysis_results")
