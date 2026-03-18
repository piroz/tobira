"""Notification dispatcher for monitoring alerts.

Provides a protocol for notification backends and a dispatcher that fans out
alerts to multiple configured channels (Slack, Teams, email).  Each notifier
supports cooldown to suppress duplicate alerts within a configurable window.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Notifier(Protocol):
    """Protocol for notification backends."""

    def send(self, title: str, body: str, *, severity: str = "info") -> bool:
        """Send a notification.

        Args:
            title: Short notification title.
            body: Notification body text.
            severity: One of ``"info"``, ``"warning"``, ``"critical"``.

        Returns:
            ``True`` if the notification was sent successfully.
        """
        ...  # pragma: no cover


@dataclass(frozen=True)
class NotificationConfig:
    """Configuration for the notification system.

    Attributes:
        enabled: Whether notifications are active.
        cooldown_seconds: Minimum seconds between identical notifications.
        channels: List of channel configuration dicts.
    """

    enabled: bool = False
    cooldown_seconds: int = 300
    channels: tuple[dict[str, Any], ...] = ()


def load_notification_config(raw: dict[str, Any]) -> NotificationConfig:
    """Create a :class:`NotificationConfig` from a raw dict (e.g. TOML section).

    Unknown keys are silently ignored.

    Args:
        raw: Dictionary of configuration values.

    Returns:
        A validated :class:`NotificationConfig`.
    """
    channels_raw = raw.get("channels", [])
    if not isinstance(channels_raw, list):
        channels_raw = []
    return NotificationConfig(
        enabled=bool(raw.get("enabled", False)),
        cooldown_seconds=int(raw.get("cooldown_seconds", 300)),
        channels=tuple(
            dict(ch) for ch in channels_raw if isinstance(ch, dict)
        ),
    )


def _resolve_env(value: str) -> str:
    """Resolve ``${ENV_VAR}`` references in a string value."""
    if value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1]
        return os.environ.get(env_name, "")
    return value


@dataclass
class NotificationDispatcher:
    """Dispatches notifications to multiple channels with cooldown.

    Attributes:
        config: Notification configuration.
        notifiers: List of instantiated notifier backends.
    """

    config: NotificationConfig
    notifiers: list[Notifier] = field(default_factory=list)
    _cooldown_cache: dict[str, float] = field(
        default_factory=dict, repr=False,
    )

    def notify(
        self,
        title: str,
        body: str,
        *,
        severity: str = "info",
    ) -> list[bool]:
        """Send a notification to all configured channels.

        Respects cooldown: if the same ``title`` was sent within the
        cooldown window, the notification is suppressed.

        Args:
            title: Short notification title.
            body: Notification body text.
            severity: One of ``"info"``, ``"warning"``, ``"critical"``.

        Returns:
            List of success booleans, one per notifier.  Empty if
            notifications are disabled or suppressed by cooldown.
        """
        if not self.config.enabled:
            return []

        now = time.monotonic()
        last_sent = self._cooldown_cache.get(title)
        if last_sent is not None:
            elapsed = now - last_sent
            if elapsed < self.config.cooldown_seconds:
                logger.debug(
                    "Notification suppressed by cooldown: %s (%.0fs remaining)",
                    title,
                    self.config.cooldown_seconds - elapsed,
                )
                return []

        results: list[bool] = []
        for notifier in self.notifiers:
            try:
                ok = notifier.send(title, body, severity=severity)
                results.append(ok)
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to send notification via %s",
                    type(notifier).__name__,
                )
                results.append(False)

        if results:
            self._cooldown_cache[title] = now

        return results


def create_dispatcher(config: NotificationConfig) -> NotificationDispatcher:
    """Create a :class:`NotificationDispatcher` from configuration.

    Instantiates the appropriate notifier backends based on the channel
    configurations.

    Args:
        config: Notification configuration.

    Returns:
        A configured :class:`NotificationDispatcher`.
    """
    notifiers: list[Notifier] = []

    for ch in config.channels:
        ch_type = str(ch.get("type", ""))
        if ch_type == "slack":
            from tobira.monitoring.slack import SlackNotifier

            webhook_url = _resolve_env(str(ch.get("webhook_url", "")))
            notifiers.append(SlackNotifier(webhook_url=webhook_url))
        elif ch_type == "teams":
            from tobira.monitoring.teams import TeamsNotifier

            webhook_url = _resolve_env(str(ch.get("webhook_url", "")))
            notifiers.append(TeamsNotifier(webhook_url=webhook_url))
        elif ch_type == "email":
            from tobira.monitoring.email_notifier import EmailNotifier

            notifiers.append(EmailNotifier(
                smtp_host=str(ch.get("smtp_host", "localhost")),
                smtp_port=int(ch.get("smtp_port", 25)),
                from_addr=str(ch.get("from_addr", "")),
                to_addrs=[str(a) for a in ch.get("to_addrs", [])],
                use_tls=bool(ch.get("use_tls", False)),
                username=_resolve_env(str(ch.get("username", ""))),
                password=_resolve_env(str(ch.get("password", ""))),
            ))
        else:
            logger.warning("Unknown notification channel type: %s", ch_type)

    return NotificationDispatcher(config=config, notifiers=notifiers)


def format_alert(
    title: str,
    body: str,
    *,
    severity: str = "info",
) -> dict[str, str]:
    """Format an alert into a structured dict for notification payloads.

    Args:
        title: Alert title.
        body: Alert body text.
        severity: Severity level.

    Returns:
        A dict with ``title``, ``body``, ``severity``, and ``icon`` keys.
    """
    icons = {
        "info": "\u2139\ufe0f",
        "warning": "\u26a0\ufe0f",
        "critical": "\U0001f6a8",
    }
    return {
        "title": title,
        "body": body,
        "severity": severity,
        "icon": icons.get(severity, "\u2139\ufe0f"),
    }
