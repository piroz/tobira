"""Slack Incoming Webhook notifier."""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass

from tobira.monitoring.notifier import format_alert

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlackNotifier:
    """Send notifications via Slack Incoming Webhook.

    Attributes:
        webhook_url: Slack Incoming Webhook URL.
    """

    webhook_url: str

    def send(self, title: str, body: str, *, severity: str = "info") -> bool:
        """Send a notification to Slack.

        Args:
            title: Notification title.
            body: Notification body.
            severity: Severity level.

        Returns:
            ``True`` if the message was sent successfully.
        """
        if not self.webhook_url:
            logger.warning("Slack webhook_url not configured")
            return False

        alert = format_alert(title, body, severity=severity)
        text = (
            f"{alert['icon']} *[{alert['severity'].upper()}] "
            f"{alert['title']}*\n{alert['body']}"
        )
        payload = json.dumps({"text": text}).encode("utf-8")

        req = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                status: int = resp.status
                return 200 <= status < 300
        except Exception:  # noqa: BLE001
            logger.exception("Failed to send Slack notification")
            return False
