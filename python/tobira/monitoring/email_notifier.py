"""SMTP email notifier."""

from __future__ import annotations

import logging
import smtplib
from dataclasses import dataclass
from email.message import EmailMessage

from tobira.monitoring.notifier import format_alert

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmailNotifier:
    """Send notifications via SMTP email.

    Attributes:
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port.
        from_addr: Sender email address.
        to_addrs: Recipient email addresses.
        use_tls: Whether to use STARTTLS.
        username: SMTP authentication username (empty to skip auth).
        password: SMTP authentication password.
    """

    smtp_host: str = "localhost"
    smtp_port: int = 25
    from_addr: str = ""
    to_addrs: tuple[str, ...] = ()
    use_tls: bool = False
    username: str = ""
    password: str = ""

    def __init__(
        self,
        *,
        smtp_host: str = "localhost",
        smtp_port: int = 25,
        from_addr: str = "",
        to_addrs: list[str] | tuple[str, ...] = (),
        use_tls: bool = False,
        username: str = "",
        password: str = "",
    ) -> None:
        object.__setattr__(self, "smtp_host", smtp_host)
        object.__setattr__(self, "smtp_port", smtp_port)
        object.__setattr__(self, "from_addr", from_addr)
        object.__setattr__(self, "to_addrs", tuple(to_addrs))
        object.__setattr__(self, "use_tls", use_tls)
        object.__setattr__(self, "username", username)
        object.__setattr__(self, "password", password)

    def send(self, title: str, body: str, *, severity: str = "info") -> bool:
        """Send a notification via email.

        Args:
            title: Notification title (used as email subject).
            body: Notification body.
            severity: Severity level.

        Returns:
            ``True`` if the email was sent successfully.
        """
        if not self.from_addr or not self.to_addrs:
            logger.warning("Email from_addr or to_addrs not configured")
            return False

        alert = format_alert(title, body, severity=severity)
        subject = f"[tobira {alert['severity'].upper()}] {alert['title']}"

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = ", ".join(self.to_addrs)
        msg.set_content(f"{alert['icon']} {alert['body']}")

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30) as smtp:
                if self.use_tls:
                    smtp.starttls()
                if self.username:
                    smtp.login(self.username, self.password)
                smtp.send_message(msg)
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Failed to send email notification")
            return False
