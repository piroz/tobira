"""tobira.milter.filter - Milter filter implementation.

Implements the milter protocol handler that receives mail from Postfix,
sends it to the tobira API for classification, and applies actions based
on the spam score.
"""

from __future__ import annotations

import email
import logging
from typing import Any

logger = logging.getLogger(__name__)


def _import_milter() -> Any:
    """Import the Milter library with a helpful error message.

    Returns:
        The Milter module.

    Raises:
        ImportError: If pymilter is not installed.
    """
    try:
        import Milter
    except ImportError:
        msg = (
            "pymilter is required for the milter integration. "
            "Install it with: pip install tobira[milter]"
        )
        raise ImportError(msg) from None
    return Milter


def _import_urllib() -> Any:
    """Import urllib modules for HTTP requests.

    Returns:
        Tuple of (urllib.request, urllib.error, json).
    """
    import json
    import urllib.error
    import urllib.request

    return urllib.request, urllib.error, json


class TobiraFilter:
    """Milter filter that classifies mail via the tobira API.

    This class is instantiated per connection by the milter framework.
    It collects mail headers and body, sends the text to the tobira API,
    and applies actions (reject, header addition) based on the score.

    Attributes:
        _api_url: URL of the tobira API server.
        _timeout: Connection timeout in seconds.
        _fail_action: Action on API failure ('accept' or 'tempfail').
        _reject_threshold: Score threshold for rejection.
        _add_headers: Whether to add classification headers.
    """

    # Class-level configuration set by configure()
    _api_url: str = "http://127.0.0.1:8000"
    _timeout: int = 10
    _fail_action: str = "accept"
    _reject_threshold: float = 0.9
    _add_headers: bool = True

    def __init__(self) -> None:
        self._subject: str = ""
        self._body_chunks: list[bytes] = []

    @classmethod
    def configure(
        cls,
        api_url: str,
        timeout: int,
        fail_action: str,
        reject_threshold: float,
        add_headers: bool,
    ) -> None:
        """Set class-level configuration for all filter instances.

        Args:
            api_url: URL of the tobira API server.
            timeout: Connection timeout in seconds.
            fail_action: Action on API failure.
            reject_threshold: Score threshold for rejection.
            add_headers: Whether to add classification headers.
        """
        cls._api_url = api_url
        cls._timeout = timeout
        cls._fail_action = fail_action
        cls._reject_threshold = reject_threshold
        cls._add_headers = add_headers

    def header(self, name: str, value: str) -> int:
        """Called for each header in the message.

        Args:
            name: Header field name.
            value: Header field value.

        Returns:
            Milter.CONTINUE to keep processing.
        """
        Milter = _import_milter()
        if name.lower() == "subject":
            self._subject = value
        return int(Milter.CONTINUE)

    def body(self, chunk: bytes) -> int:
        """Called for each body chunk.

        Args:
            chunk: A chunk of the message body.

        Returns:
            Milter.CONTINUE to keep processing.
        """
        Milter = _import_milter()
        self._body_chunks.append(chunk)
        return int(Milter.CONTINUE)

    def eom(self) -> int:
        """Called at end of message. Performs classification.

        Sends the collected message to the tobira API and applies
        actions based on the classification result.

        Returns:
            Milter action code (ACCEPT, REJECT, or TEMPFAIL).
        """
        Milter = _import_milter()
        urllib_request, urllib_error, json = _import_urllib()

        raw_body = b"".join(self._body_chunks)
        text = self._extract_text(raw_body)

        # Combine subject and body for classification
        full_text = f"{self._subject}\n{text}" if self._subject else text

        if not full_text.strip():
            return int(Milter.ACCEPT)

        try:
            result = self._call_api(full_text, urllib_request, urllib_error, json)
        except Exception:
            logger.exception("Failed to call tobira API")
            if self._fail_action == "tempfail":
                return int(Milter.TEMPFAIL)
            return int(Milter.ACCEPT)

        label = result.get("label", "ham")
        score = result.get("score", 0.0)

        # Add headers if configured
        if self._add_headers:
            self.addheader("X-Tobira-Score", f"{score:.4f}")  # type: ignore[attr-defined]
            self.addheader("X-Tobira-Label", label)  # type: ignore[attr-defined]

        # Reject if score exceeds threshold
        if self._reject_threshold > 0 and score >= self._reject_threshold:
            self.setreply("550", "5.7.1", "Message rejected as spam")  # type: ignore[attr-defined]
            logger.info(
                "Rejected message: score=%.4f label=%s subject=%s",
                score,
                label,
                self._subject[:50],
            )
            return int(Milter.REJECT)

        return int(Milter.ACCEPT)

    def close(self) -> int:
        """Called when the connection is closed. Cleans up state.

        Returns:
            Milter.CONTINUE.
        """
        Milter = _import_milter()
        self._subject = ""
        self._body_chunks = []
        return int(Milter.CONTINUE)

    def _extract_text(self, raw_body: bytes) -> str:
        """Extract plain text from raw email body.

        Parses MIME structure and extracts text/plain parts.

        Args:
            raw_body: Raw bytes of the email body.

        Returns:
            Extracted plain text content.
        """
        try:
            msg = email.message_from_bytes(raw_body)
        except Exception:
            # Fall back to raw decoding
            return raw_body.decode("utf-8", errors="replace")

        parts: list[str] = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    if isinstance(payload, bytes):
                        charset = part.get_content_charset() or "utf-8"
                        parts.append(payload.decode(charset, errors="replace"))
        else:
            payload = msg.get_payload(decode=True)
            if isinstance(payload, bytes):
                charset = msg.get_content_charset() or "utf-8"
                parts.append(payload.decode(charset, errors="replace"))

        return "\n".join(parts)

    def _call_api(
        self,
        text: str,
        urllib_request: Any,
        urllib_error: Any,
        json: Any,
    ) -> dict[str, Any]:
        """Call the tobira API /predict endpoint.

        Args:
            text: Text to classify.
            urllib_request: urllib.request module.
            urllib_error: urllib.error module.
            json: json module.

        Returns:
            API response as a dictionary.

        Raises:
            urllib.error.URLError: On connection failure.
        """
        url = f"{self._api_url.rstrip('/')}/predict"
        data = json.dumps({"text": text}).encode("utf-8")

        req = urllib_request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib_request.urlopen(req, timeout=self._timeout) as resp:
            body = resp.read()
            return json.loads(body)  # type: ignore[no-any-return]


def run_milter(
    socket_spec: str,
    api_url: str = "http://127.0.0.1:8000",
    timeout: int = 10,
    fail_action: str = "accept",
    reject_threshold: float = 0.9,
    add_headers: bool = True,
) -> None:
    """Start the milter daemon.

    Args:
        socket_spec: Socket to listen on (unix:/path or inet:port@host).
        api_url: URL of the tobira API server.
        timeout: Connection timeout in seconds.
        fail_action: Action on API failure ('accept' or 'tempfail').
        reject_threshold: Score threshold for rejection.
        add_headers: Whether to add classification headers.
    """
    Milter = _import_milter()

    TobiraFilter.configure(
        api_url=api_url,
        timeout=timeout,
        fail_action=fail_action,
        reject_threshold=reject_threshold,
        add_headers=add_headers,
    )

    logger.info("Starting tobira milter on %s", socket_spec)
    logger.info("API URL: %s, timeout: %ds", api_url, timeout)
    logger.info(
        "Fail action: %s, reject threshold: %.2f",
        fail_action, reject_threshold,
    )

    Milter.factory = TobiraFilter

    # Parse socket specification
    if socket_spec.startswith("unix:"):
        sock_path = socket_spec[5:]
        Milter.runmilter("tobira-milter", socketname=sock_path)
    elif socket_spec.startswith("inet:"):
        Milter.runmilter("tobira-milter", socketname=socket_spec)
    else:
        msg = (
            f"invalid socket spec: {socket_spec!r} "
            "(must start with 'unix:' or 'inet:')"
        )
        raise ValueError(msg)
