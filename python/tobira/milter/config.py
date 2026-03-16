"""tobira.milter.config - Configuration for the milter daemon."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MilterConfig:
    """Configuration for the tobira milter daemon.

    Attributes:
        api_url: URL of the tobira API server.
        socket: Socket address to listen on (unix:/path or inet:port@host).
        timeout: Connection timeout to tobira API in seconds.
        fail_action: Action when API is unavailable ('accept' or 'tempfail').
        reject_threshold: Reject mail if spam score >= this value (0 to disable).
        add_headers: Whether to add X-Tobira-Score/X-Tobira-Label headers.
    """

    api_url: str = "http://127.0.0.1:8000"
    socket: str = "unix:/var/run/tobira/milter.sock"
    timeout: int = 10
    fail_action: str = "accept"
    reject_threshold: float = 0.9
    add_headers: bool = True


def load_config(config_path: str | Path) -> MilterConfig:
    """Load milter configuration from an INI file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Parsed MilterConfig.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the configuration contains invalid values.
    """
    path = Path(config_path)
    if not path.exists():
        msg = f"configuration file not found: {path}"
        raise FileNotFoundError(msg)

    parser = configparser.ConfigParser()
    parser.read(path, encoding="utf-8")

    section = "milter"
    if not parser.has_section(section):
        return MilterConfig()

    api_url = parser.get(section, "api_url", fallback="http://127.0.0.1:8000")
    socket = parser.get(section, "socket", fallback="unix:/var/run/tobira/milter.sock")
    timeout = parser.getint(section, "timeout", fallback=10)
    fail_action = parser.get(section, "fail_action", fallback="accept")
    reject_threshold = parser.getfloat(section, "reject_threshold", fallback=0.9)
    add_headers = parser.getboolean(section, "add_headers", fallback=True)

    if fail_action not in ("accept", "tempfail"):
        msg = f"fail_action must be 'accept' or 'tempfail', got: {fail_action!r}"
        raise ValueError(msg)

    if not (0.0 <= reject_threshold <= 1.0):
        msg = f"reject_threshold must be between 0.0 and 1.0, got: {reject_threshold}"
        raise ValueError(msg)

    return MilterConfig(
        api_url=api_url,
        socket=socket,
        timeout=timeout,
        fail_action=fail_action,
        reject_threshold=reject_threshold,
        add_headers=add_headers,
    )
