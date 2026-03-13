"""tobira.cli.mta_detectors - Detect running MTA services."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectedMTA:
    """Represents a detected MTA service.

    Attributes:
        name: MTA identifier (rspamd, spamassassin, haraka).
        source: How it was detected (process, systemd, config_path).
    """

    name: str
    source: str


# Mapping of MTA name to process names to search for.
_PROCESS_NAMES: dict[str, list[str]] = {
    "rspamd": ["rspamd"],
    "spamassassin": ["spamd"],
    "haraka": ["haraka"],
}

# Mapping of MTA name to systemd unit names.
_SYSTEMD_UNITS: dict[str, list[str]] = {
    "rspamd": ["rspamd.service"],
    "spamassassin": ["spamassassin.service", "spamd.service"],
    "haraka": ["haraka.service"],
}

# Mapping of MTA name to config directories.
_CONFIG_DIRS: dict[str, list[str]] = {
    "rspamd": ["/etc/rspamd"],
    "spamassassin": ["/etc/spamassassin", "/etc/mail/spamassassin"],
    "haraka": ["/etc/haraka"],
}


def _check_process(mta: str) -> bool:
    """Check if any process matching the MTA is running."""
    for proc_name in _PROCESS_NAMES.get(mta, []):
        if shutil.which(proc_name) is not None:
            return True
        try:
            result = subprocess.run(
                ["pgrep", "-x", proc_name],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def _check_systemd(mta: str) -> bool:
    """Check if a systemd unit for the MTA exists."""
    for unit in _SYSTEMD_UNITS.get(mta, []):
        try:
            result = subprocess.run(
                ["systemctl", "list-unit-files", unit, "--no-legend"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and unit in result.stdout:
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    return False


def _check_config_dir(mta: str) -> bool:
    """Check if a configuration directory for the MTA exists."""
    for dir_path in _CONFIG_DIRS.get(mta, []):
        if Path(dir_path).is_dir():
            return True
    return False


def detect_mtas() -> list[DetectedMTA]:
    """Detect installed MTA services.

    Checks process lists, systemd units, and configuration directories
    for known MTA services (rspamd, spamassassin, haraka).

    Returns:
        List of detected MTA services with detection source.
    """
    detected: list[DetectedMTA] = []
    seen: set[str] = set()

    for mta in _PROCESS_NAMES:
        if _check_process(mta) and mta not in seen:
            detected.append(DetectedMTA(name=mta, source="process"))
            seen.add(mta)
            continue

        if _check_systemd(mta) and mta not in seen:
            detected.append(DetectedMTA(name=mta, source="systemd"))
            seen.add(mta)
            continue

        if _check_config_dir(mta) and mta not in seen:
            detected.append(DetectedMTA(name=mta, source="config_path"))
            seen.add(mta)

    return detected
