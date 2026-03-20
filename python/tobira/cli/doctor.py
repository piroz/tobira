"""tobira doctor - Validate configuration, backends, and connectivity."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from urllib.request import urlopen

# Default MTA plugin config paths to check.
_MTA_PLUGIN_PATHS: dict[str, list[str]] = {
    "rspamd": [
        "/etc/rspamd/local.d/external_services.conf",
        "/etc/rspamd/override.d/external_services.conf",
    ],
    "spamassassin": [
        "/etc/spamassassin/local.cf",
        "/etc/mail/spamassassin/local.cf",
    ],
    "haraka": [
        "/etc/haraka/config/plugins",
    ],
}


def _check_config(config_path: str) -> tuple[bool, str, dict[str, Any] | None]:
    """Check that the config file exists and is valid TOML.

    Returns:
        Tuple of (ok, message, parsed_config_or_None).
    """
    try:
        from tobira.config import load_toml

        config = load_toml(config_path)
    except FileNotFoundError:
        return False, f"config file not found: {config_path}", None
    except Exception as exc:
        return False, f"invalid TOML syntax: {exc}", None
    return True, "configuration file OK", config


def _check_backend(config: dict[str, Any]) -> tuple[bool, str]:
    """Check that the backend section exists and the backend can be created."""
    if "backend" not in config:
        return False, "missing [backend] section in config"

    try:
        from tobira.backends.factory import create_backend

        create_backend(config["backend"])
    except Exception as exc:
        return False, f"backend initialisation failed: {exc}"
    return True, "backend loaded successfully"


def _check_api(api_url: str) -> tuple[bool, str]:
    """Check connectivity to the /health endpoint of a running API server."""
    try:
        url = api_url.rstrip("/") + "/health"
        with urlopen(url, timeout=5) as resp:
            if resp.status == 200:
                return True, f"API server reachable at {url}"
            return False, f"API server returned status {resp.status}"
    except Exception as exc:
        return False, f"API server unreachable: {exc}"


def _check_api_key(config: dict[str, Any]) -> tuple[bool, str]:
    """Check API key authentication configuration."""
    from tobira.serving.auth import get_api_key

    serving_config = config.get("serving")
    api_key = get_api_key(serving_config)
    if api_key is None:
        return False, "API key not configured (set [serving] api_key or TOBIRA_API_KEY)"
    if len(api_key) < 16:
        return False, "API key is too short (recommend at least 16 characters)"
    return True, "API key configured"


def _check_mta_plugins() -> list[tuple[bool, str]]:
    """Check existence of MTA plugin configuration files."""
    results: list[tuple[bool, str]] = []
    for name, paths in _MTA_PLUGIN_PATHS.items():
        found = any(Path(p).exists() for p in paths)
        if found:
            results.append((True, f"{name} config found"))
        else:
            results.append((False, f"{name} config not found"))
    return results


def run_checks(
    config_path: str,
    api_url: str | None = None,
) -> list[tuple[bool, str]]:
    """Run all doctor checks and return results.

    Args:
        config_path: Path to the TOML configuration file.
        api_url: Optional URL of a running API server to check.

    Returns:
        List of (passed, message) tuples.
    """
    results: list[tuple[bool, str]] = []

    # 1. Config file
    ok, msg, config = _check_config(config_path)
    results.append((ok, msg))

    # 2. Backend
    if config is not None:
        ok, msg = _check_backend(config)
        results.append((ok, msg))

    # 3. API key
    if config is not None:
        ok, msg = _check_api_key(config)
        results.append((ok, msg))

    # 4. API server (optional)
    if api_url is not None:
        ok, msg = _check_api(api_url)
        results.append((ok, msg))

    # 5. MTA plugins
    results.extend(_check_mta_plugins())

    return results


def register(subparsers: "argparse._SubParsersAction[Any]") -> None:
    """Register the ``doctor`` subcommand.

    Args:
        subparsers: Subparser action from the parent parser.
    """
    parser = subparsers.add_parser(
        "doctor",
        help="Validate configuration, backends, and connectivity",
        description="Run diagnostic checks for tobira setup.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="URL of a running API server to check (e.g. http://localhost:8000)",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> int:
    """Execute the ``doctor`` subcommand.

    Args:
        args: Parsed arguments from argparse.

    Returns:
        Exit code (0 if all checks pass, 1 otherwise).
    """
    results = run_checks(config_path=args.config, api_url=args.api_url)

    for ok, msg in results:
        mark = "\u2705" if ok else "\u274c"
        print(f"  {mark}  {msg}")

    all_ok = all(ok for ok, _ in results)
    if all_ok:
        print("\nAll checks passed.")
    else:
        print("\nSome checks failed.")

    # Record doctor run for telemetry metrics when enabled.
    try:
        from tobira.config import load_toml
        from tobira.telemetry import TelemetryCollector, TelemetryConfig

        config = load_toml(args.config)
        tel_cfg = TelemetryConfig.from_dict(config.get("telemetry", {}))
        if tel_cfg.enabled:
            collector = TelemetryCollector(tel_cfg)
            collector.record_doctor_run(results)
    except Exception:
        pass  # telemetry must never break the CLI

    return 0 if all_ok else 1
