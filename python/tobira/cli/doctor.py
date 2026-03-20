"""tobira doctor - Validate configuration, backends, and connectivity."""

from __future__ import annotations

import argparse
import os
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


def _check_serving_deps() -> tuple[bool, str]:
    """Check that serving dependencies (fastapi, uvicorn) are importable."""
    missing: list[str] = []
    for pkg in ("fastapi", "uvicorn"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        return False, (
            f"serving dependencies not installed: {', '.join(missing)} "
            "(install with: pip install tobira[serving])"
        )
    return True, "serving dependencies available (fastapi, uvicorn)"


def _check_path_writable(path: str, label: str) -> tuple[bool, str]:
    """Check that a file path's parent directory exists and is writable.

    Args:
        path: The file path to check.
        label: Human-readable label for the check (e.g. "monitoring log_path").
    """
    p = Path(path)
    parent = p.parent
    if not parent.exists():
        return False, f"{label} parent directory does not exist: {parent}"
    if not os.access(parent, os.W_OK):
        return False, f"{label} parent directory is not writable: {parent}"
    return True, f"{label} path writable: {path}"


def _check_store(config: dict[str, Any]) -> list[tuple[bool, str]]:
    """Check store backend dependencies and connectivity.

    Validates import availability and attempts a connection test for
    PostgreSQL and Redis store backends configured in ``[store]``.
    """
    results: list[tuple[bool, str]] = []
    store_type = config.get("type", "jsonl")

    if store_type == "postgres":
        for pkg in ("psycopg", "psycopg_pool"):
            try:
                __import__(pkg)
            except ImportError:
                results.append((
                    False,
                    f"store dependency not installed: {pkg} "
                    "(install with: pip install tobira[postgres])",
                ))
                return results
        results.append((True, "store dependencies available (psycopg, psycopg_pool)"))

        dsn = config.get("dsn")
        if dsn:
            try:
                import psycopg

                conn = psycopg.connect(dsn, connect_timeout=5)
                conn.close()
                results.append((True, "PostgreSQL connection OK"))
            except Exception as exc:
                results.append((False, f"PostgreSQL connection failed: {exc}"))

    elif store_type == "redis":
        try:
            __import__("redis")
        except ImportError:
            results.append((
                False,
                "store dependency not installed: redis "
                "(install with: pip install tobira[redis])",
            ))
            return results
        results.append((True, "store dependency available (redis)"))

        redis_url = config.get("redis_url")
        if redis_url:
            try:
                import redis as redis_lib

                client = redis_lib.from_url(redis_url, socket_connect_timeout=5)
                client.ping()
                client.close()
                results.append((True, "Redis store connection OK"))
            except Exception as exc:
                results.append((False, f"Redis store connection failed: {exc}"))

    return results


def _check_monitoring_redis(redis_url: str) -> list[tuple[bool, str]]:
    """Check Redis availability for monitoring score accumulation."""
    results: list[tuple[bool, str]] = []
    try:
        __import__("redis")
    except ImportError:
        results.append((
            False,
            "monitoring redis dependency not installed: redis "
            "(install with: pip install tobira[redis])",
        ))
        return results
    results.append((True, "monitoring redis dependency available"))

    try:
        import redis as redis_lib

        client = redis_lib.from_url(redis_url, socket_connect_timeout=5)
        client.ping()
        client.close()
        results.append((True, "monitoring Redis connection OK"))
    except Exception as exc:
        results.append((False, f"monitoring Redis connection failed: {exc}"))

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

    # 5. Serving dependencies
    ok, msg = _check_serving_deps()
    results.append((ok, msg))

    # 6. Monitoring log_path writability
    if config is not None:
        monitoring = config.get("monitoring", {})
        if monitoring.get("enabled"):
            log_path = monitoring.get(
                "log_path", "/var/lib/tobira/predictions.jsonl"
            )
            ok, msg = _check_path_writable(log_path, "monitoring log_path")
            results.append((ok, msg))

            # 6b. Monitoring Redis
            redis_url = monitoring.get("redis_url")
            if redis_url:
                results.extend(_check_monitoring_redis(redis_url))

    # 7. Feedback store_path writability
    if config is not None:
        feedback = config.get("feedback", {})
        if feedback.get("enabled"):
            from tobira.data.feedback_store import DEFAULT_FEEDBACK_PATH

            store_path = feedback.get("store_path", DEFAULT_FEEDBACK_PATH)
            ok, msg = _check_path_writable(store_path, "feedback store_path")
            results.append((ok, msg))

    # 8. Store backend connection
    if config is not None:
        store_config = config.get("store")
        if store_config:
            results.extend(_check_store(store_config))

    # 9. MTA plugins
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
