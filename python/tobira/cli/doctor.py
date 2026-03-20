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


def _check_ab_test(config: dict[str, Any]) -> list[tuple[bool, str]]:
    """Check A/B test configuration when enabled."""
    results: list[tuple[bool, str]] = []
    ab_test = config.get("ab_test", {})
    if not ab_test.get("enabled"):
        return results

    variants = ab_test.get("variants")
    if not variants or not isinstance(variants, list):
        results.append((False, "[ab_test] no variants configured"))
        return results

    names = [v.get("name", "") for v in variants if isinstance(v, dict)]
    if len(names) != len(set(names)):
        results.append((False, "[ab_test] variant names are not unique"))
    else:
        results.append((True, f"[ab_test] {len(names)} variant(s) with unique names"))

    total_weight = sum(
        v.get("weight", 1.0) for v in variants if isinstance(v, dict)
    )
    if total_weight <= 0:
        results.append((False, "[ab_test] total variant weight must be positive"))
    else:
        results.append((True, f"[ab_test] total weight: {total_weight}"))

    for v in variants:
        if not isinstance(v, dict):
            continue
        vname = v.get("name", "<unnamed>")
        backend_cfg = v.get("backend")
        if not backend_cfg:
            results.append((
                False,
                f"[ab_test] variant '{vname}' has no backend config",
            ))
            continue
        try:
            from tobira.backends.factory import create_backend

            create_backend(backend_cfg)
            results.append((True, f"[ab_test] variant '{vname}' backend OK"))
        except Exception as exc:
            results.append((
                False,
                f"[ab_test] variant '{vname}' backend failed: {exc}",
            ))

    return results


def _check_active_learning(config: dict[str, Any]) -> list[tuple[bool, str]]:
    """Check Active Learning configuration when enabled."""
    results: list[tuple[bool, str]] = []
    al = config.get("active_learning", {})
    if not al.get("enabled"):
        return results

    queue_path = al.get(
        "queue_path", "/var/lib/tobira/active_learning_queue.jsonl"
    )
    ok, msg = _check_path_writable(queue_path, "[active_learning] queue_path")
    results.append((ok, msg))

    strategy = al.get("strategy", "entropy")
    valid_strategies = {"entropy", "margin", "least_confidence"}
    if strategy not in valid_strategies:
        results.append((
            False,
            f"[active_learning] invalid strategy '{strategy}' "
            f"(must be one of {sorted(valid_strategies)})",
        ))
    else:
        results.append((True, f"[active_learning] strategy: {strategy}"))

    threshold = al.get("uncertainty_threshold", 0.3)
    if not isinstance(threshold, (int, float)) or threshold < 0 or threshold > 1:
        results.append((
            False,
            f"[active_learning] uncertainty_threshold must be between 0 and 1, "
            f"got {threshold}",
        ))
    else:
        results.append((True, f"[active_learning] uncertainty_threshold: {threshold}"))

    max_queue_size = al.get("max_queue_size", 1000)
    if not isinstance(max_queue_size, int) or max_queue_size <= 0:
        results.append((
            False,
            f"[active_learning] max_queue_size must be a positive integer, "
            f"got {max_queue_size}",
        ))
    else:
        results.append((True, f"[active_learning] max_queue_size: {max_queue_size}"))

    return results


def _check_notification(config: dict[str, Any]) -> list[tuple[bool, str]]:
    """Check notification channel configuration."""
    results: list[tuple[bool, str]] = []
    notification = config.get("notification", {})
    if not notification.get("enabled"):
        return results

    channels = notification.get("channels", [])
    if not channels:
        results.append((False, "[notification] enabled but no channels configured"))
        return results

    for i, ch in enumerate(channels):
        if not isinstance(ch, dict):
            continue
        ch_type = ch.get("type", "")
        prefix = f"[notification] channel[{i}] ({ch_type})"

        if ch_type in ("slack", "teams"):
            webhook_url = ch.get("webhook_url", "")
            if not webhook_url:
                results.append((False, f"{prefix} webhook_url is empty"))
            elif (
                isinstance(webhook_url, str)
                and webhook_url.startswith("${")
                and webhook_url.endswith("}")
            ):
                env_name = webhook_url[2:-1]
                if env_name not in os.environ:
                    results.append((
                        False,
                        f"{prefix} env var {env_name} not set "
                        f"(referenced as {webhook_url})",
                    ))
                else:
                    results.append((True, f"{prefix} webhook_url OK"))
            else:
                results.append((True, f"{prefix} webhook_url OK"))
        elif ch_type == "email":
            missing: list[str] = []
            for field in ("smtp_host", "from_addr", "to_addrs"):
                val = ch.get(field)
                if not val:
                    missing.append(field)
                elif field == "to_addrs" and isinstance(val, list) and len(val) == 0:
                    missing.append(field)
            if missing:
                results.append((
                    False,
                    f"{prefix} missing required fields: {', '.join(missing)}",
                ))
            else:
                results.append((True, f"{prefix} config OK"))

            # Check env var references in email fields
            for field in ("username", "password"):
                val = ch.get(field, "")
                if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
                    env_name = val[2:-1]
                    if env_name not in os.environ:
                        results.append((
                            False,
                            f"{prefix} env var {env_name} not set "
                            f"(referenced in {field} as {val})",
                        ))
        else:
            if ch_type:
                results.append((False, f"{prefix} unknown channel type"))

    return results


def _check_telemetry_storage(config: dict[str, Any]) -> tuple[bool, str]:
    """Check that telemetry storage_dir is writable when telemetry is enabled."""
    telemetry = config.get("telemetry", {})
    if not telemetry.get("enabled"):
        return True, "[telemetry] disabled, skipping storage_dir check"

    storage_dir = telemetry.get("storage_dir", "/var/lib/tobira/telemetry")
    p = Path(storage_dir)
    if not p.exists():
        return False, f"[telemetry] storage_dir does not exist: {storage_dir}"
    if not os.access(p, os.W_OK):
        return False, f"[telemetry] storage_dir is not writable: {storage_dir}"
    return True, f"[telemetry] storage_dir writable: {storage_dir}"


def _check_header_analysis_weight(config: dict[str, Any]) -> tuple[bool, str]:
    """Check that header_analysis weight is in 0.0-1.0 range when enabled."""
    ha = config.get("header_analysis", {})
    if not ha.get("enabled"):
        return True, "[header_analysis] disabled, skipping weight check"

    weight = ha.get("weight", 0.3)
    if not isinstance(weight, (int, float)) or weight < 0.0 or weight > 1.0:
        return False, (
            f"[header_analysis] weight must be between 0.0 and 1.0, got {weight}"
        )
    return True, f"[header_analysis] weight: {weight}"


def _check_ai_detection_threshold(config: dict[str, Any]) -> tuple[bool, str]:
    """Check that ai_detection threshold is in 0.0-1.0 range when enabled."""
    ai = config.get("ai_detection", {})
    if not ai.get("enabled"):
        return True, "[ai_detection] disabled, skipping threshold check"

    threshold = ai.get("threshold", 0.65)
    if not isinstance(threshold, (int, float)) or threshold < 0.0 or threshold > 1.0:
        return False, (
            f"[ai_detection] threshold must be between 0.0 and 1.0, got {threshold}"
        )
    return True, f"[ai_detection] threshold: {threshold}"


def _check_model_path(config: dict[str, Any]) -> list[tuple[bool, str]]:
    """Check backend model file existence with specific error messages.

    For fasttext and onnx backends, validates that ``model_path`` exists
    before the backend is initialised, providing a targeted error message
    instead of the generic 'backend initialisation failed' error.
    """
    results: list[tuple[bool, str]] = []
    backend = config.get("backend", {})
    backend_type = backend.get("backend", "")
    model_path = backend.get("model_path", "")

    if backend_type == "fasttext" and model_path:
        if not Path(model_path).exists():
            results.append((
                False,
                f"[backend] fasttext model file not found: {model_path}",
            ))
        else:
            results.append((
                True, f"[backend] fasttext model file exists: {model_path}"
            ))

    elif backend_type == "onnx" and model_path:
        if not Path(model_path).exists():
            results.append((
                False,
                f"[backend] ONNX model file not found: {model_path}",
            ))
        else:
            results.append((True, f"[backend] ONNX model file exists: {model_path}"))

    return results


def _check_retrain(config: dict[str, Any]) -> list[tuple[bool, str]]:
    """Check retrain trigger configuration."""
    results: list[tuple[bool, str]] = []
    retrain = config.get("retrain", {})
    if not retrain.get("enabled", True):
        return results

    # Only validate if the section is explicitly present
    if "retrain" not in config:
        return results

    action = retrain.get("action", "log")
    valid_actions = {"log", "script", "webhook"}
    if action not in valid_actions:
        results.append((
            False,
            f"[retrain] invalid action '{action}' "
            f"(must be one of {sorted(valid_actions)})",
        ))
        return results
    results.append((True, f"[retrain] action: {action}"))

    if action == "script":
        script_path = retrain.get("script_path", "")
        if not script_path:
            results.append((
                False,
                "[retrain] action is 'script' but script_path is empty",
            ))
        else:
            p = Path(script_path)
            if not p.exists():
                results.append((
                    False,
                    f"[retrain] script_path not found: {script_path}",
                ))
            elif not os.access(p, os.X_OK):
                results.append((
                    False,
                    f"[retrain] script_path not executable: {script_path}",
                ))
            else:
                results.append((True, f"[retrain] script_path OK: {script_path}"))

    elif action == "webhook":
        webhook_url = retrain.get("webhook_url", "")
        if not webhook_url:
            results.append((
                False,
                "[retrain] action is 'webhook' but webhook_url is empty",
            ))
        else:
            results.append((True, "[retrain] webhook_url configured"))

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

    # 2. Backend model file existence (before full backend init)
    if config is not None:
        results.extend(_check_model_path(config))

    # 3. Backend
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

    # 9. A/B test configuration
    if config is not None:
        results.extend(_check_ab_test(config))

    # 10. Active Learning configuration
    if config is not None:
        results.extend(_check_active_learning(config))

    # 11. Notification channels
    if config is not None:
        results.extend(_check_notification(config))

    # 12. Retrain trigger configuration
    if config is not None:
        results.extend(_check_retrain(config))

    # 13. Telemetry storage_dir writability
    if config is not None:
        ok, msg = _check_telemetry_storage(config)
        results.append((ok, msg))

    # 14. Header analysis weight range
    if config is not None:
        ok, msg = _check_header_analysis_weight(config)
        results.append((ok, msg))

    # 15. AI detection threshold range
    if config is not None:
        ok, msg = _check_ai_detection_threshold(config)
        results.append((ok, msg))

    # 16. MTA plugins
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
