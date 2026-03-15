"""Automatic retrain trigger based on drift, feedback count, and time.

Monitors prediction quality signals and fires a retrain action when
configurable thresholds are exceeded.  The trigger itself does **not**
execute training — it records the event and notifies via one of the
supported actions (log / script / webhook).
"""

from __future__ import annotations

import json
import logging
import subprocess
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from tobira.monitoring.store import append_record, read_records

logger = logging.getLogger(__name__)

DEFAULT_RETRAIN_LOG = "/var/lib/tobira/retrain.jsonl"

_ACTION_LOG = "log"
_ACTION_SCRIPT = "script"
_ACTION_WEBHOOK = "webhook"
_VALID_ACTIONS = {_ACTION_LOG, _ACTION_SCRIPT, _ACTION_WEBHOOK}


@dataclass(frozen=True)
class RetrainConfig:
    """Configuration for the retrain trigger.

    Attributes:
        enabled: Whether the retrain trigger is active.
        drift_threshold: PSI value above which drift triggers retraining.
        min_feedback_count: Minimum new feedback records to trigger retraining.
        max_interval_days: Maximum days since last retrain before triggering.
        action: Action to take when retraining is triggered.
        script_path: Path to the script to execute (for ``"script"`` action).
        webhook_url: URL to POST to (for ``"webhook"`` action).
        retrain_log_path: JSONL file recording retrain events.
    """

    enabled: bool = True
    drift_threshold: float = 0.1
    min_feedback_count: int = 100
    max_interval_days: int = 30
    action: str = _ACTION_LOG
    script_path: str = ""
    webhook_url: str = ""
    retrain_log_path: str = DEFAULT_RETRAIN_LOG


@dataclass(frozen=True)
class RetrainEvent:
    """A recorded retrain trigger event.

    Attributes:
        timestamp: ISO-8601 UTC timestamp of the event.
        reason: Human-readable reason the trigger fired.
        action: Action that was executed.
        success: Whether the action completed without error.
        detail: Additional information (e.g. script output or error message).
    """

    timestamp: str
    reason: str
    action: str
    success: bool
    detail: str = ""


def load_retrain_config(raw: dict[str, Any]) -> RetrainConfig:
    """Create a :class:`RetrainConfig` from a raw dict (e.g. TOML section).

    Unknown keys are silently ignored.

    Args:
        raw: Dictionary of configuration values.

    Returns:
        A validated :class:`RetrainConfig`.

    Raises:
        ValueError: If *action* is not one of the supported values.
    """
    action = str(raw.get("action", _ACTION_LOG))
    if action not in _VALID_ACTIONS:
        raise ValueError(
            f"Invalid retrain action {action!r}. "
            f"Must be one of {sorted(_VALID_ACTIONS)}."
        )
    return RetrainConfig(
        enabled=bool(raw.get("enabled", True)),
        drift_threshold=float(raw.get("drift_threshold", 0.1)),
        min_feedback_count=int(raw.get("min_feedback_count", 100)),
        max_interval_days=int(raw.get("max_interval_days", 30)),
        action=action,
        script_path=str(raw.get("script_path", "")),
        webhook_url=str(raw.get("webhook_url", "")),
        retrain_log_path=str(
            raw.get("retrain_log_path", DEFAULT_RETRAIN_LOG),
        ),
    )


def _last_retrain_timestamp(log_path: str) -> str | None:
    """Return the timestamp of the most recent successful retrain event."""
    try:
        records = read_records(log_path)
    except FileNotFoundError:
        return None
    for record in reversed(records):
        if record.get("success"):
            return record.get("timestamp")
    return None


def check_retrain_needed(
    config: RetrainConfig,
    *,
    current_psi: float | None = None,
    feedback_count: int = 0,
) -> str | None:
    """Evaluate whether retraining should be triggered.

    Args:
        config: Retrain trigger configuration.
        current_psi: Current PSI value from drift detection, or ``None``
            if drift data is unavailable.
        feedback_count: Number of new feedback records since last retrain.

    Returns:
        A human-readable reason string if retraining is needed, or ``None``.
    """
    if not config.enabled:
        return None

    # Check 1: drift threshold exceeded
    if current_psi is not None and current_psi >= config.drift_threshold:
        return (
            f"Drift detected: PSI={current_psi:.4f} "
            f"(threshold={config.drift_threshold})"
        )

    # Check 2: feedback count threshold
    if feedback_count >= config.min_feedback_count:
        return (
            f"Feedback count reached: {feedback_count} "
            f"(threshold={config.min_feedback_count})"
        )

    # Check 3: time since last retrain
    last_ts = _last_retrain_timestamp(config.retrain_log_path)
    if last_ts is None:
        # No prior retrain — trigger on interval basis
        return (
            f"No previous retrain recorded "
            f"(max interval: {config.max_interval_days} days)"
        )
    try:
        last_dt = datetime.fromisoformat(last_ts)
        now = datetime.now(timezone.utc)
        days_since = (now - last_dt).total_seconds() / 86400
        if days_since >= config.max_interval_days:
            return (
                f"Time since last retrain: {days_since:.1f} days "
                f"(max interval: {config.max_interval_days} days)"
            )
    except (ValueError, TypeError):
        pass

    return None


def _execute_action(
    config: RetrainConfig,
    reason: str,
) -> RetrainEvent:
    """Execute the configured retrain action and record the event."""
    now = datetime.now(timezone.utc).isoformat()

    if config.action == _ACTION_LOG:
        logger.info("Retrain triggered: %s", reason)
        return RetrainEvent(
            timestamp=now, reason=reason, action=_ACTION_LOG, success=True,
        )

    if config.action == _ACTION_SCRIPT:
        if not config.script_path:
            return RetrainEvent(
                timestamp=now, reason=reason, action=_ACTION_SCRIPT,
                success=False, detail="No script_path configured",
            )
        try:
            result = subprocess.run(  # noqa: S603
                [config.script_path],
                capture_output=True,
                text=True,
                timeout=300,
            )
            return RetrainEvent(
                timestamp=now, reason=reason, action=_ACTION_SCRIPT,
                success=result.returncode == 0,
                detail=result.stdout[:1000] if result.returncode == 0
                else result.stderr[:1000],
            )
        except Exception as exc:  # noqa: BLE001
            return RetrainEvent(
                timestamp=now, reason=reason, action=_ACTION_SCRIPT,
                success=False, detail=str(exc)[:1000],
            )

    if config.action == _ACTION_WEBHOOK:
        if not config.webhook_url:
            return RetrainEvent(
                timestamp=now, reason=reason, action=_ACTION_WEBHOOK,
                success=False, detail="No webhook_url configured",
            )
        try:
            payload = json.dumps({
                "event": "retrain_triggered",
                "reason": reason,
                "timestamp": now,
            }).encode("utf-8")
            req = urllib.request.Request(
                config.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                status = resp.status
            return RetrainEvent(
                timestamp=now, reason=reason, action=_ACTION_WEBHOOK,
                success=200 <= status < 300,
                detail=f"HTTP {status}",
            )
        except Exception as exc:  # noqa: BLE001
            return RetrainEvent(
                timestamp=now, reason=reason, action=_ACTION_WEBHOOK,
                success=False, detail=str(exc)[:1000],
            )

    return RetrainEvent(
        timestamp=now, reason=reason, action=config.action,
        success=False, detail=f"Unknown action: {config.action}",
    )


def _event_to_dict(event: RetrainEvent) -> dict[str, Any]:
    return {
        "timestamp": event.timestamp,
        "reason": event.reason,
        "action": event.action,
        "success": event.success,
        "detail": event.detail,
    }


def trigger_retrain(
    config: RetrainConfig,
    *,
    current_psi: float | None = None,
    feedback_count: int = 0,
) -> RetrainEvent | None:
    """Check conditions and trigger retraining if needed.

    This is the main entry point.  It evaluates all trigger conditions,
    executes the configured action if retraining is needed, and records
    the event to the retrain log.

    Args:
        config: Retrain trigger configuration.
        current_psi: Current PSI value from drift detection.
        feedback_count: Number of new feedback records since last retrain.

    Returns:
        A :class:`RetrainEvent` if retraining was triggered, or ``None``.
    """
    reason = check_retrain_needed(
        config, current_psi=current_psi, feedback_count=feedback_count,
    )
    if reason is None:
        return None

    event = _execute_action(config, reason)
    append_record(config.retrain_log_path, _event_to_dict(event))
    logger.info(
        "Retrain event recorded: action=%s success=%s reason=%s",
        event.action, event.success, event.reason,
    )
    return event
