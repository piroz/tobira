"""Opt-in anonymous telemetry for deployment metrics.

All data is stored locally by default.  No information is sent to
external servers unless a future ``[telemetry]`` configuration
explicitly enables it.  The collected KPIs help operators understand
adoption, retention, and quality of their tobira deployment.

KPIs tracked:
- **Weekly Active Deployments**: count of unique weeks the health
  endpoint was hit.
- **30-day Retention**: whether the deployment was active in each
  of the last 30 days.
- **``tobira doctor`` first-pass rate**: ratio of doctor runs where
  all checks passed on the first attempt.
- **NPS**: placeholder — requires external survey tooling.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tobira

logger = logging.getLogger(__name__)

_DEFAULT_TELEMETRY_DIR = "/var/lib/tobira/telemetry"


# ── Configuration ─────────────────────────────────────────────────


@dataclass(frozen=True)
class TelemetryConfig:
    """Parsed ``[telemetry]`` configuration.

    Attributes:
        enabled: Master switch (default ``False``).
        storage_dir: Local directory for telemetry JSONL files.
    """

    enabled: bool = False
    storage_dir: str = _DEFAULT_TELEMETRY_DIR

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TelemetryConfig:
        """Create a :class:`TelemetryConfig` from a configuration dict.

        Args:
            data: Configuration dictionary (typically the ``[telemetry]``
                section of ``tobira.toml``).

        Returns:
            A :class:`TelemetryConfig` instance.
        """
        return cls(
            enabled=bool(data.get("enabled", False)),
            storage_dir=str(data.get("storage_dir", _DEFAULT_TELEMETRY_DIR)),
        )


# ── Data structures ───────────────────────────────────────────────


@dataclass(frozen=True)
class HeartbeatRecord:
    """A single heartbeat emitted on health-check.

    Attributes:
        timestamp: ISO-8601 UTC timestamp.
        version: tobira package version.
        backend_type: Name of the active backend.
        os_type: Operating system identifier.
    """

    timestamp: str
    version: str
    backend_type: str
    os_type: str


@dataclass(frozen=True)
class DoctorRunRecord:
    """Result of a single ``tobira doctor`` execution.

    Attributes:
        timestamp: ISO-8601 UTC timestamp.
        all_passed: ``True`` when every check passed.
        total_checks: Total number of checks executed.
        passed_checks: Number of checks that passed.
    """

    timestamp: str
    all_passed: bool
    total_checks: int
    passed_checks: int


@dataclass(frozen=True)
class MetricsSummary:
    """Aggregated KPI summary.

    Attributes:
        active_weeks: Number of distinct ISO weeks with at least one
            heartbeat in the last 52 weeks.
        active_days_last_30: Number of distinct days with heartbeats
            in the last 30 days.
        retention_30d: Ratio of *active_days_last_30* / 30.
        doctor_total_runs: Total number of ``tobira doctor`` runs.
        doctor_first_pass_rate: Ratio of runs where all checks passed.
        total_heartbeats: Total heartbeat count.
    """

    active_weeks: int
    active_days_last_30: int
    retention_30d: float
    doctor_total_runs: int
    doctor_first_pass_rate: float
    total_heartbeats: int


# ── Storage helpers ───────────────────────────────────────────────


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a JSON record to a JSONL file.

    Args:
        path: JSONL file path; parent directories are created as needed.
        record: Dictionary to serialise.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all records from a JSONL file.

    Args:
        path: JSONL file path.

    Returns:
        List of parsed dictionaries, or empty list if missing.
    """
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


# ── Collector ─────────────────────────────────────────────────────


class TelemetryCollector:
    """Local-only telemetry collector.

    Records heartbeats and doctor results to JSONL files under
    *storage_dir*.  No data leaves the local filesystem.

    Args:
        config: Telemetry configuration.
    """

    def __init__(self, config: TelemetryConfig) -> None:
        self._config = config
        self._base = Path(config.storage_dir)

    @property
    def enabled(self) -> bool:
        """Whether telemetry collection is enabled."""
        return self._config.enabled

    @property
    def heartbeat_path(self) -> Path:
        """Path to the heartbeat JSONL file."""
        return self._base / "heartbeats.jsonl"

    @property
    def doctor_path(self) -> Path:
        """Path to the doctor results JSONL file."""
        return self._base / "doctor_runs.jsonl"

    # -- recording --

    def record_heartbeat(self, backend_type: str) -> HeartbeatRecord:
        """Record a health-check heartbeat.

        Args:
            backend_type: Name of the active inference backend.

        Returns:
            The recorded :class:`HeartbeatRecord`.
        """
        record = HeartbeatRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            version=tobira.__version__,
            backend_type=backend_type,
            os_type=platform.system(),
        )
        if self._config.enabled:
            _append_jsonl(self.heartbeat_path, {
                "timestamp": record.timestamp,
                "version": record.version,
                "backend_type": record.backend_type,
                "os_type": record.os_type,
            })
        return record

    def record_doctor_run(
        self, results: list[tuple[bool, str]],
    ) -> DoctorRunRecord:
        """Record a ``tobira doctor`` execution.

        Args:
            results: List of *(passed, message)* tuples returned by
                ``doctor.run_checks``.

        Returns:
            The recorded :class:`DoctorRunRecord`.
        """
        passed = sum(1 for ok, _ in results if ok)
        record = DoctorRunRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            all_passed=passed == len(results),
            total_checks=len(results),
            passed_checks=passed,
        )
        if self._config.enabled:
            _append_jsonl(self.doctor_path, {
                "timestamp": record.timestamp,
                "all_passed": record.all_passed,
                "total_checks": record.total_checks,
                "passed_checks": record.passed_checks,
            })
        return record

    # -- aggregation --

    def summarize(self) -> MetricsSummary:
        """Compute a :class:`MetricsSummary` from stored data.

        Returns:
            Aggregated KPI values.  If telemetry is disabled or no data
            exists, all values are zero.
        """
        heartbeats = _read_jsonl(self.heartbeat_path)
        doctor_runs = _read_jsonl(self.doctor_path)

        now = datetime.now(timezone.utc)

        # Active weeks (last 52 weeks)
        week_set: set[str] = set()
        day_set: set[str] = set()
        for hb in heartbeats:
            try:
                ts = datetime.fromisoformat(hb["timestamp"])
            except (KeyError, ValueError):
                continue
            iso_year, iso_week, _ = ts.isocalendar()
            week_set.add(f"{iso_year}-W{iso_week:02d}")
            days_ago = (now - ts).days
            if 0 <= days_ago < 30:
                day_set.add(ts.strftime("%Y-%m-%d"))

        # Doctor first-pass rate
        total_doctor = len(doctor_runs)
        passed_doctor = sum(
            1 for r in doctor_runs if r.get("all_passed", False)
        )
        first_pass_rate = (
            passed_doctor / total_doctor if total_doctor > 0 else 0.0
        )

        return MetricsSummary(
            active_weeks=len(week_set),
            active_days_last_30=len(day_set),
            retention_30d=len(day_set) / 30.0,
            doctor_total_runs=total_doctor,
            doctor_first_pass_rate=first_pass_rate,
            total_heartbeats=len(heartbeats),
        )


def format_metrics_summary(summary: MetricsSummary) -> str:
    """Format a :class:`MetricsSummary` as human-readable text.

    Args:
        summary: The metrics summary to format.

    Returns:
        Multi-line string suitable for terminal output.
    """
    lines = [
        "=== Deployment Metrics ===",
        f"Weekly Active Deployments (unique weeks): {summary.active_weeks}",
        f"Active days (last 30): {summary.active_days_last_30}/30",
        f"30-day Retention: {summary.retention_30d:.1%}",
        f"Doctor runs: {summary.doctor_total_runs}",
        f"Doctor first-pass rate: {summary.doctor_first_pass_rate:.1%}",
        f"Total heartbeats: {summary.total_heartbeats}",
    ]
    return "\n".join(lines)
