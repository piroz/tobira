"""Tests for tobira.telemetry."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tobira.telemetry import (
    MetricsSummary,
    TelemetryCollector,
    TelemetryConfig,
    format_metrics_summary,
)

_has_fastapi = True
try:
    from fastapi.testclient import TestClient
except ImportError:
    _has_fastapi = False

requires_fastapi = pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")


# ── TelemetryConfig ──────────────────────────────────────────────


class TestTelemetryConfig:
    def test_defaults(self) -> None:
        cfg = TelemetryConfig()
        assert cfg.enabled is False
        assert cfg.storage_dir == "/var/lib/tobira/telemetry"

    def test_from_dict_empty(self) -> None:
        cfg = TelemetryConfig.from_dict({})
        assert cfg.enabled is False

    def test_from_dict_enabled(self) -> None:
        cfg = TelemetryConfig.from_dict(
            {"enabled": True, "storage_dir": "/tmp/tel"}
        )
        assert cfg.enabled is True
        assert cfg.storage_dir == "/tmp/tel"


# ── TelemetryCollector ───────────────────────────────────────────


class TestTelemetryCollector:
    def test_record_heartbeat_disabled(self, tmp_path: Path) -> None:
        cfg = TelemetryConfig(enabled=False, storage_dir=str(tmp_path))
        collector = TelemetryCollector(cfg)
        record = collector.record_heartbeat("FastTextBackend")

        assert record.backend_type == "FastTextBackend"
        assert record.version != ""
        # File should NOT be created when disabled
        assert not collector.heartbeat_path.exists()

    def test_record_heartbeat_enabled(self, tmp_path: Path) -> None:
        cfg = TelemetryConfig(enabled=True, storage_dir=str(tmp_path))
        collector = TelemetryCollector(cfg)
        record = collector.record_heartbeat("OnnxBackend")

        assert record.backend_type == "OnnxBackend"
        assert collector.heartbeat_path.exists()

        lines = collector.heartbeat_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["backend_type"] == "OnnxBackend"
        assert "timestamp" in data
        assert "version" in data
        assert "os_type" in data

    def test_record_doctor_run_disabled(self, tmp_path: Path) -> None:
        cfg = TelemetryConfig(enabled=False, storage_dir=str(tmp_path))
        collector = TelemetryCollector(cfg)
        results = [(True, "ok"), (False, "fail")]
        record = collector.record_doctor_run(results)

        assert record.all_passed is False
        assert record.total_checks == 2
        assert record.passed_checks == 1
        assert not collector.doctor_path.exists()

    def test_record_doctor_run_enabled(self, tmp_path: Path) -> None:
        cfg = TelemetryConfig(enabled=True, storage_dir=str(tmp_path))
        collector = TelemetryCollector(cfg)
        results = [(True, "ok1"), (True, "ok2")]
        record = collector.record_doctor_run(results)

        assert record.all_passed is True
        assert collector.doctor_path.exists()

        data = json.loads(collector.doctor_path.read_text().strip())
        assert data["all_passed"] is True
        assert data["total_checks"] == 2
        assert data["passed_checks"] == 2

    def test_record_multiple_heartbeats(self, tmp_path: Path) -> None:
        cfg = TelemetryConfig(enabled=True, storage_dir=str(tmp_path))
        collector = TelemetryCollector(cfg)
        collector.record_heartbeat("A")
        collector.record_heartbeat("B")

        lines = collector.heartbeat_path.read_text().strip().split("\n")
        assert len(lines) == 2


# ── summarize ────────────────────────────────────────────────────


class TestSummarize:
    def test_empty(self, tmp_path: Path) -> None:
        cfg = TelemetryConfig(enabled=True, storage_dir=str(tmp_path))
        collector = TelemetryCollector(cfg)
        summary = collector.summarize()

        assert summary.active_weeks == 0
        assert summary.active_days_last_30 == 0
        assert summary.retention_30d == pytest.approx(0.0)
        assert summary.doctor_total_runs == 0
        assert summary.doctor_first_pass_rate == pytest.approx(0.0)
        assert summary.total_heartbeats == 0

    def test_with_data(self, tmp_path: Path) -> None:
        cfg = TelemetryConfig(enabled=True, storage_dir=str(tmp_path))
        collector = TelemetryCollector(cfg)

        # Record some heartbeats
        collector.record_heartbeat("FastTextBackend")
        collector.record_heartbeat("FastTextBackend")

        # Record doctor runs
        collector.record_doctor_run([(True, "ok"), (True, "ok")])
        collector.record_doctor_run([(True, "ok"), (False, "fail")])

        summary = collector.summarize()

        assert summary.total_heartbeats == 2
        assert summary.active_weeks >= 1
        assert summary.active_days_last_30 >= 1
        assert summary.retention_30d > 0.0
        assert summary.doctor_total_runs == 2
        assert summary.doctor_first_pass_rate == pytest.approx(0.5)

    def test_doctor_all_pass(self, tmp_path: Path) -> None:
        cfg = TelemetryConfig(enabled=True, storage_dir=str(tmp_path))
        collector = TelemetryCollector(cfg)

        collector.record_doctor_run([(True, "a"), (True, "b")])
        collector.record_doctor_run([(True, "c"), (True, "d")])

        summary = collector.summarize()
        assert summary.doctor_first_pass_rate == pytest.approx(1.0)


# ── format_metrics_summary ───────────────────────────────────────


class TestFormatMetricsSummary:
    def test_output_contains_kpis(self) -> None:
        summary = MetricsSummary(
            active_weeks=4,
            active_days_last_30=20,
            retention_30d=20 / 30.0,
            doctor_total_runs=10,
            doctor_first_pass_rate=0.8,
            total_heartbeats=100,
        )
        text = format_metrics_summary(summary)

        assert "Weekly Active Deployments" in text
        assert "4" in text
        assert "20/30" in text
        assert "66.7%" in text
        assert "80.0%" in text
        assert "100" in text


# ── Health endpoint integration ──────────────────────────────────


@requires_fastapi
class TestHealthTelemetry:
    def test_health_without_telemetry(self) -> None:
        from tobira.backends.protocol import BackendProtocol
        from tobira.serving.server import create_app

        backend = MagicMock(spec=BackendProtocol)
        app = create_app(backend)
        client = TestClient(app)

        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data.get("telemetry_enabled") is None

    def test_health_with_telemetry_disabled(self) -> None:
        from tobira.backends.protocol import BackendProtocol
        from tobira.serving.server import create_app

        backend = MagicMock(spec=BackendProtocol)
        app = create_app(backend, telemetry={"enabled": False})
        client = TestClient(app)

        resp = client.get("/v1/health")
        data = resp.json()
        assert data["telemetry_enabled"] is False

    def test_health_with_telemetry_enabled(self, tmp_path: Path) -> None:
        from tobira.backends.protocol import BackendProtocol
        from tobira.serving.server import create_app

        backend = MagicMock(spec=BackendProtocol)
        app = create_app(
            backend,
            telemetry={"enabled": True, "storage_dir": str(tmp_path)},
        )
        client = TestClient(app)

        resp = client.get("/v1/health")
        data = resp.json()
        assert data["telemetry_enabled"] is True

        # Heartbeat should be recorded
        heartbeat_file = tmp_path / "heartbeats.jsonl"
        assert heartbeat_file.exists()
        lines = heartbeat_file.read_text().strip().split("\n")
        assert len(lines) == 1


# ── Doctor CLI telemetry integration ─────────────────────────────


class TestDoctorTelemetry:
    def test_doctor_records_when_telemetry_enabled(
        self, tmp_path: Path,
    ) -> None:
        """Doctor run results are stored when telemetry is enabled."""
        config_file = tmp_path / "tobira.toml"
        tel_dir = tmp_path / "telemetry"
        config_file.write_text(
            f'[backend]\nbackend = "fasttext"\nmodel_path = "/nonexistent"\n'
            f'\n[telemetry]\nenabled = true\nstorage_dir = "{tel_dir}"\n'
        )

        from tobira.telemetry import TelemetryCollector, TelemetryConfig

        # run_checks itself doesn't call telemetry; the CLI _run does.
        # We test the collector directly.
        results = [(True, "ok"), (False, "backend failed")]
        cfg = TelemetryConfig(enabled=True, storage_dir=str(tel_dir))
        collector = TelemetryCollector(cfg)
        record = collector.record_doctor_run(results)

        assert record.all_passed is False
        assert record.total_checks == 2
        doctor_file = tel_dir / "doctor_runs.jsonl"
        assert doctor_file.exists()


# ── Monitor CLI telemetry integration ────────────────────────────


class TestMonitorTelemetry:
    def test_print_metrics_summary(self, tmp_path: Path) -> None:
        """_print_metrics_summary produces output without errors."""
        from tobira.cli.monitor_cmd import _print_metrics_summary

        # Create some telemetry data
        cfg = TelemetryConfig(enabled=True, storage_dir=str(tmp_path))
        collector = TelemetryCollector(cfg)
        collector.record_heartbeat("test")

        import io
        import sys

        captured = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = captured
            _print_metrics_summary(str(tmp_path))
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        assert "Deployment Metrics" in output
        assert "Weekly Active Deployments" in output
