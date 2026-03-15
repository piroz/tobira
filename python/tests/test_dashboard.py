"""Tests for tobira.serving.dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tobira.backends.protocol import BackendProtocol, PredictionResult

_has_fastapi = True
try:
    from fastapi.testclient import TestClient
except ImportError:
    _has_fastapi = False

requires_fastapi = pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")


def _make_mock_backend() -> MagicMock:
    backend = MagicMock(spec=BackendProtocol)
    backend.predict.return_value = PredictionResult(
        label="spam", score=0.95, labels={"spam": 0.95, "ham": 0.05}
    )
    return backend


def _write_log(path: Path, records: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_records(
    n: int, label: str = "spam", score: float = 0.8,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for i in range(n):
        records.append({
            "timestamp": f"2025-06-01T{i % 24:02d}:00:00+00:00",
            "label": label,
            "score": round(score + (i % 5) * 0.04, 4),
            "latency_ms": 10.0 + i * 0.5,
        })
    return records


# ── dashboard helper function tests ──────────────────────────


class TestReadLogRecords:
    def test_reads_valid_records(self, tmp_path: Path) -> None:
        from tobira.serving.dashboard import _read_log_records

        log_file = tmp_path / "log.jsonl"
        _write_log(log_file, [{"score": 0.9}, {"score": 0.1}])

        records = _read_log_records(str(log_file))
        assert len(records) == 2

    def test_returns_empty_for_missing_file(self) -> None:
        from tobira.serving.dashboard import _read_log_records

        records = _read_log_records("/nonexistent/log.jsonl")
        assert records == []

    def test_skips_invalid_json_lines(self, tmp_path: Path) -> None:
        from tobira.serving.dashboard import _read_log_records

        log_file = tmp_path / "log.jsonl"
        log_file.write_text('{"score": 0.9}\ninvalid json\n{"score": 0.1}\n')

        records = _read_log_records(str(log_file))
        assert len(records) == 2


class TestComputeSummary:
    def test_empty_records(self) -> None:
        from tobira.serving.dashboard import _compute_summary

        result = _compute_summary([])
        assert result["total_predictions"] == 0
        assert result["last_24h"]["spam"] == 0
        assert result["avg_latency_ms"] == 0.0

    def test_counts_labels(self) -> None:
        from tobira.serving.dashboard import _compute_summary

        records = _make_records(5, label="spam") + _make_records(3, label="ham")
        result = _compute_summary(records)
        assert result["total_predictions"] == 8

    def test_avg_latency(self) -> None:
        from tobira.serving.dashboard import _compute_summary

        records = [
            {"timestamp": "2025-06-01T00:00:00+00:00", "latency_ms": 10.0},
            {"timestamp": "2025-06-01T01:00:00+00:00", "latency_ms": 20.0},
        ]
        result = _compute_summary(records)
        assert result["avg_latency_ms"] == 15.0


class TestComputeDistribution:
    def test_empty_records(self) -> None:
        from tobira.serving.dashboard import _compute_distribution

        result = _compute_distribution([])
        assert result["total"] == 0
        assert result["bins"] == []

    def test_produces_bins(self) -> None:
        from tobira.serving.dashboard import _compute_distribution

        records = [{"score": 0.1 * i} for i in range(11)]
        result = _compute_distribution(records, bins=10)
        assert len(result["bins"]) == 10
        assert len(result["counts"]) == 10
        assert result["total"] == 11


class TestComputeDrift:
    def test_insufficient_data(self) -> None:
        from tobira.serving.dashboard import _compute_drift

        records = [{"score": 0.5}] * 10
        result = _compute_drift(records)
        assert result["status"] == "insufficient_data"

    def test_sufficient_data(self) -> None:
        from tobira.serving.dashboard import _compute_drift

        records = [{"score": 0.5 + (i % 10) * 0.05} for i in range(80)]
        result = _compute_drift(records)
        assert result["status"] in ("ok", "warning", "alert")
        assert result["psi"] is not None


class TestGetBackendStatus:
    def test_returns_type_and_status(self) -> None:
        from tobira.serving.dashboard import _get_backend_status

        backend = _make_mock_backend()
        result = _get_backend_status(backend)
        assert result["type"] == "MagicMock"
        assert result["status"] == "ok"


class TestGetDashboardHTML:
    def test_returns_html(self) -> None:
        from tobira.serving.dashboard import get_dashboard_html

        html = get_dashboard_html()
        assert "<!DOCTYPE html>" in html
        assert "tobira Dashboard" in html


# ── endpoint tests ───────────────────────────────────────────


@requires_fastapi
class TestDashboardEndpoints:
    def _create_app(
        self, tmp_path: Path, records: list[dict[str, object]] | None = None,
    ) -> TestClient:
        from tobira.serving.server import create_app

        log_file = tmp_path / "predictions.jsonl"
        if records:
            _write_log(log_file, records)

        backend = _make_mock_backend()
        app = create_app(
            backend,
            dashboard={"enabled": True, "log_path": str(log_file)},
        )
        return TestClient(app)

    def test_dashboard_html(self, tmp_path: Path) -> None:
        client = self._create_app(tmp_path)
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "tobira Dashboard" in resp.text

    def test_stats_summary(self, tmp_path: Path) -> None:
        records = _make_records(5)
        client = self._create_app(tmp_path, records)
        resp = client.get("/api/stats/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_predictions"] == 5

    def test_stats_distribution(self, tmp_path: Path) -> None:
        records = _make_records(10)
        client = self._create_app(tmp_path, records)
        resp = client.get("/api/stats/distribution")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 10
        assert len(data["bins"]) == 20

    def test_stats_drift_insufficient(self, tmp_path: Path) -> None:
        records = _make_records(5)
        client = self._create_app(tmp_path, records)
        resp = client.get("/api/stats/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "insufficient_data"

    def test_stats_drift_sufficient(self, tmp_path: Path) -> None:
        records = _make_records(80)
        client = self._create_app(tmp_path, records)
        resp = client.get("/api/stats/drift")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "warning", "alert")

    def test_stats_backends(self, tmp_path: Path) -> None:
        client = self._create_app(tmp_path)
        resp = client.get("/api/stats/backends")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "type" in data

    def test_no_dashboard_when_disabled(self) -> None:
        from tobira.serving.server import create_app

        backend = _make_mock_backend()
        app = create_app(backend)
        client = TestClient(app)
        resp = client.get("/dashboard")
        assert resp.status_code == 404

    def test_dashboard_uses_monitoring_log_path(self, tmp_path: Path) -> None:
        from tobira.serving.server import create_app

        log_file = tmp_path / "custom.jsonl"
        _write_log(log_file, _make_records(3))

        backend = _make_mock_backend()
        app = create_app(
            backend,
            monitoring={"enabled": True, "log_path": str(log_file)},
            dashboard={"enabled": True},
        )
        client = TestClient(app)
        resp = client.get("/api/stats/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_predictions"] == 3


@requires_fastapi
class TestMainWithDashboard:
    def test_dashboard_config_passed_to_create_app(self) -> None:
        from unittest.mock import patch

        from tobira.serving.server import main

        mock_uvicorn = MagicMock()
        mock_import = MagicMock(return_value=(MagicMock(), mock_uvicorn))

        config = {
            "backend": {"type": "fasttext"},
            "dashboard": {"enabled": True},
        }

        with (
            patch("tobira.serving.server._import_deps", mock_import),
            patch("tobira.serving.server.load_toml", return_value=config),
            patch(
                "tobira.serving.server.create_backend",
                return_value=_make_mock_backend(),
            ),
            patch("tobira.serving.server.create_app") as mock_create_app,
        ):
            mock_create_app.return_value = MagicMock()
            main("/tmp/config.toml")

            mock_create_app.assert_called_once()
            call_kwargs = mock_create_app.call_args
            assert call_kwargs[1]["dashboard"] == {"enabled": True}
