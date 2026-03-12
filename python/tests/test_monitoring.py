"""Tests for tobira.monitoring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tobira.backends.protocol import BackendProtocol, PredictionResult
from tobira.monitoring.store import append_record, read_records

_has_fastapi = True
try:
    from fastapi.testclient import TestClient
except ImportError:
    _has_fastapi = False

requires_fastapi = pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")


def _make_mock_backend() -> MagicMock:
    """Create a mock backend that implements BackendProtocol."""
    backend = MagicMock(spec=BackendProtocol)
    backend.predict.return_value = PredictionResult(
        label="spam", score=0.95, labels={"spam": 0.95, "ham": 0.05}
    )
    return backend


# ── store tests ──────────────────────────────────────────────────


class TestAppendRecord:
    def test_creates_file_and_writes_record(self, tmp_path: Path) -> None:
        log_file = tmp_path / "metrics.jsonl"
        append_record(log_file, {"label": "spam", "score": 0.95})

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["label"] == "spam"
        assert data["score"] == pytest.approx(0.95)

    def test_appends_multiple_records(self, tmp_path: Path) -> None:
        log_file = tmp_path / "metrics.jsonl"
        append_record(log_file, {"label": "spam"})
        append_record(log_file, {"label": "ham"})

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["label"] == "spam"
        assert json.loads(lines[1])["label"] == "ham"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        log_file = tmp_path / "sub" / "dir" / "metrics.jsonl"
        append_record(log_file, {"label": "spam"})
        assert log_file.exists()


class TestReadRecords:
    def test_reads_all_records(self, tmp_path: Path) -> None:
        log_file = tmp_path / "metrics.jsonl"
        log_file.write_text(
            '{"label": "spam", "score": 0.95}\n'
            '{"label": "ham", "score": 0.10}\n'
        )

        records = read_records(log_file)
        assert len(records) == 2
        assert records[0]["label"] == "spam"
        assert records[1]["label"] == "ham"

    def test_raises_on_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError, match="log file not found"):
            read_records("/nonexistent/file.jsonl")

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        log_file = tmp_path / "metrics.jsonl"
        log_file.write_text('{"label": "spam"}\n\n{"label": "ham"}\n')

        records = read_records(log_file)
        assert len(records) == 2


# ── collector middleware tests ───────────────────────────────────


@requires_fastapi
class TestPredictionCollector:
    def test_logs_predict_request(self, tmp_path: Path) -> None:
        from tobira.serving.server import create_app

        log_file = tmp_path / "predictions.jsonl"
        backend = _make_mock_backend()
        mon = {"enabled": True, "log_path": str(log_file)}
        app = create_app(backend, monitoring=mon)
        client = TestClient(app)

        resp = client.post("/predict", json={"text": "buy now!!!"})

        assert resp.status_code == 200
        records = read_records(log_file)
        assert len(records) == 1
        record = records[0]
        assert record["label"] == "spam"
        assert record["score"] == pytest.approx(0.95)
        assert "latency_ms" in record
        assert "timestamp" in record

    def test_does_not_log_text(self, tmp_path: Path) -> None:
        from tobira.serving.server import create_app

        log_file = tmp_path / "predictions.jsonl"
        backend = _make_mock_backend()
        mon = {"enabled": True, "log_path": str(log_file)}
        app = create_app(backend, monitoring=mon)
        client = TestClient(app)

        client.post("/predict", json={"text": "secret message"})

        records = read_records(log_file)
        record = records[0]
        assert "text" not in record

    def test_does_not_log_health_requests(self, tmp_path: Path) -> None:
        from tobira.serving.server import create_app

        log_file = tmp_path / "predictions.jsonl"
        backend = _make_mock_backend()
        mon = {"enabled": True, "log_path": str(log_file)}
        app = create_app(backend, monitoring=mon)
        client = TestClient(app)

        client.get("/health")

        assert not log_file.exists()

    def test_does_not_log_failed_requests(self, tmp_path: Path) -> None:
        from tobira.serving.server import create_app

        log_file = tmp_path / "predictions.jsonl"
        backend = _make_mock_backend()
        mon = {"enabled": True, "log_path": str(log_file)}
        app = create_app(backend, monitoring=mon)
        client = TestClient(app)

        client.post("/predict", json={})  # 422 validation error

        assert not log_file.exists()

    def test_no_middleware_when_disabled(self, tmp_path: Path) -> None:
        from tobira.serving.server import create_app

        log_file = tmp_path / "predictions.jsonl"
        backend = _make_mock_backend()
        mon = {"enabled": False, "log_path": str(log_file)}
        app = create_app(backend, monitoring=mon)
        client = TestClient(app)

        client.post("/predict", json={"text": "buy now!!!"})

        assert not log_file.exists()

    def test_no_middleware_without_monitoring_config(self, tmp_path: Path) -> None:
        from tobira.serving.server import create_app

        log_file = tmp_path / "predictions.jsonl"
        backend = _make_mock_backend()
        app = create_app(backend)
        client = TestClient(app)

        client.post("/predict", json={"text": "buy now!!!"})

        assert not log_file.exists()


@requires_fastapi
class TestMainWithMonitoring:
    def test_monitoring_config_passed_to_create_app(self, tmp_path: Path) -> None:
        from unittest.mock import patch

        from tobira.serving.server import main

        mock_uvicorn = MagicMock()
        mock_import = MagicMock(return_value=(MagicMock(), mock_uvicorn))

        config = {
            "backend": {"type": "fasttext"},
            "monitoring": {"enabled": True, "log_path": "/tmp/test.jsonl"},
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
            assert call_kwargs[1]["monitoring"] == {
                "enabled": True,
                "log_path": "/tmp/test.jsonl",
            }


# ── lazy import tests ────────────────────────────────────────────


class TestMonitoringLazyImport:
    def test_lazy_prediction_collector_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "PredictionCollector")

    def test_lazy_store_imports(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "append_record")
        assert hasattr(tobira.monitoring, "read_records")

    def test_unknown_attr_raises(self) -> None:
        import tobira.monitoring

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = tobira.monitoring.nonexistent_attr  # type: ignore[attr-defined]
