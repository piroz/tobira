"""Tests for tobira.monitoring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tobira.backends.protocol import BackendProtocol, PredictionResult
from tobira.monitoring.analyzer import (
    AnalysisReport,
    BackendSuggestion,
    KSResult,
    PSIResult,
    RatePoint,
    ThresholdSuggestion,
    analyze,
    compute_ks,
    compute_psi,
)
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

    def test_lazy_analyze_import(self) -> None:
        import tobira.monitoring

        assert hasattr(tobira.monitoring, "analyze")


# ── analyzer helper ─────────────────────────────────────────────


def _make_records(
    n: int,
    *,
    score_base: float = 0.5,
    label: int | None = None,
    day: str = "2025-01-01",
) -> list[dict[str, object]]:
    """Create N synthetic prediction log records."""
    records: list[dict[str, object]] = []
    for i in range(n):
        r: dict[str, object] = {
            "timestamp": f"{day}T{i % 24:02d}:00:00+00:00",
            "score": round(score_base + (i % 10) * 0.05, 4),
        }
        if label is not None:
            r["label"] = label
        records.append(r)
    return records


# ── analyzer tests ──────────────────────────────────────────────


class TestComputePSI:
    def test_identical_distributions(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        result = compute_psi(scores, scores)
        assert result.psi == pytest.approx(0.0, abs=0.01)
        assert result.level == "ok"

    def test_shifted_distribution(self) -> None:
        reference = [0.1, 0.15, 0.2, 0.25, 0.3] * 20
        current = [0.6, 0.65, 0.7, 0.75, 0.8] * 20
        result = compute_psi(reference, current)
        assert result.psi > 0.0
        assert result.level in ("warning", "alert")

    def test_constant_scores(self) -> None:
        result = compute_psi([0.5] * 10, [0.5] * 10)
        assert result.psi == 0.0
        assert result.level == "ok"

    def test_result_frozen(self) -> None:
        result = PSIResult(psi=0.05, level="ok")
        with pytest.raises(AttributeError):
            result.psi = 0.1  # type: ignore[misc]


class TestComputeKS:
    def test_identical_distributions(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = compute_ks(scores, scores)
        assert result.statistic == pytest.approx(0.0)
        assert not result.significant

    def test_different_distributions(self) -> None:
        reference = [0.1, 0.2, 0.3, 0.4, 0.5] * 20
        current = [0.6, 0.7, 0.8, 0.9, 1.0] * 20
        result = compute_ks(reference, current)
        assert result.statistic > 0.0
        assert result.significant

    def test_result_frozen(self) -> None:
        result = KSResult(statistic=0.1, significant=False)
        with pytest.raises(AttributeError):
            result.statistic = 0.2  # type: ignore[misc]


class TestAnalyze:
    def test_empty_records(self) -> None:
        report = analyze([])
        assert report.total_records == 0
        assert len(report.warnings) > 0

    def test_few_records(self) -> None:
        records = _make_records(10)
        report = analyze(records)
        assert report.total_records == 10
        assert report.psi is None
        assert any("Insufficient" in w for w in report.warnings)

    def test_sufficient_records_for_drift(self) -> None:
        records = _make_records(60)
        report = analyze(records)
        assert report.total_records == 60
        assert report.psi is not None
        assert report.ks is not None

    def test_fp_fn_rates_with_labels(self) -> None:
        records: list[dict[str, object]] = []
        for i in range(20):
            records.append({
                "timestamp": f"2025-01-01T{i % 24:02d}:00:00+00:00",
                "score": 0.8 if i % 2 == 0 else 0.2,
                "label": 1 if i % 2 == 0 else 0,
            })
        report = analyze(records)
        assert len(report.fp_rates) > 0
        assert len(report.fn_rates) > 0

    def test_threshold_suggestion_with_enough_labels(self) -> None:
        records: list[dict[str, object]] = []
        for i in range(40):
            is_spam = i < 20
            records.append({
                "timestamp": f"2025-01-01T{i % 24:02d}:00:00+00:00",
                "score": 0.9 - (i % 5) * 0.05 if is_spam else 0.1 + (i % 5) * 0.05,
                "label": 1 if is_spam else 0,
            })
        report = analyze(records)
        assert report.threshold_suggestion is not None
        assert report.threshold_suggestion.expected_f1 > 0.0

    def test_no_threshold_suggestion_with_few_labels(self) -> None:
        records = _make_records(20, label=1)
        report = analyze(records)
        assert report.threshold_suggestion is None

    def test_backend_suggestion_200_labels(self) -> None:
        records = _make_records(200, label=1)
        report = analyze(records)
        assert report.backend_suggestion is not None
        assert "fastText" in report.backend_suggestion.suggestion

    def test_backend_suggestion_1000_labels(self) -> None:
        records = _make_records(1000, label=1)
        report = analyze(records)
        assert report.backend_suggestion is not None
        assert "BERT" in report.backend_suggestion.suggestion

    def test_no_backend_suggestion_few_labels(self) -> None:
        records = _make_records(50, label=1)
        report = analyze(records)
        assert report.backend_suggestion is None

    def test_period_filter(self) -> None:
        old_records = _make_records(30, day="2020-01-01")
        report = analyze(old_records, period_days=7)
        assert report.total_records == 0

    def test_report_frozen(self) -> None:
        report = AnalysisReport(total_records=0, period_start="", period_end="")
        with pytest.raises(AttributeError):
            report.total_records = 5  # type: ignore[misc]


class TestAnalyzerDataclasses:
    def test_rate_point_frozen(self) -> None:
        pt = RatePoint(period="2025-01-01", rate=0.1, count=10)
        with pytest.raises(AttributeError):
            pt.rate = 0.2  # type: ignore[misc]

    def test_threshold_suggestion_frozen(self) -> None:
        ts = ThresholdSuggestion(current=0.5, suggested=0.6, expected_f1=0.9)
        with pytest.raises(AttributeError):
            ts.current = 0.3  # type: ignore[misc]

    def test_backend_suggestion_frozen(self) -> None:
        bs = BackendSuggestion(suggestion="test", labeled_count=100)
        with pytest.raises(AttributeError):
            bs.labeled_count = 200  # type: ignore[misc]


# ── monitor command tests ───────────────────────────────────────


class TestMonitorCommand:
    def test_run_text_format(self, tmp_path: Path) -> None:
        from tobira.cli import main

        log_file = tmp_path / "test.jsonl"
        records = _make_records(40, label=1)
        with open(log_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        exit_code = main(["monitor", str(log_file)])
        assert exit_code == 0

    def test_run_json_format(self, tmp_path: Path) -> None:
        from tobira.cli import main

        log_file = tmp_path / "test.jsonl"
        records = _make_records(40, label=1)
        with open(log_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        exit_code = main(["monitor", str(log_file), "--format", "json"])
        assert exit_code == 0

    def test_missing_file(self) -> None:
        from tobira.cli import main

        exit_code = main(["monitor", "/nonexistent/file.jsonl"])
        assert exit_code == 1

    def test_period_option(self, tmp_path: Path) -> None:
        from tobira.cli import main

        log_file = tmp_path / "test.jsonl"
        records = _make_records(10)
        with open(log_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        exit_code = main(["monitor", str(log_file), "--period", "30"])
        assert exit_code == 0
