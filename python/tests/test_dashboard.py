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


def _make_multi_day_records() -> list[dict[str, object]]:
    """Create records spanning multiple days for trend testing."""
    records: list[dict[str, object]] = []
    for day in range(1, 8):
        for i in range(5):
            records.append({
                "timestamp": f"2025-06-{day:02d}T{i + 10:02d}:00:00+00:00",
                "label": "spam" if i < 3 else "ham",
                "score": 0.9 if i < 3 else 0.2,
                "latency_ms": 12.0,
            })
    return records


# -- dashboard helper function tests ------------------------------------------


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

    def test_contains_chart_js_reference(self) -> None:
        from tobira.serving.dashboard import get_dashboard_html

        html = get_dashboard_html()
        assert "chart.js" in html.lower() or "Chart" in html

    def test_contains_traffic_light(self) -> None:
        from tobira.serving.dashboard import get_dashboard_html

        html = get_dashboard_html()
        assert "traffic-light" in html
        assert "traffic-dot" in html

    def test_contains_trend_chart(self) -> None:
        from tobira.serving.dashboard import get_dashboard_html

        html = get_dashboard_html()
        assert "trend-chart" in html

    def test_contains_recent_table(self) -> None:
        from tobira.serving.dashboard import get_dashboard_html

        html = get_dashboard_html()
        assert "recent-tbody" in html
        assert "pagination" in html


# -- new helper function tests -------------------------------------------------


class TestComputeTrend:
    def test_empty_records(self) -> None:
        from tobira.serving.dashboard import _compute_trend

        result = _compute_trend([])
        assert result["dates"] == []
        assert result["spam"] == []
        assert result["ham"] == []

    def test_aggregates_by_day(self) -> None:
        from tobira.serving.dashboard import _compute_trend

        records = _make_multi_day_records()
        result = _compute_trend(records)
        assert len(result["dates"]) == 7
        assert result["dates"][0] == "2025-06-01"
        assert result["dates"][-1] == "2025-06-07"
        # Each day has 3 spam and 2 ham
        assert result["spam"][0] == 3
        assert result["ham"][0] == 2

    def test_skips_records_without_timestamp(self) -> None:
        from tobira.serving.dashboard import _compute_trend

        records = [
            {"label": "spam"},
            {"timestamp": "2025-06-01T10:00:00+00:00", "label": "spam"},
        ]
        result = _compute_trend(records)
        assert len(result["dates"]) == 1


class TestGetRecentPredictions:
    def test_empty_records(self) -> None:
        from tobira.serving.dashboard import _get_recent_predictions

        result = _get_recent_predictions([])
        assert result["items"] == []
        assert result["total"] == 0
        assert result["page"] == 1
        assert result["total_pages"] == 1

    def test_returns_most_recent_first(self) -> None:
        from tobira.serving.dashboard import _get_recent_predictions

        records = _make_records(5)
        result = _get_recent_predictions(records, page=1, per_page=3)
        assert len(result["items"]) == 3
        assert result["total"] == 5
        assert result["total_pages"] == 2
        # Most recent first (sorted by timestamp descending)
        assert result["items"][0]["timestamp"] >= result["items"][1]["timestamp"]

    def test_pagination(self) -> None:
        from tobira.serving.dashboard import _get_recent_predictions

        records = _make_records(10)
        p1 = _get_recent_predictions(records, page=1, per_page=3)
        p2 = _get_recent_predictions(records, page=2, per_page=3)
        assert p1["page"] == 1
        assert p2["page"] == 2
        assert p1["items"] != p2["items"]
        assert p1["total_pages"] == 4  # ceil(10/3)

    def test_page_clamp(self) -> None:
        from tobira.serving.dashboard import _get_recent_predictions

        records = _make_records(5)
        result = _get_recent_predictions(records, page=999, per_page=3)
        assert result["page"] == 2  # clamped to last page

    def test_items_contain_expected_fields(self) -> None:
        from tobira.serving.dashboard import _get_recent_predictions

        records = _make_records(1)
        result = _get_recent_predictions(records, page=1, per_page=10)
        item = result["items"][0]
        assert "timestamp" in item
        assert "label" in item
        assert "score" in item
        assert "latency_ms" in item


class TestComputeHealthStatus:
    def test_healthy_system(self) -> None:
        from tobira.serving.dashboard import _compute_health_status

        backend = _make_mock_backend()
        records = _make_records(80)
        result = _compute_health_status(records, backend)
        assert result["overall"] in ("green", "yellow", "red")
        assert len(result["components"]) == 3
        # Backend should be green
        assert result["components"][0]["status"] == "green"

    def test_no_records(self) -> None:
        from tobira.serving.dashboard import _compute_health_status

        backend = _make_mock_backend()
        result = _compute_health_status([], backend)
        assert result["components"][1]["status"] == "yellow"  # model activity
        assert result["components"][1]["detail"] == "No prediction data yet"

    def test_overall_worst_status(self) -> None:
        from tobira.serving.dashboard import _compute_health_status

        backend = _make_mock_backend()
        # With only a few records, model activity will be yellow (old timestamps)
        # and drift will be insufficient_data (yellow)
        records = _make_records(5)
        result = _compute_health_status(records, backend)
        # Overall should reflect the worst component
        statuses = [c["status"] for c in result["components"]]
        priority = {"red": 0, "yellow": 1, "green": 2}
        expected_worst = min(statuses, key=lambda s: priority.get(s, 2))
        assert result["overall"] == expected_worst


# -- endpoint tests ------------------------------------------------------------


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

    def test_stats_trend(self, tmp_path: Path) -> None:
        records = _make_multi_day_records()
        client = self._create_app(tmp_path, records)
        resp = client.get("/api/stats/trend")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["dates"]) == 7
        assert len(data["spam"]) == 7
        assert len(data["ham"]) == 7

    def test_stats_trend_empty(self, tmp_path: Path) -> None:
        client = self._create_app(tmp_path)
        resp = client.get("/api/stats/trend")
        assert resp.status_code == 200
        data = resp.json()
        assert data["dates"] == []

    def test_stats_recent(self, tmp_path: Path) -> None:
        records = _make_records(25)
        client = self._create_app(tmp_path, records)
        resp = client.get("/api/stats/recent?page=1&per_page=10")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 10
        assert data["total"] == 25
        assert data["page"] == 1
        assert data["total_pages"] == 3

    def test_stats_recent_default_params(self, tmp_path: Path) -> None:
        records = _make_records(5)
        client = self._create_app(tmp_path, records)
        resp = client.get("/api/stats/recent")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 5
        assert data["per_page"] == 20

    def test_stats_health(self, tmp_path: Path) -> None:
        records = _make_records(10)
        client = self._create_app(tmp_path, records)
        resp = client.get("/api/stats/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall"] in ("green", "yellow", "red")
        assert len(data["components"]) == 3

    def test_stats_health_empty(self, tmp_path: Path) -> None:
        client = self._create_app(tmp_path)
        resp = client.get("/api/stats/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["overall"] in ("green", "yellow", "red")

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


# ── feedback UI tests ────────────────────────────────────────


class TestComputeFeedbackStats:
    def test_empty_file(self, tmp_path: Path) -> None:
        from tobira.serving.dashboard import _compute_feedback_stats

        result = _compute_feedback_stats(str(tmp_path / "missing.jsonl"))
        assert result == {"total": 0, "spam_reports": 0, "ham_reports": 0}

    def test_counts_labels(self, tmp_path: Path) -> None:
        from tobira.serving.dashboard import _compute_feedback_stats

        fb_file = tmp_path / "feedback.jsonl"
        lines = [
            json.dumps({"label": "spam", "text": "a", "id": "1"}),
            json.dumps({"label": "ham", "text": "b", "id": "2"}),
            json.dumps({"label": "spam", "text": "c", "id": "3"}),
        ]
        fb_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = _compute_feedback_stats(str(fb_file))
        assert result["total"] == 3
        assert result["spam_reports"] == 2
        assert result["ham_reports"] == 1


@requires_fastapi
class TestDashboardFeedbackEndpoints:
    def _create_app(
        self,
        tmp_path: Path,
        records: list[dict[str, object]] | None = None,
        feedback_enabled: bool = True,
    ) -> TestClient:
        from tobira.serving.server import create_app

        log_file = tmp_path / "predictions.jsonl"
        if records:
            _write_log(log_file, records)

        feedback_path = str(tmp_path / "feedback.jsonl")
        backend = _make_mock_backend()
        feedback_cfg = (
            {"enabled": True, "store_path": feedback_path}
            if feedback_enabled
            else None
        )
        app = create_app(
            backend,
            dashboard={"enabled": True, "log_path": str(log_file)},
            feedback=feedback_cfg,
        )
        return TestClient(app)

    def test_predictions_recent(self, tmp_path: Path) -> None:
        records = _make_records(5)
        client = self._create_app(tmp_path, records)
        resp = client.get("/api/stats/recent?page=1&per_page=20")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["items"]) == 5
        assert data["total"] == 5

    def test_feedback_stats_empty(self, tmp_path: Path) -> None:
        client = self._create_app(tmp_path)
        resp = client.get("/api/feedback/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_submit_feedback(self, tmp_path: Path) -> None:
        client = self._create_app(tmp_path)
        resp = client.post(
            "/api/feedback",
            json={"text": "spam text", "correct_label": "ham", "source": "dashboard"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert "id" in data

        # Stats should reflect the feedback
        resp = client.get("/api/feedback/stats")
        assert resp.json()["total"] == 1
        assert resp.json()["ham_reports"] == 1

    def test_submit_feedback_invalid_label(self, tmp_path: Path) -> None:
        client = self._create_app(tmp_path)
        resp = client.post(
            "/api/feedback",
            json={"text": "text", "correct_label": "invalid"},
        )
        assert resp.status_code == 422

    def test_feedback_not_available_when_disabled(self, tmp_path: Path) -> None:
        client = self._create_app(tmp_path, feedback_enabled=False)
        resp = client.post(
            "/api/feedback",
            json={"text": "text", "correct_label": "spam"},
        )
        assert resp.status_code in (404, 405)

    def test_feedback_stats_not_available_when_disabled(
        self, tmp_path: Path,
    ) -> None:
        client = self._create_app(tmp_path, feedback_enabled=False)
        resp = client.get("/api/feedback/stats")
        assert resp.status_code == 404

    def test_dashboard_html_includes_feedback_ui(self, tmp_path: Path) -> None:
        client = self._create_app(tmp_path)
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "Feedback Stats" in resp.text
        assert "Recent Predictions" in resp.text

    def test_dashboard_html_excludes_feedback_when_disabled(
        self, tmp_path: Path,
    ) -> None:
        client = self._create_app(tmp_path, feedback_enabled=False)
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "Feedback Stats" not in resp.text
