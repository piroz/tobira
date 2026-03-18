"""Tests for the feedback loop feature."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tobira.data.feedback_store import (
    FeedbackRecord,
    load_feedback,
    store_feedback,
)
from tobira.serving.schemas import (
    DashboardFeedbackRequest,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackStatsResponse,
)

_has_fastapi = True
try:
    from fastapi.testclient import TestClient
except ImportError:
    _has_fastapi = False

requires_fastapi = pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")


class TestFeedbackSchemas:
    def test_feedback_request_valid(self) -> None:
        req = FeedbackRequest(text="hello", label="spam", source="rspamd")
        assert req.text == "hello"
        assert req.label == "spam"
        assert req.source == "rspamd"

    def test_feedback_request_default_source(self) -> None:
        req = FeedbackRequest(text="hello", label="ham")
        assert req.source == "unknown"

    def test_feedback_request_rejects_invalid_label(self) -> None:
        with pytest.raises(Exception):
            FeedbackRequest(text="hello", label="invalid")

    def test_feedback_request_accepts_spam(self) -> None:
        req = FeedbackRequest(text="hello", label="spam")
        assert req.label == "spam"

    def test_feedback_request_accepts_ham(self) -> None:
        req = FeedbackRequest(text="hello", label="ham")
        assert req.label == "ham"

    def test_feedback_request_rejects_missing_text(self) -> None:
        with pytest.raises(Exception):
            FeedbackRequest(label="spam")  # type: ignore[call-arg]

    def test_feedback_request_rejects_missing_label(self) -> None:
        with pytest.raises(Exception):
            FeedbackRequest(text="hello")  # type: ignore[call-arg]

    def test_feedback_response(self) -> None:
        resp = FeedbackResponse(status="accepted", id="abc-123")
        assert resp.status == "accepted"
        assert resp.id == "abc-123"

    def test_dashboard_feedback_request_valid(self) -> None:
        req = DashboardFeedbackRequest(
            text="hello", correct_label="spam", source="dashboard",
        )
        assert req.text == "hello"
        assert req.correct_label == "spam"
        assert req.source == "dashboard"

    def test_dashboard_feedback_request_default_source(self) -> None:
        req = DashboardFeedbackRequest(text="hello", correct_label="ham")
        assert req.source == "dashboard"

    def test_dashboard_feedback_request_rejects_invalid_label(self) -> None:
        with pytest.raises(Exception):
            DashboardFeedbackRequest(text="hello", correct_label="invalid")

    def test_feedback_stats_response(self) -> None:
        resp = FeedbackStatsResponse(total=10, spam_reports=7, ham_reports=3)
        assert resp.total == 10
        assert resp.spam_reports == 7
        assert resp.ham_reports == 3


class TestFeedbackStoreClass:
    """Tests for the StoreProtocol-backed FeedbackStore class."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        from tobira.data.feedback_store import FeedbackStore
        from tobira.monitoring.store import JsonlStore

        store = JsonlStore(base_dir=str(tmp_path))
        fs = FeedbackStore(store)

        record = fs.save("test spam email", "spam", "rspamd")
        assert record.label == "spam"
        assert record.source == "rspamd"
        assert record.id

        loaded = fs.load_all()
        assert len(loaded) == 1
        assert loaded[0].id == record.id

    def test_save_anonymises_pii(self, tmp_path: Path) -> None:
        from tobira.data.feedback_store import FeedbackStore
        from tobira.monitoring.store import JsonlStore

        store = JsonlStore(base_dir=str(tmp_path))
        fs = FeedbackStore(store)

        record = fs.save("Contact user@example.com", "spam", "test")
        assert "user@example.com" not in record.text
        assert "<EMAIL>" in record.text

    def test_custom_collection(self, tmp_path: Path) -> None:
        from tobira.data.feedback_store import FeedbackStore
        from tobira.monitoring.store import JsonlStore

        store = JsonlStore(base_dir=str(tmp_path))
        fs = FeedbackStore(store, collection="custom_feedback")

        fs.save("text", "spam", "test")
        assert (tmp_path / "custom_feedback.jsonl").exists()


class TestFeedbackStore:
    def test_store_and_load(self, tmp_path: Path) -> None:
        path = tmp_path / "feedback.jsonl"
        record = store_feedback("test spam email", "spam", "rspamd", path=path)

        assert isinstance(record, FeedbackRecord)
        assert record.label == "spam"
        assert record.source == "rspamd"
        assert record.id  # non-empty UUID
        assert record.timestamp  # non-empty timestamp

        loaded = load_feedback(path)
        assert len(loaded) == 1
        assert loaded[0].id == record.id
        assert loaded[0].label == "spam"

    def test_store_multiple(self, tmp_path: Path) -> None:
        path = tmp_path / "feedback.jsonl"
        r1 = store_feedback("spam text", "spam", "rspamd", path=path)
        r2 = store_feedback("ham text", "ham", "haraka", path=path)

        loaded = load_feedback(path)
        assert len(loaded) == 2
        assert loaded[0].id == r1.id
        assert loaded[1].id == r2.id

    def test_store_anonymises_pii(self, tmp_path: Path) -> None:
        path = tmp_path / "feedback.jsonl"
        text_with_pii = "Contact user@example.com for details"
        record = store_feedback(text_with_pii, "spam", "test", path=path)

        assert "user@example.com" not in record.text
        assert "<EMAIL>" in record.text

    def test_store_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "feedback.jsonl"
        store_feedback("text", "spam", "test", path=path)

        assert path.exists()

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_feedback(tmp_path / "missing.jsonl")

    def test_record_jsonl_format(self, tmp_path: Path) -> None:
        path = tmp_path / "feedback.jsonl"
        store_feedback("text", "spam", "test", path=path)

        raw = path.read_text(encoding="utf-8").strip()
        data = json.loads(raw)
        assert "id" in data
        assert "text" in data
        assert "label" in data
        assert "source" in data
        assert "timestamp" in data


def _make_mock_backend():  # type: ignore[no-untyped-def]
    from unittest.mock import MagicMock

    from tobira.backends.protocol import BackendProtocol, PredictionResult

    backend = MagicMock(spec=BackendProtocol)
    backend.predict.return_value = PredictionResult(
        label="spam", score=0.95, labels={"spam": 0.95, "ham": 0.05}
    )
    return backend


@requires_fastapi
class TestFeedbackEndpoint:
    def test_feedback_endpoint_enabled(self, tmp_path: Path) -> None:
        from tobira.serving.server import create_app

        path = tmp_path / "feedback.jsonl"
        app = create_app(
            _make_mock_backend(),
            feedback={"enabled": True, "store_path": str(path)},
        )
        client = TestClient(app)

        resp = client.post(
            "/feedback",
            json={"text": "spam email", "label": "spam", "source": "rspamd"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accepted"
        assert "id" in data

        loaded = load_feedback(path)
        assert len(loaded) == 1
        assert loaded[0].label == "spam"

    def test_feedback_endpoint_disabled(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        client = TestClient(app)

        resp = client.post(
            "/feedback",
            json={"text": "text", "label": "spam"},
        )
        assert resp.status_code == 404 or resp.status_code == 405

    def test_feedback_invalid_label(self, tmp_path: Path) -> None:
        from tobira.serving.server import create_app

        path = tmp_path / "feedback.jsonl"
        app = create_app(
            _make_mock_backend(),
            feedback={"enabled": True, "store_path": str(path)},
        )
        client = TestClient(app)

        resp = client.post(
            "/feedback",
            json={"text": "text", "label": "invalid"},
        )
        assert resp.status_code == 422

    def test_feedback_missing_text(self, tmp_path: Path) -> None:
        from tobira.serving.server import create_app

        path = tmp_path / "feedback.jsonl"
        app = create_app(
            _make_mock_backend(),
            feedback={"enabled": True, "store_path": str(path)},
        )
        client = TestClient(app)

        resp = client.post("/feedback", json={"label": "spam"})
        assert resp.status_code == 422

    def test_feedback_anonymises_pii(self, tmp_path: Path) -> None:
        from tobira.serving.server import create_app

        path = tmp_path / "feedback.jsonl"
        app = create_app(
            _make_mock_backend(),
            feedback={"enabled": True, "store_path": str(path)},
        )
        client = TestClient(app)

        resp = client.post(
            "/feedback",
            json={
                "text": "Contact user@example.com now!",
                "label": "spam",
                "source": "test",
            },
        )
        assert resp.status_code == 200

        loaded = load_feedback(path)
        assert "user@example.com" not in loaded[0].text
        assert "<EMAIL>" in loaded[0].text
