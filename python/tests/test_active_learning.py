"""Tests for the Active Learning module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tobira.evaluation.active_learning import (
    SamplingStrategy,
    UncertainSample,
    UncertaintySampler,
    compute_entropy,
    compute_least_confidence,
    compute_margin,
)
from tobira.serving.schemas import (
    ActiveLearningLabelRequest,
    ActiveLearningLabelResponse,
    ActiveLearningQueueResponse,
    ActiveLearningSampleResponse,
    ActiveLearningStatsResponse,
)

_has_fastapi = True
try:
    from fastapi.testclient import TestClient
except ImportError:
    _has_fastapi = False

requires_fastapi = pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")


# ---------------------------------------------------------------------------
# Uncertainty computation
# ---------------------------------------------------------------------------


class TestUncertaintyFunctions:
    def test_entropy_uniform_binary(self) -> None:
        result = compute_entropy({"spam": 0.5, "ham": 0.5})
        assert abs(result - 1.0) < 1e-6

    def test_entropy_certain(self) -> None:
        result = compute_entropy({"spam": 1.0, "ham": 0.0})
        assert abs(result) < 1e-6

    def test_entropy_multiclass_uniform(self) -> None:
        labels = {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}
        result = compute_entropy(labels)
        assert abs(result - 2.0) < 1e-6

    def test_margin_equal_probs(self) -> None:
        result = compute_margin({"spam": 0.5, "ham": 0.5})
        assert abs(result - 1.0) < 1e-6

    def test_margin_certain(self) -> None:
        result = compute_margin({"spam": 1.0, "ham": 0.0})
        assert abs(result) < 1e-6

    def test_margin_single_class(self) -> None:
        result = compute_margin({"spam": 0.9})
        assert result == 0.0

    def test_least_confidence_uncertain(self) -> None:
        result = compute_least_confidence({"spam": 0.5, "ham": 0.5})
        assert abs(result - 0.5) < 1e-6

    def test_least_confidence_certain(self) -> None:
        result = compute_least_confidence({"spam": 1.0, "ham": 0.0})
        assert abs(result) < 1e-6

    def test_least_confidence_empty(self) -> None:
        result = compute_least_confidence({})
        assert result == 1.0


# ---------------------------------------------------------------------------
# UncertaintySampler
# ---------------------------------------------------------------------------


class TestUncertaintySampler:
    def test_init_with_string_strategy(self) -> None:
        sampler = UncertaintySampler(strategy="entropy")
        assert sampler.strategy == SamplingStrategy.ENTROPY

    def test_init_with_enum_strategy(self) -> None:
        sampler = UncertaintySampler(strategy=SamplingStrategy.MARGIN)
        assert sampler.strategy == SamplingStrategy.MARGIN

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError):
            UncertaintySampler(strategy="invalid")

    def test_compute_uncertainty(self) -> None:
        sampler = UncertaintySampler(strategy="entropy")
        result = sampler.compute_uncertainty({"spam": 0.5, "ham": 0.5})
        assert abs(result - 1.0) < 1e-6

    def test_select_candidates_basic(self) -> None:
        sampler = UncertaintySampler(
            strategy="entropy",
            uncertainty_threshold=0.0,
        )
        records = [
            {
                "text": "certain spam",
                "score": 0.95,
                "labels": {"spam": 0.95, "ham": 0.05},
            },
            {
                "text": "uncertain",
                "score": 0.55,
                "labels": {"spam": 0.55, "ham": 0.45},
            },
            {
                "text": "very uncertain",
                "score": 0.5,
                "labels": {"spam": 0.5, "ham": 0.5},
            },
        ]
        candidates = sampler.select_candidates(records, n=2)
        assert len(candidates) == 2
        # Most uncertain first
        assert candidates[0].text == "very uncertain"

    def test_select_candidates_threshold_filter(self) -> None:
        sampler = UncertaintySampler(
            strategy="least_confidence",
            uncertainty_threshold=0.4,
        )
        records = [
            {"text": "certain", "score": 0.95, "labels": {"spam": 0.95, "ham": 0.05}},
            {"text": "uncertain", "score": 0.55, "labels": {"spam": 0.55, "ham": 0.45}},
        ]
        candidates = sampler.select_candidates(records, n=10)
        assert len(candidates) == 1
        assert candidates[0].text == "uncertain"

    def test_select_candidates_skips_no_text(self) -> None:
        sampler = UncertaintySampler(strategy="entropy", uncertainty_threshold=0.0)
        records = [
            {"score": 0.5, "labels": {"spam": 0.5, "ham": 0.5}},
            {"text": "", "score": 0.5, "labels": {"spam": 0.5, "ham": 0.5}},
            {"text": "ok", "score": 0.5, "labels": {"spam": 0.5, "ham": 0.5}},
        ]
        candidates = sampler.select_candidates(records, n=10)
        assert len(candidates) == 1

    def test_select_candidates_skips_no_labels(self) -> None:
        sampler = UncertaintySampler(strategy="entropy", uncertainty_threshold=0.0)
        records = [
            {"text": "no labels", "score": 0.5},
            {"text": "empty labels", "score": 0.5, "labels": {}},
        ]
        candidates = sampler.select_candidates(records, n=10)
        assert len(candidates) == 0


class TestQueueManagement:
    def test_add_and_load_queue(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "queue.jsonl"
        sampler = UncertaintySampler(queue_path=queue_path)

        sample = UncertainSample(
            id="test-1",
            text="test email",
            score=0.5,
            labels={"spam": 0.5, "ham": 0.5},
            uncertainty=1.0,
            strategy="entropy",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        added = sampler.add_to_queue([sample])
        assert added == 1

        loaded = sampler.load_queue()
        assert len(loaded) == 1
        assert loaded[0].id == "test-1"
        assert loaded[0].text == "test email"

    def test_queue_respects_max_size(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "queue.jsonl"
        sampler = UncertaintySampler(queue_path=queue_path, max_queue_size=2)

        samples = [
            UncertainSample(
                id=f"test-{i}",
                text=f"email {i}",
                score=0.5,
                labels={"spam": 0.5, "ham": 0.5},
                uncertainty=1.0,
                strategy="entropy",
                timestamp="2025-01-01T00:00:00+00:00",
            )
            for i in range(5)
        ]
        added = sampler.add_to_queue(samples)
        assert added == 2

        loaded = sampler.load_queue()
        assert len(loaded) == 2

    def test_load_empty_queue(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "queue.jsonl"
        sampler = UncertaintySampler(queue_path=queue_path)
        loaded = sampler.load_queue()
        assert loaded == []

    def test_label_sample(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "queue.jsonl"
        sampler = UncertaintySampler(queue_path=queue_path)

        sample = UncertainSample(
            id="test-1",
            text="test email",
            score=0.5,
            labels={"spam": 0.5, "ham": 0.5},
            uncertainty=1.0,
            strategy="entropy",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        sampler.add_to_queue([sample])

        updated = sampler.label_sample("test-1", "spam")
        assert updated is not None
        assert updated.labeled is True
        assert updated.assigned_label == "spam"

        # Verify persisted
        loaded = sampler.load_queue()
        assert loaded[0].labeled is True
        assert loaded[0].assigned_label == "spam"

    def test_label_nonexistent_returns_none(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "queue.jsonl"
        sampler = UncertaintySampler(queue_path=queue_path)
        result = sampler.label_sample("nonexistent", "spam")
        assert result is None

    def test_get_pending(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "queue.jsonl"
        sampler = UncertaintySampler(queue_path=queue_path)

        samples = [
            UncertainSample(
                id=f"test-{i}",
                text=f"email {i}",
                score=0.5,
                labels={"spam": 0.5, "ham": 0.5},
                uncertainty=1.0,
                strategy="entropy",
                timestamp="2025-01-01T00:00:00+00:00",
            )
            for i in range(3)
        ]
        sampler.add_to_queue(samples)
        sampler.label_sample("test-0", "spam")

        pending = sampler.get_pending()
        assert len(pending) == 2
        assert all(not s.labeled for s in pending)

    def test_queue_stats(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "queue.jsonl"
        sampler = UncertaintySampler(queue_path=queue_path)

        samples = [
            UncertainSample(
                id=f"test-{i}",
                text=f"email {i}",
                score=0.5,
                labels={"spam": 0.5, "ham": 0.5},
                uncertainty=1.0,
                strategy="entropy",
                timestamp="2025-01-01T00:00:00+00:00",
            )
            for i in range(3)
        ]
        sampler.add_to_queue(samples)
        sampler.label_sample("test-0", "spam")
        sampler.label_sample("test-1", "ham")

        stats = sampler.queue_stats()
        assert stats["total"] == 3
        assert stats["pending"] == 1
        assert stats["labeled"] == 2
        assert stats["label_counts"] == {"spam": 1, "ham": 1}

    def test_queue_jsonl_format(self, tmp_path: Path) -> None:
        queue_path = tmp_path / "queue.jsonl"
        sampler = UncertaintySampler(queue_path=queue_path)

        sample = UncertainSample(
            id="test-1",
            text="test",
            score=0.5,
            labels={"spam": 0.5, "ham": 0.5},
            uncertainty=1.0,
            strategy="entropy",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        sampler.add_to_queue([sample])

        raw = queue_path.read_text(encoding="utf-8").strip()
        data = json.loads(raw)
        assert "id" in data
        assert "text" in data
        assert "labels" in data
        assert "uncertainty" in data


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class TestActiveLearningSchemas:
    def test_sample_response(self) -> None:
        resp = ActiveLearningSampleResponse(
            id="abc",
            text="test",
            score=0.5,
            labels={"spam": 0.5, "ham": 0.5},
            uncertainty=1.0,
            strategy="entropy",
            timestamp="2025-01-01",
            labeled=False,
        )
        assert resp.id == "abc"
        assert resp.assigned_label is None

    def test_queue_response(self) -> None:
        resp = ActiveLearningQueueResponse(
            samples=[],
            total=0,
            pending=0,
            labeled=0,
        )
        assert resp.total == 0

    def test_label_request_valid(self) -> None:
        req = ActiveLearningLabelRequest(sample_id="abc", label="spam")
        assert req.label == "spam"

    def test_label_request_rejects_invalid(self) -> None:
        with pytest.raises(Exception):
            ActiveLearningLabelRequest(sample_id="abc", label="invalid")

    def test_label_response(self) -> None:
        resp = ActiveLearningLabelResponse(
            status="labeled", sample_id="abc", label="spam",
        )
        assert resp.status == "labeled"

    def test_stats_response(self) -> None:
        resp = ActiveLearningStatsResponse(
            total=10, pending=3, labeled=7, label_counts={"spam": 5, "ham": 2}
        )
        assert resp.total == 10


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


def _make_mock_backend():  # type: ignore[no-untyped-def]
    from unittest.mock import MagicMock

    from tobira.backends.protocol import BackendProtocol, PredictionResult

    backend = MagicMock(spec=BackendProtocol)
    backend.predict.return_value = PredictionResult(
        label="spam", score=0.95, labels={"spam": 0.95, "ham": 0.05}
    )
    return backend


@requires_fastapi
class TestActiveLearningEndpoints:
    def _create_app_with_queue(self, tmp_path: Path):  # type: ignore[no-untyped-def]
        from tobira.serving.server import create_app

        queue_path = tmp_path / "queue.jsonl"
        app = create_app(
            _make_mock_backend(),
            active_learning={
                "enabled": True,
                "queue_path": str(queue_path),
                "strategy": "entropy",
                "uncertainty_threshold": 0.0,
            },
        )
        return app, queue_path

    def test_queue_endpoint_empty(self, tmp_path: Path) -> None:
        app, _ = self._create_app_with_queue(tmp_path)
        client = TestClient(app)

        resp = client.get("/active-learning/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["pending"] == 0
        assert data["samples"] == []

    def test_stats_endpoint(self, tmp_path: Path) -> None:
        app, _ = self._create_app_with_queue(tmp_path)
        client = TestClient(app)

        resp = client.get("/active-learning/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_label_endpoint_not_found(self, tmp_path: Path) -> None:
        app, _ = self._create_app_with_queue(tmp_path)
        client = TestClient(app)

        resp = client.post(
            "/active-learning/label",
            json={"sample_id": "nonexistent", "label": "spam"},
        )
        assert resp.status_code == 404

    def test_label_endpoint_invalid_label(self, tmp_path: Path) -> None:
        app, _ = self._create_app_with_queue(tmp_path)
        client = TestClient(app)

        resp = client.post(
            "/active-learning/label",
            json={"sample_id": "abc", "label": "invalid"},
        )
        assert resp.status_code == 422

    def test_full_workflow(self, tmp_path: Path) -> None:
        """Test: add sample to queue, list it, label it."""
        app, queue_path = self._create_app_with_queue(tmp_path)

        # Manually add a sample to the queue
        sampler = app.state.active_learning_sampler
        sample = UncertainSample(
            id="workflow-1",
            text="test email",
            score=0.5,
            labels={"spam": 0.5, "ham": 0.5},
            uncertainty=1.0,
            strategy="entropy",
            timestamp="2025-01-01T00:00:00+00:00",
        )
        sampler.add_to_queue([sample])

        client = TestClient(app)

        # Check queue
        resp = client.get("/active-learning/queue")
        assert resp.status_code == 200
        data = resp.json()
        assert data["pending"] == 1
        assert data["samples"][0]["id"] == "workflow-1"

        # Label it
        resp = client.post(
            "/active-learning/label",
            json={"sample_id": "workflow-1", "label": "spam"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "labeled"

        # Verify it's no longer pending
        resp = client.get("/active-learning/queue")
        data = resp.json()
        assert data["pending"] == 0
        assert data["samples"] == []

        # Stats should show labeled
        resp = client.get("/active-learning/stats")
        data = resp.json()
        assert data["labeled"] == 1
        assert data["label_counts"]["spam"] == 1

    def test_endpoints_disabled_by_default(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        client = TestClient(app)

        resp = client.get("/active-learning/queue")
        assert resp.status_code == 404 or resp.status_code == 405


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestActiveLearningCLI:
    def test_cli_help(self) -> None:
        from tobira.cli import main

        result = main(["active-learning"])
        assert result == 1  # prints help

    def test_cli_stats_empty(self, tmp_path: Path) -> None:
        from tobira.cli import main

        queue_path = tmp_path / "queue.jsonl"
        result = main(["active-learning", "stats", "--queue-path", str(queue_path)])
        assert result == 0

    def test_cli_queue_empty(self, tmp_path: Path) -> None:
        from tobira.cli import main

        queue_path = tmp_path / "queue.jsonl"
        result = main(["active-learning", "queue", "--queue-path", str(queue_path)])
        assert result == 0

    def test_cli_label_nonexistent(self, tmp_path: Path) -> None:
        from tobira.cli import main

        queue_path = tmp_path / "queue.jsonl"
        result = main([
            "active-learning", "label", "nonexistent", "spam",
            "--queue-path", str(queue_path),
        ])
        assert result == 1

    def test_cli_select_missing_log(self, tmp_path: Path) -> None:
        from tobira.cli import main

        result = main([
            "active-learning", "select",
            "--log-path", str(tmp_path / "missing.jsonl"),
            "--queue-path", str(tmp_path / "queue.jsonl"),
        ])
        assert result == 1

    def test_cli_select_and_queue(self, tmp_path: Path) -> None:
        from tobira.cli import main
        from tobira.monitoring.store import append_record

        log_path = tmp_path / "predictions.jsonl"
        queue_path = tmp_path / "queue.jsonl"

        # Create prediction logs with text field
        for i in range(5):
            append_record(log_path, {
                "text": f"email {i}",
                "score": 0.5,
                "labels": {"spam": 0.5, "ham": 0.5},
                "label": "spam",
                "timestamp": "2025-01-01T00:00:00+00:00",
            })

        result = main([
            "active-learning", "select",
            "--log-path", str(log_path),
            "--queue-path", str(queue_path),
            "--strategy", "entropy",
            "-n", "3",
        ])
        assert result == 0
        assert queue_path.exists()

        # Now check stats
        result = main([
            "active-learning", "stats",
            "--queue-path", str(queue_path),
            "--json",
        ])
        assert result == 0
