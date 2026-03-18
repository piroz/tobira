"""Tests for tobira.serving.ha module."""

from __future__ import annotations

import os
import signal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tobira.backends.protocol import BackendProtocol, PredictionResult
from tobira.serving.ha import (
    GracefulShutdown,
    InstanceInfo,
    InstanceRegistry,
    LeaderElection,
    ReadinessState,
    _pid_exists,
)

_has_fastapi = True
try:
    from fastapi.testclient import TestClient
except ImportError:
    _has_fastapi = False

requires_fastapi = pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")


class TestReadinessState:
    def test_initial_state_not_ready(self) -> None:
        state = ReadinessState()
        assert state.ready is False

    def test_set_ready(self) -> None:
        state = ReadinessState()
        state.set_ready()
        assert state.ready is True

    def test_set_not_ready(self) -> None:
        state = ReadinessState()
        state.set_ready()
        state.set_not_ready()
        assert state.ready is False


class TestInstanceInfo:
    def test_to_dict(self) -> None:
        info = InstanceInfo(
            instance_id="test-1",
            host="127.0.0.1",
            port=8000,
            pid=12345,
            started_at=1000.0,
        )
        d = info.to_dict()
        assert d["instance_id"] == "test-1"
        assert d["host"] == "127.0.0.1"
        assert d["port"] == 8000
        assert d["pid"] == 12345

    def test_from_dict(self) -> None:
        d = {
            "instance_id": "test-1",
            "host": "127.0.0.1",
            "port": 8000,
            "pid": 12345,
            "started_at": 1000.0,
        }
        info = InstanceInfo.from_dict(d)
        assert info.instance_id == "test-1"
        assert info.port == 8000

    def test_roundtrip(self) -> None:
        info = InstanceInfo(
            instance_id="x", host="h", port=9000, pid=1, started_at=0.0
        )
        restored = InstanceInfo.from_dict(info.to_dict())
        assert restored.instance_id == info.instance_id
        assert restored.port == info.port


class TestLeaderElection:
    def test_acquire_and_release(self, tmp_path: Path) -> None:
        le = LeaderElection(state_dir=str(tmp_path))
        assert le.try_acquire() is True
        assert le.is_leader is True
        le.release()
        assert le.is_leader is False

    def test_second_instance_cannot_acquire(self, tmp_path: Path) -> None:
        le1 = LeaderElection(state_dir=str(tmp_path))
        le2 = LeaderElection(state_dir=str(tmp_path))
        assert le1.try_acquire() is True
        assert le2.try_acquire() is False
        le1.release()
        # Now le2 can acquire
        assert le2.try_acquire() is True
        le2.release()

    def test_release_without_acquire(self, tmp_path: Path) -> None:
        le = LeaderElection(state_dir=str(tmp_path))
        le.release()  # Should not raise


class TestInstanceRegistry:
    def test_register_and_list(self, tmp_path: Path) -> None:
        reg = InstanceRegistry(state_dir=str(tmp_path))
        info = InstanceInfo(
            instance_id="i1",
            host="127.0.0.1",
            port=8000,
            pid=os.getpid(),
        )
        reg.register(info)
        instances = reg.list_instances()
        assert len(instances) == 1
        assert instances[0].instance_id == "i1"

    def test_deregister(self, tmp_path: Path) -> None:
        reg = InstanceRegistry(state_dir=str(tmp_path))
        info = InstanceInfo(
            instance_id="i1",
            host="127.0.0.1",
            port=8000,
            pid=os.getpid(),
        )
        reg.register(info)
        reg.deregister("i1")
        instances = reg.list_instances()
        assert len(instances) == 0

    def test_prunes_stale_instances(self, tmp_path: Path) -> None:
        reg = InstanceRegistry(state_dir=str(tmp_path))
        # Register with a PID that doesn't exist
        info = InstanceInfo(
            instance_id="stale",
            host="127.0.0.1",
            port=8000,
            pid=999999999,
        )
        reg.register(info)
        instances = reg.list_instances()
        assert len(instances) == 0

    def test_list_empty_registry(self, tmp_path: Path) -> None:
        reg = InstanceRegistry(state_dir=str(tmp_path))
        assert reg.list_instances() == []

    def test_corrupted_file_returns_empty(self, tmp_path: Path) -> None:
        reg = InstanceRegistry(state_dir=str(tmp_path))
        instances_file = tmp_path / "instances.json"
        instances_file.write_text("not valid json")
        assert reg.list_instances() == []


class TestPidExists:
    def test_current_pid_exists(self) -> None:
        assert _pid_exists(os.getpid()) is True

    def test_nonexistent_pid(self) -> None:
        assert _pid_exists(999999999) is False


class TestGracefulShutdown:
    def test_initial_state(self) -> None:
        readiness = ReadinessState()
        gs = GracefulShutdown(readiness)
        assert gs.shutting_down is False

    def test_install_does_not_raise(self) -> None:
        readiness = ReadinessState()
        gs = GracefulShutdown(readiness)
        gs.install()
        # Clean up — restore default signal handling
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        signal.signal(signal.SIGINT, signal.SIG_DFL)


def _make_mock_backend() -> MagicMock:
    backend = MagicMock(spec=BackendProtocol)
    backend.predict.return_value = PredictionResult(
        label="spam", score=0.95, labels={"spam": 0.95, "ham": 0.05}
    )
    return backend


@requires_fastapi
class TestHealthEndpoints:
    def test_readiness_ready(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is True

    def test_readiness_not_ready(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        # Simulate pre-startup state by setting readiness to False
        app.state.readiness.set_not_ready()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ready"] is False
        assert data["reason"] is not None

    def test_liveness(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["alive"] is True

    def test_predict_returns_503_when_not_ready(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        app.state.readiness.set_not_ready()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/predict", json={"text": "test"})
        assert resp.status_code == 503

    def test_predict_works_when_ready(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/predict", json={"text": "buy now!!!"})
        assert resp.status_code == 200

    def test_health_always_returns_ok(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        app.state.readiness.set_not_ready()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
