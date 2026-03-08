"""Tests for tobira.serving."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tobira.backends.protocol import BackendProtocol, PredictionResult
from tobira.serving.schemas import HealthResponse, PredictRequest, PredictResponse

_has_fastapi = True
try:
    from fastapi.testclient import TestClient
except ImportError:
    _has_fastapi = False

requires_fastapi = pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")


class TestSchemas:
    def test_predict_request(self) -> None:
        req = PredictRequest(text="hello")
        assert req.text == "hello"

    def test_predict_response(self) -> None:
        resp = PredictResponse(
            label="spam", score=0.95, labels={"spam": 0.95, "ham": 0.05}
        )
        assert resp.label == "spam"
        assert resp.score == 0.95
        assert resp.labels == {"spam": 0.95, "ham": 0.05}

    def test_health_response(self) -> None:
        resp = HealthResponse(status="ok")
        assert resp.status == "ok"


def _make_mock_backend() -> MagicMock:
    """Create a mock backend that implements BackendProtocol."""
    backend = MagicMock(spec=BackendProtocol)
    backend.predict.return_value = PredictionResult(
        label="spam", score=0.95, labels={"spam": 0.95, "ham": 0.05}
    )
    return backend


@requires_fastapi
class TestPredict:
    def test_predict(self) -> None:
        from tobira.serving.server import create_app

        backend = _make_mock_backend()
        app = create_app(backend)
        client = TestClient(app)

        resp = client.post("/predict", json={"text": "buy now!!!"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "spam"
        assert data["score"] == pytest.approx(0.95)
        assert data["labels"] == {
            "spam": pytest.approx(0.95),
            "ham": pytest.approx(0.05),
        }
        backend.predict.assert_called_once_with("buy now!!!")

    def test_predict_missing_text(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        client = TestClient(app)

        resp = client.post("/predict", json={})
        assert resp.status_code == 422


@requires_fastapi
class TestHealth:
    def test_health(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        client = TestClient(app)

        resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestLoadConfig:
    def test_missing_file_raises(self) -> None:
        from tobira.serving.server import _load_config

        with pytest.raises(FileNotFoundError, match="config file not found"):
            _load_config("/nonexistent/config.toml")

    def test_load_valid_toml(self, tmp_path: Path) -> None:
        from tobira.serving.server import _load_config

        config_file = tmp_path / "config.toml"  # type: ignore[operator]
        config_file.write_text(
            '[backend]\ntype = "fasttext"\nmodel_path = "/tmp/m.bin"\n'
        )

        result = _load_config(str(config_file))
        assert result["backend"]["type"] == "fasttext"
        assert result["backend"]["model_path"] == "/tmp/m.bin"


class TestCreateApp:
    @requires_fastapi
    def test_returns_fastapi_app(self) -> None:
        from fastapi import FastAPI

        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        assert isinstance(app, FastAPI)

    def test_import_error_without_fastapi(self) -> None:
        from tobira.serving.server import _import_deps

        with patch.dict("sys.modules", {"fastapi": None}):
            with pytest.raises(ImportError, match="serving dependencies"):
                _import_deps()
