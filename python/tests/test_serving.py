"""Tests for tobira.serving."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tobira.backends.protocol import BackendProtocol, PredictionResult
from tobira.serving.schemas import (
    MAX_TEXT_LENGTH,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

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

    def test_predict_text_too_long(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        client = TestClient(app)

        long_text = "a" * (MAX_TEXT_LENGTH + 1)
        resp = client.post("/predict", json={"text": long_text})
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
        from tobira.config import load_toml

        with pytest.raises(FileNotFoundError, match="config file not found"):
            load_toml("/nonexistent/config.toml")

    def test_load_valid_toml(self, tmp_path: Path) -> None:
        from tobira.config import load_toml

        config_file = tmp_path / "config.toml"  # type: ignore[operator]
        config_file.write_text(
            '[backend]\ntype = "fasttext"\nmodel_path = "/tmp/m.bin"\n'
        )

        result = load_toml(str(config_file))
        assert result["backend"]["type"] == "fasttext"
        assert result["backend"]["model_path"] == "/tmp/m.bin"


class TestCreateApp:
    @requires_fastapi
    def test_returns_fastapi_app(self) -> None:
        from fastapi import FastAPI

        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        assert isinstance(app, FastAPI)

    @requires_fastapi
    def test_openapi_metadata(self) -> None:
        import tobira
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        assert app.title == "tobira"
        assert app.description == "Spam prediction API powered by tobira."
        assert app.version == tobira.__version__

    def test_import_error_without_fastapi(self) -> None:
        from tobira.serving.server import _import_deps

        with patch.dict("sys.modules", {"fastapi": None}):
            with pytest.raises(ImportError, match="serving dependencies"):
                _import_deps()


class TestSchemaValidation:
    def test_predict_request_rejects_missing_text(self) -> None:
        with pytest.raises(Exception):
            PredictRequest()  # type: ignore[call-arg]

    def test_predict_response_rejects_missing_fields(self) -> None:
        with pytest.raises(Exception):
            PredictResponse()  # type: ignore[call-arg]

    def test_health_response_rejects_missing_status(self) -> None:
        with pytest.raises(Exception):
            HealthResponse()  # type: ignore[call-arg]

    def test_predict_request_rejects_too_long_text(self) -> None:
        with pytest.raises(Exception):
            PredictRequest(text="a" * (MAX_TEXT_LENGTH + 1))

    def test_predict_request_accepts_max_length_text(self) -> None:
        req = PredictRequest(text="a" * MAX_TEXT_LENGTH)
        assert len(req.text) == MAX_TEXT_LENGTH


class TestMain:
    @requires_fastapi
    @patch("tobira.serving.server._import_deps")
    @patch("tobira.serving.server.load_toml")
    @patch("tobira.serving.server.create_backend")
    @patch("tobira.serving.server.create_app")
    def test_main_calls_uvicorn(
        self,
        mock_create_app: MagicMock,
        mock_create_backend: MagicMock,
        mock_load_toml: MagicMock,
        mock_import: MagicMock,
    ) -> None:
        from tobira.serving.server import main

        mock_uvicorn = MagicMock()
        mock_import.return_value = (MagicMock(), mock_uvicorn)
        mock_load_toml.return_value = {"backend": {"type": "fasttext"}}
        mock_create_backend.return_value = _make_mock_backend()
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        main("/tmp/config.toml", host="0.0.0.0", port=9000)

        mock_load_toml.assert_called_once_with("/tmp/config.toml")
        mock_uvicorn.run.assert_called_once_with(
            mock_app, host="0.0.0.0", port=9000, access_log=False
        )


class TestServingLazyImport:
    def test_lazy_create_app_import(self) -> None:
        import tobira.serving

        assert hasattr(tobira.serving, "create_app")

    def test_unknown_attr_raises(self) -> None:
        import tobira.serving

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = tobira.serving.nonexistent_attr  # type: ignore[attr-defined]


class TestVersion:
    def test_version_exists(self) -> None:
        import tobira

        assert hasattr(tobira, "__version__")
        assert isinstance(tobira.__version__, str)
