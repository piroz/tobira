"""Tests for tobira.serving.auth — API key authentication."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from tobira.backends.protocol import BackendProtocol, PredictionResult

_has_fastapi = True
try:
    from fastapi.testclient import TestClient
except ImportError:
    _has_fastapi = False

requires_fastapi = pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")

TEST_API_KEY = "test-secret-key-1234567890"


def _make_mock_backend() -> MagicMock:
    backend = MagicMock(spec=BackendProtocol)
    backend.predict.return_value = PredictionResult(
        label="spam", score=0.95, labels={"spam": 0.95, "ham": 0.05}
    )
    return backend


class TestGetApiKey:
    def test_returns_none_when_no_config_no_env(self) -> None:
        from tobira.serving.auth import get_api_key

        with patch.dict(os.environ, {}, clear=True):
            assert get_api_key(None) is None

    def test_returns_config_value(self) -> None:
        from tobira.serving.auth import get_api_key

        with patch.dict(os.environ, {}, clear=True):
            assert get_api_key({"api_key": "my-key"}) == "my-key"

    def test_env_overrides_config(self) -> None:
        from tobira.serving.auth import get_api_key

        with patch.dict(os.environ, {"TOBIRA_API_KEY": "env-key"}):
            assert get_api_key({"api_key": "config-key"}) == "env-key"

    def test_env_alone(self) -> None:
        from tobira.serving.auth import get_api_key

        with patch.dict(os.environ, {"TOBIRA_API_KEY": "env-key"}):
            assert get_api_key(None) == "env-key"

    def test_returns_none_for_empty_config(self) -> None:
        from tobira.serving.auth import get_api_key

        with patch.dict(os.environ, {}, clear=True):
            assert get_api_key({}) is None


@requires_fastapi
class TestAuthEndpoints:
    def test_predict_without_auth_when_disabled(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(_make_mock_backend())
        client = TestClient(app)

        resp = client.post("/v1/predict", json={"text": "hello"})
        assert resp.status_code == 200

    def test_predict_returns_401_without_token(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(
            _make_mock_backend(),
            serving={"api_key": TEST_API_KEY},
        )
        client = TestClient(app)

        resp = client.post("/v1/predict", json={"text": "hello"})
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "Bearer"

    def test_predict_returns_401_with_wrong_token(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(
            _make_mock_backend(),
            serving={"api_key": TEST_API_KEY},
        )
        client = TestClient(app)

        resp = client.post(
            "/v1/predict",
            json={"text": "hello"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 401

    def test_predict_succeeds_with_valid_token(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(
            _make_mock_backend(),
            serving={"api_key": TEST_API_KEY},
        )
        client = TestClient(app)

        resp = client.post(
            "/v1/predict",
            json={"text": "hello"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert resp.status_code == 200
        assert resp.json()["label"] == "spam"

    def test_legacy_predict_requires_auth(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(
            _make_mock_backend(),
            serving={"api_key": TEST_API_KEY},
        )
        client = TestClient(app)

        resp = client.post("/predict", json={"text": "hello"})
        assert resp.status_code == 401

    def test_legacy_predict_succeeds_with_valid_token(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(
            _make_mock_backend(),
            serving={"api_key": TEST_API_KEY},
        )
        client = TestClient(app)

        resp = client.post(
            "/predict",
            json={"text": "hello"},
            headers={"Authorization": f"Bearer {TEST_API_KEY}"},
        )
        assert resp.status_code == 200

    def test_health_does_not_require_auth(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(
            _make_mock_backend(),
            serving={"api_key": TEST_API_KEY},
        )
        client = TestClient(app)

        resp = client.get("/v1/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_ready_does_not_require_auth(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(
            _make_mock_backend(),
            serving={"api_key": TEST_API_KEY},
        )
        client = TestClient(app)

        resp = client.get("/v1/health/ready")
        assert resp.status_code == 200

    def test_health_live_does_not_require_auth(self) -> None:
        from tobira.serving.server import create_app

        app = create_app(
            _make_mock_backend(),
            serving={"api_key": TEST_API_KEY},
        )
        client = TestClient(app)

        resp = client.get("/v1/health/live")
        assert resp.status_code == 200

    def test_env_api_key_is_used(self) -> None:
        from tobira.serving.server import create_app

        with patch.dict(os.environ, {"TOBIRA_API_KEY": TEST_API_KEY}):
            app = create_app(_make_mock_backend())
            client = TestClient(app)

            resp = client.post("/v1/predict", json={"text": "hello"})
            assert resp.status_code == 401

            resp = client.post(
                "/v1/predict",
                json={"text": "hello"},
                headers={"Authorization": f"Bearer {TEST_API_KEY}"},
            )
            assert resp.status_code == 200


class TestDoctorApiKeyCheck:
    def test_check_reports_not_configured(self) -> None:
        from tobira.cli.doctor import _check_api_key

        with patch.dict(os.environ, {}, clear=True):
            ok, msg = _check_api_key({})
            assert not ok
            assert "not configured" in msg

    def test_check_reports_too_short(self) -> None:
        from tobira.cli.doctor import _check_api_key

        with patch.dict(os.environ, {}, clear=True):
            ok, msg = _check_api_key({"serving": {"api_key": "short"}})
            assert not ok
            assert "too short" in msg

    def test_check_passes_with_valid_key(self) -> None:
        from tobira.cli.doctor import _check_api_key

        with patch.dict(os.environ, {}, clear=True):
            ok, msg = _check_api_key(
                {"serving": {"api_key": "a-sufficiently-long-api-key"}}
            )
            assert ok
            assert "configured" in msg

    def test_check_uses_env_variable(self) -> None:
        from tobira.cli.doctor import _check_api_key

        with patch.dict(os.environ, {"TOBIRA_API_KEY": "env-key-long-enough!!"}):
            ok, msg = _check_api_key({})
            assert ok
