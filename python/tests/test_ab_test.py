"""Tests for tobira.serving.ab_test and A/B test integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tobira.backends.protocol import BackendProtocol, PredictionResult
from tobira.serving.ab_test import (
    ABTestRouter,
    ABVariant,
    VariantResult,
    create_ab_router,
)

_has_fastapi = True
try:
    from fastapi.testclient import TestClient
except ImportError:
    _has_fastapi = False

requires_fastapi = pytest.mark.skipif(not _has_fastapi, reason="fastapi not installed")


def _make_mock_backend(
    label: str = "spam", score: float = 0.95,
) -> MagicMock:
    backend = MagicMock(spec=BackendProtocol)
    backend.predict.return_value = PredictionResult(
        label=label, score=score, labels={label: score, "ham": 1.0 - score}
    )
    return backend


# ── VariantResult tests ────────────────────────────────────


class TestVariantResult:
    def test_avg_latency_zero_predictions(self) -> None:
        vr = VariantResult()
        assert vr.avg_latency_ms == 0.0

    def test_avg_score_zero_predictions(self) -> None:
        vr = VariantResult()
        assert vr.avg_score == 0.0

    def test_avg_latency(self) -> None:
        vr = VariantResult(predictions=2, total_latency_ms=20.0)
        assert vr.avg_latency_ms == pytest.approx(10.0)

    def test_avg_score(self) -> None:
        vr = VariantResult(predictions=4, score_sum=3.0)
        assert vr.avg_score == pytest.approx(0.75)


# ── ABTestRouter tests ─────────────────────────────────────


class TestABTestRouter:
    def test_requires_variants(self) -> None:
        with pytest.raises(ValueError, match="at least one variant"):
            ABTestRouter(variants=[])

    def test_requires_positive_weight(self) -> None:
        backend = _make_mock_backend()
        with pytest.raises(ValueError, match="total variant weight"):
            ABTestRouter(variants=[ABVariant(name="a", backend=backend, weight=0.0)])

    def test_single_variant_always_selected(self) -> None:
        backend = _make_mock_backend()
        router = ABTestRouter(variants=[ABVariant(name="only", backend=backend)])
        for _ in range(10):
            v = router.select_variant("test")
            assert v.name == "only"

    def test_predict_returns_result_and_name(self) -> None:
        backend = _make_mock_backend()
        router = ABTestRouter(variants=[ABVariant(name="ctrl", backend=backend)])
        result, name = router.predict("hello")
        assert name == "ctrl"
        assert result.label == "spam"
        assert result.score == pytest.approx(0.95)

    def test_predict_accumulates_results(self) -> None:
        backend = _make_mock_backend()
        router = ABTestRouter(variants=[ABVariant(name="ctrl", backend=backend)])
        for _ in range(5):
            router.predict("hello")

        results = router.get_results()
        assert results["ctrl"]["predictions"] == 5
        assert results["ctrl"]["avg_score"] > 0
        assert results["ctrl"]["errors"] == 0

    def test_predict_tracks_errors(self) -> None:
        backend = _make_mock_backend()
        backend.predict.side_effect = RuntimeError("fail")
        router = ABTestRouter(variants=[ABVariant(name="ctrl", backend=backend)])

        with pytest.raises(RuntimeError):
            router.predict("hello")

        results = router.get_results()
        assert results["ctrl"]["errors"] == 1
        assert results["ctrl"]["predictions"] == 0

    def test_reset_results(self) -> None:
        backend = _make_mock_backend()
        router = ABTestRouter(variants=[ABVariant(name="ctrl", backend=backend)])
        router.predict("hello")
        router.reset_results()

        results = router.get_results()
        assert results["ctrl"]["predictions"] == 0
        assert results["ctrl"]["errors"] == 0

    def test_variant_names(self) -> None:
        b1 = _make_mock_backend()
        b2 = _make_mock_backend()
        router = ABTestRouter(
            variants=[
                ABVariant(name="a", backend=b1),
                ABVariant(name="b", backend=b2),
            ]
        )
        assert router.variant_names == ["a", "b"]

    def test_weighted_distribution(self) -> None:
        b1 = _make_mock_backend()
        b2 = _make_mock_backend()
        router = ABTestRouter(
            variants=[
                ABVariant(name="control", backend=b1, weight=80),
                ABVariant(name="candidate", backend=b2, weight=20),
            ]
        )
        counts: dict[str, int] = {"control": 0, "candidate": 0}
        for i in range(1000):
            _, name = router.predict(f"text-{i}")
            counts[name] += 1

        # With 80/20 split, control should get roughly 800 ± tolerance
        assert counts["control"] > 600
        assert counts["candidate"] > 50

    def test_hash_strategy_deterministic(self) -> None:
        b1 = _make_mock_backend()
        b2 = _make_mock_backend()
        router = ABTestRouter(
            variants=[
                ABVariant(name="a", backend=b1, weight=50),
                ABVariant(name="b", backend=b2, weight=50),
            ],
            strategy="hash",
        )
        # Same text should always go to the same variant
        _, name1 = router.predict("deterministic-text")
        router.reset_results()
        _, name2 = router.predict("deterministic-text")
        assert name1 == name2


    def test_duplicate_variant_names_rejected(self) -> None:
        backend = _make_mock_backend()
        with pytest.raises(ValueError, match="variant names must be unique"):
            ABTestRouter(
                variants=[
                    ABVariant(name="same", backend=backend),
                    ABVariant(name="same", backend=backend),
                ]
            )


# ── create_ab_router factory tests ────────────────────────


class TestCreateABRouter:
    def test_creates_router_from_config(self) -> None:
        from unittest.mock import patch

        mock_backend = _make_mock_backend()
        config = {
            "variants": [
                {"name": "control", "weight": 80, "backend": {"type": "dummy"}},
                {"name": "candidate", "weight": 20, "backend": {"type": "dummy"}},
            ],
            "strategy": "hash",
        }
        with patch(
            "tobira.backends.factory.create_backend", return_value=mock_backend
        ):
            router = create_ab_router(config)

        assert router.variant_names == ["control", "candidate"]
        assert router._strategy == "hash"

    def test_default_strategy_is_random(self) -> None:
        from unittest.mock import patch

        mock_backend = _make_mock_backend()
        config = {
            "variants": [
                {"name": "only", "backend": {"type": "dummy"}},
            ],
        }
        with patch(
            "tobira.backends.factory.create_backend", return_value=mock_backend
        ):
            router = create_ab_router(config)

        assert router._strategy == "random"

    def test_backend_creation_failure_includes_variant_name(self) -> None:
        from unittest.mock import patch

        config = {
            "variants": [
                {"name": "broken", "backend": {"type": "nonexistent"}},
            ],
        }
        with patch(
            "tobira.backends.factory.create_backend",
            side_effect=RuntimeError("no such backend"),
        ):
            with pytest.raises(ValueError, match="variant 'broken'"):
                create_ab_router(config)


# ── server integration tests ───────────────────────────────


@requires_fastapi
class TestABTestServerIntegration:
    def test_predict_includes_model_version(self) -> None:
        from tobira.serving.server import create_app

        backend = _make_mock_backend()
        app = create_app(backend)
        # Inject ab_router after creation to avoid create_backend calls
        app.state.ab_router = ABTestRouter(
            variants=[ABVariant(name="control", backend=backend, weight=100)]
        )
        client = TestClient(app)

        resp = client.post("/predict", json={"text": "buy now!!!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_version"] == "control"

    def test_predict_no_ab_test_no_model_version(self) -> None:
        from tobira.serving.server import create_app

        backend = _make_mock_backend()
        app = create_app(backend)
        client = TestClient(app)

        resp = client.post("/predict", json={"text": "buy now!!!"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_version"] is None

    def test_ab_test_results_endpoint(self) -> None:
        from tobira.serving.ab_test import register_ab_test_routes
        from tobira.serving.server import create_app

        backend = _make_mock_backend()
        control = _make_mock_backend(label="spam", score=0.9)

        ab_router = ABTestRouter(
            variants=[
                ABVariant(name="control", backend=control, weight=100),
            ]
        )

        app = create_app(backend)
        app.state.ab_router = ab_router
        register_ab_test_routes(app, ab_router)

        client = TestClient(app)
        # Make some predictions
        for _ in range(3):
            client.post("/predict", json={"text": "hello"})

        resp = client.get("/api/ab-test/results")
        assert resp.status_code == 200
        data = resp.json()
        assert "variants" in data
        assert "control" in data["variants"]
        assert data["variants"]["control"]["predictions"] == 3

    def test_no_ab_test_no_results_endpoint(self) -> None:
        from tobira.serving.server import create_app

        backend = _make_mock_backend()
        app = create_app(backend)
        client = TestClient(app)

        resp = client.get("/api/ab-test/results")
        assert resp.status_code == 404


# ── schema tests ───────────────────────────────────────────


class TestABTestSchemas:
    def test_predict_response_model_version_defaults_none(self) -> None:
        from tobira.serving.schemas import PredictResponse

        resp = PredictResponse(
            label="spam", score=0.95, labels={"spam": 0.95, "ham": 0.05}
        )
        assert resp.model_version is None

    def test_predict_response_with_model_version(self) -> None:
        from tobira.serving.schemas import PredictResponse

        resp = PredictResponse(
            label="spam",
            score=0.95,
            labels={"spam": 0.95, "ham": 0.05},
            model_version="control",
        )
        assert resp.model_version == "control"

    def test_ab_test_results_response(self) -> None:
        from tobira.serving.schemas import ABTestResultsResponse, ABTestVariantResult

        resp = ABTestResultsResponse(
            variants={
                "control": ABTestVariantResult(
                    predictions=100,
                    avg_latency_ms=12.5,
                    avg_score=0.85,
                    label_counts={"spam": 80, "ham": 20},
                    errors=0,
                ),
            }
        )
        assert resp.variants["control"].predictions == 100


# ── CLI tests ──────────────────────────────────────────────


class TestABTestCLI:
    def test_register_adds_subcommand(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        # ab-test should be a valid subcommand
        args = parser.parse_args(["ab-test", "--url", "http://localhost:9000"])
        assert args.command == "ab-test"
        assert args.url == "http://localhost:9000"

    def test_default_url(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["ab-test"])
        assert args.url == "http://127.0.0.1:8000"

    def test_timeout_argument(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["ab-test", "--timeout", "30"])
        assert args.timeout == 30

    def test_connection_error(self) -> None:
        from unittest.mock import patch

        from tobira.cli import main

        with patch(
            "urllib.request.urlopen",
            side_effect=ConnectionError("refused"),
        ):
            result = main(["ab-test", "--url", "http://localhost:99999"])

        assert result == 1

    def test_empty_results(self, capsys: pytest.CaptureFixture[str]) -> None:
        import json
        from unittest.mock import MagicMock, patch

        from tobira.cli import main

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"variants": {}}).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = main(["ab-test"])

        assert result == 0
        captured = capsys.readouterr()
        assert "No A/B test results available" in captured.out
