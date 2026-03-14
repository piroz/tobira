"""Tests for tobira.evaluation.benchmark."""

from __future__ import annotations

import json

import pytest

from tobira.backends.protocol import PredictionResult
from tobira.evaluation.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    LatencyStats,
    _compute_latency_stats,
    benchmark_to_dict,
    benchmark_to_json,
    benchmark_to_markdown,
    run_benchmark,
    run_comparative_benchmark,
)


class _MockBackend:
    """A mock backend that returns fixed predictions."""

    def __init__(
        self,
        spam_score: float = 0.9,
        *,
        fail_on: set[int] | None = None,
    ) -> None:
        self.spam_score = spam_score
        self.fail_on = fail_on or set()
        self.call_count = 0

    def predict(self, text: str) -> PredictionResult:
        self.call_count += 1
        if self.call_count in self.fail_on:
            raise RuntimeError("simulated failure")
        return PredictionResult(
            label="spam" if self.spam_score >= 0.5 else "ham",
            score=self.spam_score,
            labels={"spam": self.spam_score, "ham": 1.0 - self.spam_score},
        )


class TestLatencyStats:
    def test_single_value(self) -> None:
        stats = _compute_latency_stats([5.0])
        assert stats.p50 == 5.0
        assert stats.p95 == 5.0
        assert stats.mean == 5.0
        assert stats.min == 5.0
        assert stats.max == 5.0

    def test_multiple_values(self) -> None:
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = _compute_latency_stats(latencies)
        assert stats.p50 == 3.0
        assert stats.mean == pytest.approx(3.0)
        assert stats.min == 1.0
        assert stats.max == 5.0

    def test_percentiles_ordered(self) -> None:
        latencies = list(range(1, 101))
        stats = _compute_latency_stats([float(x) for x in latencies])
        assert stats.p50 <= stats.p95
        assert stats.p95 <= stats.p99

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="not be empty"):
            _compute_latency_stats([])

    def test_frozen(self) -> None:
        stats = LatencyStats(p50=1.0, p95=2.0, p99=3.0, mean=1.5, min=0.5, max=3.5)
        with pytest.raises(AttributeError):
            stats.p50 = 0.0  # type: ignore[misc]


class TestRunBenchmark:
    def test_basic(self) -> None:
        backend = _MockBackend(spam_score=0.9)
        texts = ["spam text 1", "spam text 2", "ham text 1"]
        labels = [1, 1, 0]
        config = BenchmarkConfig(warmup_runs=0, num_runs=1)

        result = run_benchmark(
            backend, texts, labels, backend_name="mock", config=config
        )

        assert result.backend_name == "mock"
        assert result.num_samples == 3
        assert result.errors == 0
        assert result.throughput > 0
        assert result.total_time_ms > 0
        assert result.latency.p50 > 0

    def test_perfect_accuracy(self) -> None:
        backend = _MockBackend(spam_score=0.9)
        texts = ["spam 1", "spam 2"]
        labels = [1, 1]
        config = BenchmarkConfig(warmup_runs=0, num_runs=1)

        result = run_benchmark(backend, texts, labels, config=config)

        assert result.metrics.accuracy == 1.0
        assert result.metrics.precision == 1.0
        assert result.metrics.recall == 1.0
        assert result.metrics.f1 == 1.0

    def test_with_warmup(self) -> None:
        backend = _MockBackend(spam_score=0.9)
        texts = ["text 1", "text 2"]
        labels = [1, 1]
        config = BenchmarkConfig(warmup_runs=2, num_runs=1)

        run_benchmark(backend, texts, labels, config=config)

        # warmup_runs=2, num_runs=1, 2 texts
        # warmup: 2 calls, measured: 2 calls = 4 total
        assert backend.call_count == 4

    def test_multiple_runs(self) -> None:
        backend = _MockBackend(spam_score=0.9)
        texts = ["text 1", "text 2"]
        labels = [1, 1]
        config = BenchmarkConfig(warmup_runs=0, num_runs=3)

        run_benchmark(backend, texts, labels, config=config)

        # 3 runs * 2 texts = 6 calls
        assert backend.call_count == 6

    def test_default_config(self) -> None:
        backend = _MockBackend(spam_score=0.9)
        texts = ["text"]
        labels = [1]

        run_benchmark(backend, texts, labels)

        # default: warmup=1, num_runs=5 → 1 + 5 = 6 calls
        assert backend.call_count == 6

    def test_length_mismatch(self) -> None:
        backend = _MockBackend()
        with pytest.raises(ValueError, match="same length"):
            run_benchmark(backend, ["text"], [1, 0])

    def test_empty_texts(self) -> None:
        backend = _MockBackend()
        with pytest.raises(ValueError, match="not be empty"):
            run_benchmark(backend, [], [])

    def test_prediction_errors_counted(self) -> None:
        # Fail on calls 2 and 3 (within measured runs)
        backend = _MockBackend(spam_score=0.9, fail_on={2, 3})
        texts = ["text 1", "text 2"]
        labels = [1, 0]
        config = BenchmarkConfig(warmup_runs=0, num_runs=1)

        result = run_benchmark(backend, texts, labels, config=config)

        assert result.errors > 0

    def test_frozen(self) -> None:
        result = BenchmarkResult(
            backend_name="test",
            metrics=pytest.importorskip("tobira.evaluation.metrics").compute_metrics(
                [1, 0], [1, 0]
            ),
            latency=LatencyStats(
                p50=1.0, p95=2.0, p99=3.0, mean=1.5, min=0.5, max=3.5
            ),
            throughput=100.0,
            total_time_ms=10.0,
            num_samples=2,
            errors=0,
        )
        with pytest.raises(AttributeError):
            result.backend_name = "other"  # type: ignore[misc]


class TestRunComparativeBenchmark:
    def test_multiple_backends(self) -> None:
        backends = {
            "fast": _MockBackend(spam_score=0.9),
            "slow": _MockBackend(spam_score=0.1),
        }
        texts = ["spam text", "ham text"]
        labels = [1, 0]
        config = BenchmarkConfig(warmup_runs=0, num_runs=1)

        results = run_comparative_benchmark(
            backends, texts, labels, config=config
        )

        assert len(results) == 2
        assert results[0].backend_name == "fast"
        assert results[1].backend_name == "slow"

    def test_empty_backends(self) -> None:
        with pytest.raises(ValueError, match="not be empty"):
            run_comparative_benchmark({}, ["text"], [1])


class TestBenchmarkToDict:
    def test_contains_all_fields(self) -> None:
        backend = _MockBackend(spam_score=0.9)
        config = BenchmarkConfig(warmup_runs=0, num_runs=1)
        result = run_benchmark(
            backend, ["text"], [1], backend_name="test", config=config
        )

        d = benchmark_to_dict(result)

        assert d["backend_name"] == "test"
        assert d["num_samples"] == 1
        assert "metrics" in d
        assert "latency_ms" in d
        assert "throughput_per_sec" in d
        assert d["metrics"]["confusion_matrix"]["tp"] >= 0


class TestBenchmarkToJson:
    def test_valid_json(self) -> None:
        backend = _MockBackend(spam_score=0.9)
        config = BenchmarkConfig(warmup_runs=0, num_runs=1)
        result = run_benchmark(
            backend, ["text"], [1], backend_name="test", config=config
        )

        output = benchmark_to_json([result])
        data = json.loads(output)

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["backend_name"] == "test"


class TestBenchmarkToMarkdown:
    def test_contains_tables(self) -> None:
        backend = _MockBackend(spam_score=0.9)
        config = BenchmarkConfig(warmup_runs=0, num_runs=1)
        result = run_benchmark(
            backend, ["spam"], [1], backend_name="mock-backend", config=config
        )

        md = benchmark_to_markdown([result])

        assert "## Accuracy Comparison" in md
        assert "## Latency Comparison (ms)" in md
        assert "## Throughput Comparison" in md
        assert "mock-backend" in md
        assert "| Backend |" in md

    def test_multiple_backends(self) -> None:
        config = BenchmarkConfig(warmup_runs=0, num_runs=1)
        results = [
            run_benchmark(
                _MockBackend(spam_score=0.9),
                ["text"],
                [1],
                backend_name="a",
                config=config,
            ),
            run_benchmark(
                _MockBackend(spam_score=0.1),
                ["text"],
                [1],
                backend_name="b",
                config=config,
            ),
        ]

        md = benchmark_to_markdown(results)

        assert md.count("| a ") >= 3  # appears in all 3 tables
        assert md.count("| b ") >= 3


class TestBenchmarkPublicAPI:
    def test_imports_from_package(self) -> None:
        from tobira.evaluation import (
            BenchmarkConfig,
            BenchmarkResult,
            LatencyStats,
            benchmark_to_json,
            benchmark_to_markdown,
            run_benchmark,
            run_comparative_benchmark,
        )

        assert BenchmarkConfig is not None
        assert BenchmarkResult is not None
        assert LatencyStats is not None
        assert benchmark_to_json is not None
        assert benchmark_to_markdown is not None
        assert run_benchmark is not None
        assert run_comparative_benchmark is not None
