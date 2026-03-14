"""Backend benchmark suite for comparing accuracy, latency, and throughput."""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass
from typing import Any

from tobira.backends.protocol import BackendProtocol, PredictionResult
from tobira.evaluation.metrics import MetricsResult, compute_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencyStats:
    """Latency statistics in milliseconds.

    Attributes:
        p50: Median latency.
        p95: 95th percentile latency.
        p99: 99th percentile latency.
        mean: Mean latency.
        min: Minimum latency.
        max: Maximum latency.
    """

    p50: float
    p95: float
    p99: float
    mean: float
    min: float
    max: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Result of benchmarking a single backend.

    Attributes:
        backend_name: Name identifying the backend.
        metrics: Accuracy, precision, recall, F1, and confusion matrix.
        latency: Latency statistics in milliseconds.
        throughput: Predictions per second.
        total_time_ms: Total benchmark time in milliseconds.
        num_samples: Number of samples evaluated.
        errors: Number of prediction errors encountered.
    """

    backend_name: str
    metrics: MetricsResult
    latency: LatencyStats
    throughput: float
    total_time_ms: float
    num_samples: int
    errors: int


@dataclass
class BenchmarkConfig:
    """Configuration for running benchmarks.

    Attributes:
        warmup_runs: Number of warmup predictions (not measured).
        num_runs: Number of measured prediction runs per sample.
        threshold: Classification threshold for binary predictions.
    """

    warmup_runs: int = 1
    num_runs: int = 5
    threshold: float = 0.5


def _compute_latency_stats(latencies_ms: list[float]) -> LatencyStats:
    """Compute latency percentiles from a list of latency values.

    Args:
        latencies_ms: List of latency values in milliseconds.

    Returns:
        A LatencyStats instance.

    Raises:
        ValueError: If latencies list is empty.
    """
    if not latencies_ms:
        raise ValueError("latencies must not be empty")

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    def _percentile(pct: float) -> float:
        idx = (pct / 100.0) * (n - 1)
        lower = int(idx)
        upper = min(lower + 1, n - 1)
        frac = idx - lower
        return sorted_lat[lower] * (1 - frac) + sorted_lat[upper] * frac

    return LatencyStats(
        p50=_percentile(50),
        p95=_percentile(95),
        p99=_percentile(99),
        mean=statistics.mean(sorted_lat),
        min=sorted_lat[0],
        max=sorted_lat[-1],
    )


def run_benchmark(
    backend: BackendProtocol,
    texts: list[str],
    labels: list[int],
    *,
    backend_name: str = "",
    config: BenchmarkConfig | None = None,
) -> BenchmarkResult:
    """Benchmark a single backend on the given dataset.

    Args:
        backend: Backend instance to benchmark.
        texts: List of input texts.
        labels: Ground truth binary labels (0 or 1).
        backend_name: Human-readable name for the backend.
        config: Benchmark configuration. Uses defaults if not provided.

    Returns:
        A BenchmarkResult with accuracy, latency, and throughput data.

    Raises:
        ValueError: If texts and labels have different lengths or are empty.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")
    if not texts:
        raise ValueError("texts must not be empty")

    cfg = config or BenchmarkConfig()

    # Warmup runs (not measured)
    warmup_count = min(cfg.warmup_runs, len(texts))
    for i in range(warmup_count):
        try:
            backend.predict(texts[i])
        except Exception:
            logger.warning("warmup prediction %d failed", i)

    # Measured runs
    all_latencies: list[float] = []
    predictions: list[PredictionResult] = []
    errors = 0

    for run_idx in range(cfg.num_runs):
        for text in texts:
            start = time.perf_counter()
            try:
                result = backend.predict(text)
                elapsed_ms = (time.perf_counter() - start) * 1000
                all_latencies.append(elapsed_ms)
                # Only collect predictions from the first run for metrics
                if run_idx == 0:
                    predictions.append(result)
            except Exception:
                elapsed_ms = (time.perf_counter() - start) * 1000
                all_latencies.append(elapsed_ms)
                errors += 1
                if run_idx == 0:
                    # Use a default ham prediction for failed predictions
                    predictions.append(
                        PredictionResult(
                            label="ham", score=0.0, labels={"spam": 0.0, "ham": 1.0}
                        )
                    )
                logger.warning("prediction failed for text (run %d)", run_idx)

    # Compute metrics from first-run predictions
    scores = [p.labels.get("spam", p.score) for p in predictions]
    y_pred = [1 if s >= cfg.threshold else 0 for s in scores]
    metrics = compute_metrics(labels, y_pred)

    # Compute latency stats
    latency = _compute_latency_stats(all_latencies)

    # Throughput: predictions per second
    total_time_ms = sum(all_latencies)
    total_predictions = len(all_latencies)
    throughput = (total_predictions / total_time_ms * 1000) if total_time_ms > 0 else 0

    return BenchmarkResult(
        backend_name=backend_name,
        metrics=metrics,
        latency=latency,
        throughput=throughput,
        total_time_ms=total_time_ms,
        num_samples=len(texts),
        errors=errors,
    )


def run_comparative_benchmark(
    backends: dict[str, BackendProtocol],
    texts: list[str],
    labels: list[int],
    *,
    config: BenchmarkConfig | None = None,
) -> list[BenchmarkResult]:
    """Benchmark multiple backends on the same dataset.

    Args:
        backends: Mapping of backend name to backend instance.
        texts: List of input texts.
        labels: Ground truth binary labels (0 or 1).
        config: Benchmark configuration. Uses defaults if not provided.

    Returns:
        List of BenchmarkResult, one per backend.

    Raises:
        ValueError: If backends dict is empty, or texts/labels are invalid.
    """
    if not backends:
        raise ValueError("backends must not be empty")

    results: list[BenchmarkResult] = []
    for name, backend in backends.items():
        logger.info("benchmarking backend: %s", name)
        result = run_benchmark(
            backend, texts, labels, backend_name=name, config=config
        )
        results.append(result)
    return results


def benchmark_to_dict(result: BenchmarkResult) -> dict[str, Any]:
    """Convert a BenchmarkResult to a plain dict for serialization.

    Args:
        result: A BenchmarkResult instance.

    Returns:
        A dict suitable for JSON serialization.
    """
    cm = result.metrics.confusion_matrix
    return {
        "backend_name": result.backend_name,
        "num_samples": result.num_samples,
        "errors": result.errors,
        "metrics": {
            "accuracy": result.metrics.accuracy,
            "precision": result.metrics.precision,
            "recall": result.metrics.recall,
            "f1": result.metrics.f1,
            "confusion_matrix": {
                "tp": cm.tp,
                "fp": cm.fp,
                "fn": cm.fn,
                "tn": cm.tn,
            },
        },
        "latency_ms": {
            "p50": result.latency.p50,
            "p95": result.latency.p95,
            "p99": result.latency.p99,
            "mean": result.latency.mean,
            "min": result.latency.min,
            "max": result.latency.max,
        },
        "throughput_per_sec": result.throughput,
        "total_time_ms": result.total_time_ms,
    }


def benchmark_to_json(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a JSON string.

    Args:
        results: List of BenchmarkResult instances.

    Returns:
        A JSON string.
    """
    data = [benchmark_to_dict(r) for r in results]
    return json.dumps(data, indent=2)


def benchmark_to_markdown(results: list[BenchmarkResult]) -> str:
    """Format benchmark results as a Markdown table.

    Args:
        results: List of BenchmarkResult instances.

    Returns:
        A Markdown-formatted string with comparison tables.
    """
    lines: list[str] = []

    # Accuracy table
    lines.append("## Accuracy Comparison")
    lines.append("")
    lines.append(
        "| Backend | Accuracy | Precision | Recall | F1 | Errors |"
    )
    lines.append(
        "|---------|----------|-----------|--------|----|--------|"
    )
    for r in results:
        m = r.metrics
        lines.append(
            f"| {r.backend_name} "
            f"| {m.accuracy:.4f} "
            f"| {m.precision:.4f} "
            f"| {m.recall:.4f} "
            f"| {m.f1:.4f} "
            f"| {r.errors} |"
        )

    lines.append("")

    # Latency table
    lines.append("## Latency Comparison (ms)")
    lines.append("")
    lines.append(
        "| Backend | p50 | p95 | p99 | Mean | Min | Max |"
    )
    lines.append(
        "|---------|-----|-----|-----|------|-----|-----|"
    )
    for r in results:
        lat = r.latency
        lines.append(
            f"| {r.backend_name} "
            f"| {lat.p50:.2f} "
            f"| {lat.p95:.2f} "
            f"| {lat.p99:.2f} "
            f"| {lat.mean:.2f} "
            f"| {lat.min:.2f} "
            f"| {lat.max:.2f} |"
        )

    lines.append("")

    # Throughput table
    lines.append("## Throughput Comparison")
    lines.append("")
    lines.append("| Backend | Throughput (pred/sec) | Total Time (ms) | Samples |")
    lines.append("|---------|---------------------|-----------------|---------|")
    for r in results:
        lines.append(
            f"| {r.backend_name} "
            f"| {r.throughput:.1f} "
            f"| {r.total_time_ms:.1f} "
            f"| {r.num_samples} |"
        )

    return "\n".join(lines)
