"""A/B test management for model comparison and canary deployments.

Provides traffic splitting across multiple backend variants and result
collection for comparing accuracy, latency, and score distributions.
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from tobira.backends.protocol import BackendProtocol, PredictionResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ABVariant:
    """A single variant in an A/B test.

    Attributes:
        name: Human-readable variant name (e.g. "control", "candidate").
        backend: The backend instance for this variant.
        weight: Traffic weight (relative, not percentage).
    """

    name: str
    backend: BackendProtocol
    weight: float = 1.0


@dataclass
class VariantResult:
    """Accumulated results for a single variant."""

    predictions: int = 0
    total_latency_ms: float = 0.0
    label_counts: dict[str, int] = field(default_factory=dict)
    score_sum: float = 0.0
    errors: int = 0

    @property
    def avg_latency_ms(self) -> float:
        if self.predictions == 0:
            return 0.0
        return self.total_latency_ms / self.predictions

    @property
    def avg_score(self) -> float:
        if self.predictions == 0:
            return 0.0
        return self.score_sum / self.predictions


class ABTestRouter:
    """Routes prediction requests to variants based on configured weights.

    Supports two routing strategies:
    - ``"random"``: Each request is independently assigned to a variant
      using weighted random selection (default, safest).
    - ``"hash"``: Deterministic assignment based on a hash of the input
      text, ensuring the same input always goes to the same variant.
    """

    def __init__(
        self,
        variants: list[ABVariant],
        strategy: str = "random",
    ) -> None:
        if not variants:
            raise ValueError("at least one variant is required")
        names = [v.name for v in variants]
        if len(names) != len(set(names)):
            raise ValueError("variant names must be unique")
        total_weight = sum(v.weight for v in variants)
        if total_weight <= 0:
            raise ValueError("total variant weight must be positive")

        self._variants = variants
        self._strategy = strategy
        self._cumulative_weights = self._build_cumulative(variants)
        self._results: dict[str, VariantResult] = {
            v.name: VariantResult() for v in variants
        }
        self._lock = threading.Lock()

    @staticmethod
    def _build_cumulative(variants: list[ABVariant]) -> list[float]:
        cumulative: list[float] = []
        total = 0.0
        for v in variants:
            total += v.weight
            cumulative.append(total)
        return cumulative

    def select_variant(self, text: str) -> ABVariant:
        """Select a variant for the given request."""
        if self._strategy == "hash":
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            value = int(digest[:8], 16) / 0xFFFFFFFF
        else:
            value = random.random()  # noqa: S311

        total = self._cumulative_weights[-1]
        target = value * total
        for i, threshold in enumerate(self._cumulative_weights):
            if target <= threshold:
                return self._variants[i]
        return self._variants[-1]

    def predict(self, text: str) -> tuple[PredictionResult, str]:
        """Run prediction through a selected variant.

        Returns:
            Tuple of (prediction result, variant name).
        """
        variant = self.select_variant(text)
        start = time.monotonic()
        try:
            result = variant.backend.predict(text)
        except Exception:
            with self._lock:
                self._results[variant.name].errors += 1
            raise
        elapsed_ms = (time.monotonic() - start) * 1000

        with self._lock:
            vr = self._results[variant.name]
            vr.predictions += 1
            vr.total_latency_ms += elapsed_ms
            vr.score_sum += result.score
            label = result.label
            vr.label_counts[label] = vr.label_counts.get(label, 0) + 1

        return result, variant.name

    def get_results(self) -> dict[str, dict[str, Any]]:
        """Return collected results for all variants."""
        with self._lock:
            results: dict[str, dict[str, Any]] = {}
            for name, vr in self._results.items():
                results[name] = {
                    "predictions": vr.predictions,
                    "avg_latency_ms": round(vr.avg_latency_ms, 2),
                    "avg_score": round(vr.avg_score, 4),
                    "label_counts": dict(vr.label_counts),
                    "errors": vr.errors,
                }
            return results

    def reset_results(self) -> None:
        """Clear all accumulated results."""
        with self._lock:
            for vr in self._results.values():
                vr.predictions = 0
                vr.total_latency_ms = 0.0
                vr.score_sum = 0.0
                vr.label_counts.clear()
                vr.errors = 0

    @property
    def variant_names(self) -> list[str]:
        return [v.name for v in self._variants]


def create_ab_router(
    config: dict[str, Any],
) -> ABTestRouter:
    """Create an ABTestRouter from a configuration dict.

    Expected config format::

        {
            "variants": [
                {"name": "control", "weight": 80, "backend": {...}},
                {"name": "candidate", "weight": 20, "backend": {...}},
            ],
            "strategy": "random"  # optional, default "random"
        }

    Args:
        config: A/B test configuration dict.

    Returns:
        Configured ABTestRouter instance.
    """
    from tobira.backends.factory import create_backend

    variant_configs = config["variants"]
    variants: list[ABVariant] = []
    for vc in variant_configs:
        try:
            backend = create_backend(vc["backend"])
        except Exception as e:
            raise ValueError(
                f"failed to create backend for variant '{vc['name']}': {e}"
            ) from e
        variants.append(
            ABVariant(
                name=vc["name"],
                backend=backend,
                weight=vc.get("weight", 1.0),
            )
        )

    strategy = config.get("strategy", "random")
    return ABTestRouter(variants=variants, strategy=strategy)


def register_ab_test_routes(app: Any, ab_router: ABTestRouter) -> None:
    """Register A/B test result endpoints on a FastAPI app.

    Args:
        app: A FastAPI application instance.
        ab_router: The ABTestRouter managing the experiment.
    """
    from tobira.serving.schemas import ABTestResultsResponse, ABTestVariantResult

    @app.get(
        "/api/ab-test/results",
        response_model=ABTestResultsResponse,
        tags=["ab-test"],
    )
    async def ab_test_results() -> ABTestResultsResponse:
        raw = ab_router.get_results()
        variants_resp = {
            name: ABTestVariantResult(**data) for name, data in raw.items()
        }
        return ABTestResultsResponse(variants=variants_resp)
