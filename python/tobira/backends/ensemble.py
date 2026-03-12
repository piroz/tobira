"""Ensemble backend implementation."""

from __future__ import annotations

import logging
from typing import Literal, Sequence

from tobira.backends.protocol import BackendProtocol, PredictionResult

logger = logging.getLogger(__name__)


class EnsembleBackend:
    """Ensemble backend that aggregates results from multiple backends.

    Runs multiple child backends sequentially and aggregates their predictions
    using either weighted average or majority vote.

    Args:
        backends: Sequence of child backends to aggregate.
        weights: Optional weights for each backend. Must have the same length
            as *backends*. When ``None``, all backends are weighted equally.
        strategy: Aggregation strategy — ``"weighted_average"`` (default) or
            ``"majority_vote"``.
    """

    def __init__(
        self,
        backends: Sequence[BackendProtocol],
        weights: Sequence[float] | None = None,
        strategy: Literal["weighted_average", "majority_vote"] = "weighted_average",
    ) -> None:
        if len(backends) < 2:
            raise ValueError("EnsembleBackend requires at least 2 backends")
        if weights is not None and len(weights) != len(backends):
            raise ValueError(
                f"weights length ({len(weights)}) must match "
                f"backends length ({len(backends)})"
            )
        if strategy not in ("weighted_average", "majority_vote"):
            raise ValueError(
                f"strategy must be 'weighted_average' or 'majority_vote', "
                f"got {strategy!r}"
            )
        self._backends = list(backends)
        self._weights = list(weights) if weights is not None else None
        self._strategy = strategy

    def predict(self, text: str) -> PredictionResult:
        """Run ensemble inference on the given text."""
        results: list[tuple[int, PredictionResult]] = []
        errors: list[Exception] = []

        for i, backend in enumerate(self._backends):
            try:
                results.append((i, backend.predict(text)))
            except Exception as exc:
                logger.warning("Backend %d failed: %s", i, exc)
                errors.append(exc)

        if not results:
            raise RuntimeError(
                f"All {len(self._backends)} backends failed. "
                f"Errors: {errors}"
            )

        if errors:
            logger.info(
                "EnsembleBackend: %d/%d backends succeeded",
                len(results),
                len(self._backends),
            )

        if self._strategy == "majority_vote":
            return self._majority_vote(results)
        return self._weighted_average(results)

    def _weight_for(self, index: int) -> float:
        """Return the weight for the backend at *index*."""
        if self._weights is None:
            return 1.0
        return self._weights[index]

    def _weighted_average(
        self, results: list[tuple[int, PredictionResult]]
    ) -> PredictionResult:
        """Aggregate results using weighted average of label scores."""
        all_labels: set[str] = set()
        for _, r in results:
            all_labels.update(r.labels.keys())

        total_weight = sum(self._weight_for(i) for i, _ in results)

        averaged: dict[str, float] = {}
        for label in all_labels:
            weighted_sum = sum(
                self._weight_for(i) * r.labels.get(label, 0.0) for i, r in results
            )
            averaged[label] = weighted_sum / total_weight

        best_label = max(averaged, key=averaged.__getitem__)
        return PredictionResult(
            label=best_label,
            score=averaged[best_label],
            labels=averaged,
        )

    def _majority_vote(
        self, results: list[tuple[int, PredictionResult]]
    ) -> PredictionResult:
        """Aggregate results using majority vote."""
        vote_counts: dict[str, float] = {}
        for i, r in results:
            vote_counts[r.label] = vote_counts.get(r.label, 0.0) + self._weight_for(i)

        best_label = max(vote_counts, key=vote_counts.__getitem__)

        # Average scores from results that voted for the winner
        winner_scores: dict[str, float] = {}
        winner_weight = 0.0
        for i, r in results:
            if r.label == best_label:
                w = self._weight_for(i)
                winner_weight += w
                for lbl, score in r.labels.items():
                    winner_scores[lbl] = winner_scores.get(lbl, 0.0) + w * score

        labels = {lbl: s / winner_weight for lbl, s in winner_scores.items()}
        return PredictionResult(
            label=best_label,
            score=labels[best_label],
            labels=labels,
        )
