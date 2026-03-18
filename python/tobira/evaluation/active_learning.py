"""Active Learning module for efficient labeling sample selection.

Provides uncertainty-based sampling strategies to identify prediction
samples where the model is least confident, enabling targeted labeling
to maximise model improvement per labeled sample.

The module reads prediction logs from :mod:`tobira.monitoring.store` and
manages a JSONL-based labeling queue.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from tobira.monitoring.store import append_record, read_records

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_PATH = "/var/lib/tobira/active_learning_queue.jsonl"
DEFAULT_MAX_QUEUE_SIZE = 1000


class SamplingStrategy(str, Enum):
    """Uncertainty sampling strategies for Active Learning."""

    ENTROPY = "entropy"
    MARGIN = "margin"
    LEAST_CONFIDENCE = "least_confidence"


@dataclass(frozen=True)
class UncertainSample:
    """A prediction sample identified as uncertain.

    Attributes:
        id: Unique identifier for the sample.
        text: The email text that was classified.
        score: The prediction confidence score.
        labels: Per-class scores from the prediction.
        uncertainty: Computed uncertainty value (higher = more uncertain).
        strategy: The sampling strategy used to compute uncertainty.
        timestamp: ISO-8601 timestamp of when the sample was added to the queue.
        labeled: Whether this sample has been labeled.
        assigned_label: The label assigned by the reviewer, if any.
    """

    id: str
    text: str
    score: float
    labels: dict[str, float]
    uncertainty: float
    strategy: str
    timestamp: str
    labeled: bool = False
    assigned_label: str | None = None


def compute_entropy(labels: dict[str, float]) -> float:
    """Compute Shannon entropy from class probability distribution.

    Args:
        labels: Dictionary mapping class names to probabilities.

    Returns:
        Entropy value. Higher values indicate more uncertainty.
    """
    entropy = 0.0
    for prob in labels.values():
        if prob > 0:
            entropy -= prob * math.log2(prob)
    return entropy


def compute_margin(labels: dict[str, float]) -> float:
    """Compute margin uncertainty (1 - margin between top two classes).

    A smaller margin between the two most probable classes indicates
    higher uncertainty. We return ``1 - margin`` so that higher values
    mean more uncertain.

    Args:
        labels: Dictionary mapping class names to probabilities.

    Returns:
        Uncertainty value in [0, 1]. Higher = more uncertain.
    """
    if len(labels) < 2:
        return 0.0
    sorted_probs = sorted(labels.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1]
    return 1.0 - margin


def compute_least_confidence(labels: dict[str, float]) -> float:
    """Compute least-confidence uncertainty (1 - max probability).

    Args:
        labels: Dictionary mapping class names to probabilities.

    Returns:
        Uncertainty value in [0, 1]. Higher = more uncertain.
    """
    if not labels:
        return 1.0
    return 1.0 - max(labels.values())


_STRATEGY_FUNCTIONS = {
    SamplingStrategy.ENTROPY: compute_entropy,
    SamplingStrategy.MARGIN: compute_margin,
    SamplingStrategy.LEAST_CONFIDENCE: compute_least_confidence,
}


class UncertaintySampler:
    """Select uncertain samples from prediction logs for labeling.

    Args:
        strategy: Uncertainty sampling strategy to use.
        uncertainty_threshold: Minimum uncertainty to consider a sample
            as a candidate. Defaults to 0.3.
        max_queue_size: Maximum number of samples in the labeling queue.
        queue_path: Path to the JSONL labeling queue file.
    """

    def __init__(
        self,
        strategy: str | SamplingStrategy = SamplingStrategy.ENTROPY,
        uncertainty_threshold: float = 0.3,
        max_queue_size: int = DEFAULT_MAX_QUEUE_SIZE,
        queue_path: str | Path = DEFAULT_QUEUE_PATH,
    ) -> None:
        if isinstance(strategy, str):
            strategy = SamplingStrategy(strategy)
        self.strategy = strategy
        self.uncertainty_threshold = uncertainty_threshold
        self.max_queue_size = max_queue_size
        self.queue_path = Path(queue_path)
        self._compute_fn = _STRATEGY_FUNCTIONS[self.strategy]

    def compute_uncertainty(self, labels: dict[str, float]) -> float:
        """Compute uncertainty for a single prediction.

        Args:
            labels: Per-class probability scores.

        Returns:
            Uncertainty value.
        """
        return self._compute_fn(labels)

    def select_candidates(
        self,
        records: list[dict[str, Any]],
        n: int = 10,
    ) -> list[UncertainSample]:
        """Select the most uncertain samples from prediction records.

        Args:
            records: Prediction log records. Each record should have
                ``text``, ``score``, and ``labels`` keys.
            n: Maximum number of candidates to return.

        Returns:
            List of :class:`UncertainSample` sorted by uncertainty
            (most uncertain first).
        """
        candidates: list[tuple[float, dict[str, Any]]] = []
        for record in records:
            labels = record.get("labels")
            if not isinstance(labels, dict) or not labels:
                continue
            text = record.get("text")
            if not text:
                continue
            uncertainty = self._compute_fn(labels)
            if uncertainty >= self.uncertainty_threshold:
                candidates.append((uncertainty, record))

        candidates.sort(key=lambda x: x[0], reverse=True)
        now = datetime.now(timezone.utc).isoformat()

        samples: list[UncertainSample] = []
        for uncertainty, record in candidates[:n]:
            sample = UncertainSample(
                id=str(uuid.uuid4()),
                text=record["text"],
                score=record.get("score", 0.0),
                labels=record["labels"],
                uncertainty=round(uncertainty, 6),
                strategy=self.strategy.value,
                timestamp=now,
            )
            samples.append(sample)
        return samples

    def add_to_queue(self, samples: list[UncertainSample]) -> int:
        """Add uncertain samples to the labeling queue.

        Respects the ``max_queue_size`` limit. Already-queued samples
        beyond the limit are silently dropped.

        Args:
            samples: Samples to add.

        Returns:
            Number of samples actually added.
        """
        current = self.load_queue()
        remaining = self.max_queue_size - len(current)
        if remaining <= 0:
            return 0

        to_add = samples[:remaining]
        for sample in to_add:
            append_record(self.queue_path, _sample_to_dict(sample))
        return len(to_add)

    def load_queue(self) -> list[UncertainSample]:
        """Load all samples from the labeling queue.

        Returns:
            List of :class:`UncertainSample` from the queue file.
            Returns an empty list if the file does not exist.
        """
        if not self.queue_path.exists():
            return []
        try:
            raw = read_records(self.queue_path)
        except FileNotFoundError:
            return []
        return [_dict_to_sample(r) for r in raw]

    def get_pending(self) -> list[UncertainSample]:
        """Return only unlabeled samples from the queue.

        Returns:
            List of unlabeled :class:`UncertainSample`.
        """
        return [s for s in self.load_queue() if not s.labeled]

    def label_sample(self, sample_id: str, label: str) -> UncertainSample | None:
        """Assign a label to a queued sample.

        Rewrites the queue file with the updated sample.

        Args:
            sample_id: The ID of the sample to label.
            label: The label to assign (e.g. ``"spam"`` or ``"ham"``).

        Returns:
            The updated :class:`UncertainSample`, or ``None`` if not found.
        """
        queue = self.load_queue()
        updated: UncertainSample | None = None
        new_queue: list[UncertainSample] = []

        for sample in queue:
            if sample.id == sample_id:
                updated = UncertainSample(
                    id=sample.id,
                    text=sample.text,
                    score=sample.score,
                    labels=sample.labels,
                    uncertainty=sample.uncertainty,
                    strategy=sample.strategy,
                    timestamp=sample.timestamp,
                    labeled=True,
                    assigned_label=label,
                )
                new_queue.append(updated)
            else:
                new_queue.append(sample)

        if updated is not None:
            _rewrite_queue(self.queue_path, new_queue)
        return updated

    def queue_stats(self) -> dict[str, Any]:
        """Return summary statistics about the labeling queue.

        Returns:
            Dictionary with ``total``, ``pending``, ``labeled``, and
            ``label_counts`` keys.
        """
        queue = self.load_queue()
        pending = sum(1 for s in queue if not s.labeled)
        labeled = sum(1 for s in queue if s.labeled)
        label_counts: dict[str, int] = {}
        for s in queue:
            if s.assigned_label is not None:
                label_counts[s.assigned_label] = (
                    label_counts.get(s.assigned_label, 0) + 1
                )
        return {
            "total": len(queue),
            "pending": pending,
            "labeled": labeled,
            "label_counts": label_counts,
        }


def _sample_to_dict(sample: UncertainSample) -> dict[str, Any]:
    return {
        "id": sample.id,
        "text": sample.text,
        "score": sample.score,
        "labels": sample.labels,
        "uncertainty": sample.uncertainty,
        "strategy": sample.strategy,
        "timestamp": sample.timestamp,
        "labeled": sample.labeled,
        "assigned_label": sample.assigned_label,
    }


def _dict_to_sample(d: dict[str, Any]) -> UncertainSample:
    return UncertainSample(
        id=d["id"],
        text=d["text"],
        score=d.get("score", 0.0),
        labels=d.get("labels", {}),
        uncertainty=d.get("uncertainty", 0.0),
        strategy=d.get("strategy", "entropy"),
        timestamp=d.get("timestamp", ""),
        labeled=d.get("labeled", False),
        assigned_label=d.get("assigned_label"),
    )


def _rewrite_queue(path: Path, samples: list[UncertainSample]) -> None:
    """Rewrite the queue file with the given samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(_sample_to_dict(sample), ensure_ascii=False) + "\n")
