"""Drift detection, FP/FN analysis, and threshold suggestion engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from tobira.monitoring.drift import DriftResult, _classify_psi

_MIN_SAMPLES = 30
_NUM_BINS = 10


@dataclass(frozen=True)
class PSIResult:
    """Population Stability Index result.

    Attributes:
        psi: The PSI value.
        level: Severity level (``ok``, ``warning``, or ``alert``).
    """

    psi: float
    level: str


@dataclass(frozen=True)
class KSResult:
    """Kolmogorov-Smirnov test result.

    Attributes:
        statistic: The KS statistic (max absolute difference between CDFs).
        significant: Whether the difference is significant at alpha=0.05.
    """

    statistic: float
    significant: bool


@dataclass(frozen=True)
class RatePoint:
    """A single data point in a rate time-series.

    Attributes:
        period: Period label (e.g. ``"2024-01-01"``).
        rate: The rate value.
        count: Number of samples in this period.
    """

    period: str
    rate: float
    count: int


@dataclass(frozen=True)
class ThresholdSuggestion:
    """Threshold adjustment suggestion.

    Attributes:
        current: Current threshold (midpoint of score range).
        suggested: Suggested optimal threshold.
        expected_f1: Expected F1 score at the suggested threshold.
    """

    current: float
    suggested: float
    expected_f1: float


@dataclass(frozen=True)
class BackendSuggestion:
    """Backend promotion suggestion.

    Attributes:
        suggestion: Human-readable suggestion text.
        labeled_count: Number of labeled samples available.
    """

    suggestion: str
    labeled_count: int


@dataclass(frozen=True)
class AnalysisReport:
    """Complete analysis report.

    Attributes:
        total_records: Total number of log records analysed.
        period_start: Start of the analysis period (ISO format).
        period_end: End of the analysis period (ISO format).
        psi: PSI result comparing first/second halves, or ``None``.
        ks: KS test result, or ``None``.
        fp_rates: False positive rate time-series.
        fn_rates: False negative rate time-series.
        threshold_suggestion: Threshold adjustment suggestion, or ``None``.
        backend_suggestion: Backend promotion suggestion, or ``None``.
        warnings: List of warning messages.
    """

    total_records: int
    period_start: str
    period_end: str
    psi: PSIResult | None = None
    ks: KSResult | None = None
    fp_rates: list[RatePoint] = field(default_factory=list)
    fn_rates: list[RatePoint] = field(default_factory=list)
    threshold_suggestion: ThresholdSuggestion | None = None
    backend_suggestion: BackendSuggestion | None = None
    warnings: list[str] = field(default_factory=list)


def compute_psi(
    reference: list[float],
    current: list[float],
    num_bins: int = _NUM_BINS,
) -> PSIResult:
    """Compute the Population Stability Index between two score distributions.

    Delegates to :func:`tobira.monitoring.drift.compute_psi` for the
    algorithm and wraps the result in a :class:`PSIResult`.

    Args:
        reference: Baseline score distribution.
        current: Current score distribution.
        num_bins: Number of bins for the histogram.

    Returns:
        A PSIResult with the PSI value and severity level.
    """
    from tobira.monitoring.drift import compute_psi as _compute_psi

    psi = _compute_psi(reference, current, bins=num_bins)
    return PSIResult(psi=psi, level=_classify_psi(psi))


def compute_ks(
    reference: list[float],
    current: list[float],
) -> KSResult:
    """Compute the Kolmogorov-Smirnov statistic between two distributions.

    Delegates to :func:`tobira.monitoring.drift.compute_ks_test` for the
    algorithm and wraps the result in a :class:`KSResult`.

    Args:
        reference: Baseline score distribution.
        current: Current score distribution.

    Returns:
        A KSResult with the statistic and significance flag.
    """
    from tobira.monitoring.drift import compute_ks_test as _compute_ks_test

    stat, pvalue = _compute_ks_test(reference, current)
    return KSResult(statistic=stat, significant=pvalue < 0.05)


def _filter_by_period(
    records: list[dict[str, object]],
    period_days: int | None,
) -> list[dict[str, object]]:
    """Filter records to the most recent ``period_days`` days."""
    if period_days is None:
        return records

    now = datetime.now(timezone.utc)
    result: list[dict[str, object]] = []
    for r in records:
        ts = r.get("timestamp")
        if not isinstance(ts, str):
            continue
        try:
            dt = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            continue
        diff = (now - dt).total_seconds()
        if diff <= period_days * 86400:
            result.append(r)
    return result


def _compute_rates(
    records: list[dict[str, object]],
    threshold: float,
) -> tuple[list[RatePoint], list[RatePoint]]:
    """Compute daily FP and FN rates.

    Records with a ``"label"`` field are used as ground truth.

    Returns:
        Tuple of (fp_rates, fn_rates).
    """
    daily: dict[str, list[tuple[object, float]]] = {}
    for r in records:
        label = r.get("label")
        score = r.get("score")
        ts = r.get("timestamp")
        if not isinstance(score, (int, float)) or not isinstance(ts, str):
            continue
        if label is None:
            continue
        day = ts[:10]
        daily.setdefault(day, []).append((label, float(score)))

    fp_rates: list[RatePoint] = []
    fn_rates: list[RatePoint] = []

    for day in sorted(daily):
        items = daily[day]
        n = len(items)
        fp = sum(1 for lbl, s in items if lbl == 0 and s >= threshold)
        fn = sum(1 for lbl, s in items if lbl == 1 and s < threshold)
        negatives = sum(1 for lbl, _ in items if lbl == 0)
        positives = sum(1 for lbl, _ in items if lbl == 1)

        fp_rate = fp / negatives if negatives > 0 else 0.0
        fn_rate = fn / positives if positives > 0 else 0.0

        fp_rates.append(RatePoint(period=day, rate=round(fp_rate, 4), count=n))
        fn_rates.append(RatePoint(period=day, rate=round(fn_rate, 4), count=n))

    return fp_rates, fn_rates


def _suggest_threshold(
    records: list[dict[str, object]],
) -> ThresholdSuggestion | None:
    """Suggest optimal threshold using labeled records."""
    y_true: list[int] = []
    scores: list[float] = []
    for r in records:
        label = r.get("label")
        score = r.get("score")
        if isinstance(label, int) and isinstance(score, (int, float)):
            y_true.append(label)
            scores.append(float(score))

    if len(y_true) < _MIN_SAMPLES:
        return None
    if len(set(y_true)) < 2:
        return None

    from tobira.evaluation.threshold import find_best_threshold_f1

    result = find_best_threshold_f1(y_true, scores)

    current = (min(scores) + max(scores)) / 2
    return ThresholdSuggestion(
        current=round(current, 4),
        suggested=round(result.threshold, 4),
        expected_f1=round(result.f1, 4),
    )


def _suggest_backend(
    records: list[dict[str, object]],
) -> BackendSuggestion | None:
    """Suggest backend promotion based on labeled sample count."""
    labeled = sum(
        1 for r in records
        if isinstance(r.get("label"), int)
    )

    if labeled >= 1000:
        return BackendSuggestion(
            suggestion="Sufficient labeled data for BERT/ONNX fine-tuning.",
            labeled_count=labeled,
        )
    if labeled >= 200:
        return BackendSuggestion(
            suggestion="Sufficient labeled data for fastText training.",
            labeled_count=labeled,
        )
    return None


def analyze(
    records: list[dict[str, object]],
    period_days: int | None = None,
) -> AnalysisReport:
    """Run a complete analysis on prediction log records.

    Args:
        records: List of prediction log records (dicts with ``timestamp``,
            ``score``, ``label``, etc.).
        period_days: If set, only analyse records from the last N days.

    Returns:
        An AnalysisReport with all analysis results.
    """
    filtered = _filter_by_period(records, period_days)
    warnings: list[str] = []

    if len(filtered) == 0:
        return AnalysisReport(
            total_records=0,
            period_start="",
            period_end="",
            warnings=["No records found for the specified period."],
        )

    timestamps = [
        str(r["timestamp"]) for r in filtered if "timestamp" in r
    ]
    period_start = min(timestamps) if timestamps else ""
    period_end = max(timestamps) if timestamps else ""

    scores: list[float] = []
    for r in filtered:
        s = r.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))

    # Score distribution drift
    psi: PSIResult | None = None
    ks: KSResult | None = None
    if len(scores) >= _MIN_SAMPLES * 2:
        mid = len(scores) // 2
        reference = scores[:mid]
        current = scores[mid:]
        psi = compute_psi(reference, current)
        ks = compute_ks(reference, current)
        if psi.level == "warning":
            warnings.append(
                f"Score distribution shift detected (PSI={psi.psi})."
            )
        elif psi.level == "alert":
            warnings.append(
                f"Significant score distribution shift (PSI={psi.psi})."
            )
        if ks.significant:
            warnings.append(
                f"KS test significant (statistic={ks.statistic})."
            )
    elif len(scores) < _MIN_SAMPLES:
        warnings.append(
            f"Insufficient samples ({len(scores)}) for drift analysis"
            f" (minimum {_MIN_SAMPLES})."
        )

    # FP/FN rates
    threshold = 0.5
    fp_rates, fn_rates = _compute_rates(filtered, threshold)

    # Threshold suggestion
    threshold_suggestion = _suggest_threshold(filtered)

    # Backend suggestion
    backend_suggestion = _suggest_backend(filtered)

    return AnalysisReport(
        total_records=len(filtered),
        period_start=period_start,
        period_end=period_end,
        psi=psi,
        ks=ks,
        fp_rates=fp_rates,
        fn_rates=fn_rates,
        threshold_suggestion=threshold_suggestion,
        backend_suggestion=backend_suggestion,
        warnings=warnings,
    )


def analyze_drift_from_redis(
    redis_url: str,
    *,
    key_prefix: str = "tobira:scores",
    window_seconds: int = 86400,
    psi_threshold: float = 0.2,
) -> DriftResult | None:
    """Run drift detection using scores stored in Redis.

    Compares the stored baseline distribution against scores accumulated in
    the current window.  Returns ``None`` if there are insufficient samples
    (fewer than ``_MIN_SAMPLES`` in either distribution).

    Args:
        redis_url: Redis connection URL.
        key_prefix: Key prefix for :class:`~tobira.monitoring.store.RedisScoreStore`.
        window_seconds: Time window (seconds) for current scores.
        psi_threshold: PSI value above which drift is flagged.

    Returns:
        A :class:`~tobira.monitoring.drift.DriftResult`, or ``None`` when
        data is insufficient.
    """
    from tobira.monitoring.drift import detect_drift
    from tobira.monitoring.store import RedisScoreStore

    store = RedisScoreStore(
        redis_url=redis_url,
        key_prefix=key_prefix,
        window_seconds=window_seconds,
    )
    baseline = store.get_baseline()
    current = store.get_scores()

    if len(baseline) < _MIN_SAMPLES or len(current) < _MIN_SAMPLES:
        return None

    return detect_drift(baseline, current, psi_threshold=psi_threshold)
