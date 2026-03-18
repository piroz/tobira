"""Drift detection, FP/FN analysis, threshold suggestion, and phase advisor."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

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


class DeploymentPhase(Enum):
    """Deployment phase in the staged rollout flow.

    See STRATEGY.md section 6 for details.
    """

    A = "A"
    B = "B"
    C = "C"
    D = "D"


@dataclass(frozen=True)
class PhaseTransitionCondition:
    """A single condition for phase transition.

    Attributes:
        description: Human-readable description of the condition.
        met: Whether the condition is satisfied.
        detail: Additional detail (e.g. current value vs threshold).
    """

    description: str
    met: bool
    detail: str


@dataclass(frozen=True)
class PhaseAdvice:
    """Phase transition recommendation.

    Attributes:
        current_phase: Current deployment phase.
        conditions: List of conditions checked for transition.
        ready: Whether all conditions are met for transition.
        recommendation: Human-readable recommendation text.
    """

    current_phase: DeploymentPhase
    conditions: tuple[PhaseTransitionCondition, ...]
    ready: bool
    recommendation: str


@dataclass(frozen=True)
class PhaseTransitionConfig:
    """Configurable thresholds for phase transition conditions.

    Attributes:
        min_labeled_for_training: Minimum labeled samples to start training
            (Phase A→B). Default 500.
        f1_threshold: Minimum F1 score to deploy model (Phase B→C).
            Default 0.95.
        min_eval_samples: Minimum evaluation samples for F1 calculation
            (Phase B→C). Default 100.
        min_operation_days: Minimum days of operation before Phase C→D.
            Default 30.
    """

    min_labeled_for_training: int = 500
    f1_threshold: float = 0.95
    min_eval_samples: int = 100
    min_operation_days: int = 30


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
    phase_advice: PhaseAdvice | None = None
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


def _advise_phase_transition(
    records: list[dict[str, object]],
    current_phase: DeploymentPhase,
    config: PhaseTransitionConfig | None = None,
) -> PhaseAdvice:
    """Evaluate phase transition conditions and produce advice.

    Args:
        records: Prediction log records.
        current_phase: The current deployment phase.
        config: Transition condition thresholds. Uses defaults if ``None``.

    Returns:
        A PhaseAdvice with conditions checked and recommendation.
    """
    cfg = config or PhaseTransitionConfig()
    conditions: list[PhaseTransitionCondition] = []

    if current_phase == DeploymentPhase.A:
        # A→B: need enough labeled data to start training
        labeled = sum(
            1 for r in records if isinstance(r.get("label"), int)
        )
        met = labeled >= cfg.min_labeled_for_training
        conditions.append(PhaseTransitionCondition(
            description="Labeled data count",
            met=met,
            detail=f"{labeled} / {cfg.min_labeled_for_training} samples",
        ))
        ready = met
        if ready:
            recommendation = (
                f"Phase B（ML モデル学習）への移行を推奨します。"
                f"ラベル付きデータが {labeled} 件蓄積されています。"
            )
        else:
            recommendation = (
                f"Phase B への移行にはラベル付きデータが"
                f"あと {cfg.min_labeled_for_training - labeled} 件必要です。"
            )

    elif current_phase == DeploymentPhase.B:
        # B→C: need F1 above threshold with enough eval samples
        y_true: list[int] = []
        scores: list[float] = []
        for r in records:
            label = r.get("label")
            score = r.get("score")
            if isinstance(label, int) and isinstance(score, (int, float)):
                y_true.append(label)
                scores.append(float(score))

        has_enough_samples = len(y_true) >= cfg.min_eval_samples
        conditions.append(PhaseTransitionCondition(
            description="Evaluation sample count",
            met=has_enough_samples,
            detail=f"{len(y_true)} / {cfg.min_eval_samples} samples",
        ))

        f1_met = False
        f1_value = 0.0
        if has_enough_samples and len(set(y_true)) >= 2:
            from tobira.evaluation.threshold import find_best_threshold_f1

            result = find_best_threshold_f1(y_true, scores)
            f1_value = result.f1
            f1_met = f1_value >= cfg.f1_threshold

        conditions.append(PhaseTransitionCondition(
            description="F1 score",
            met=f1_met,
            detail=f"{f1_value:.3f} / {cfg.f1_threshold} threshold",
        ))

        ready = has_enough_samples and f1_met
        if ready:
            recommendation = (
                f"Phase C（ML モデル投入）への移行を推奨します。"
                f"F1 スコア {f1_value:.3f} が"
                f"閾値 {cfg.f1_threshold} をクリアしています。"
                f" `tobira doctor` で移行前チェックを実行してください。"
            )
        else:
            parts: list[str] = []
            if not has_enough_samples:
                parts.append(
                    f"評価データがあと"
                    f" {cfg.min_eval_samples - len(y_true)} 件必要です"
                )
            if not f1_met:
                parts.append(
                    f"F1 スコア {f1_value:.3f} が"
                    f"閾値 {cfg.f1_threshold} に未到達です"
                )
            recommendation = (
                "Phase C への移行条件が未達です。" + "。".join(parts) + "。"
            )

    elif current_phase == DeploymentPhase.C:
        # C→D: operation period + no drift
        timestamps: list[datetime] = []
        for r in records:
            ts = r.get("timestamp")
            if isinstance(ts, str):
                try:
                    timestamps.append(datetime.fromisoformat(ts))
                except (ValueError, TypeError):
                    pass

        operation_days = 0
        if len(timestamps) >= 2:
            span = max(timestamps) - min(timestamps)
            operation_days = span.days

        days_met = operation_days >= cfg.min_operation_days
        conditions.append(PhaseTransitionCondition(
            description="Operation period",
            met=days_met,
            detail=f"{operation_days} / {cfg.min_operation_days} days",
        ))

        # Check for drift in scores
        drift_scores: list[float] = []
        for r in records:
            s = r.get("score")
            if isinstance(s, (int, float)):
                drift_scores.append(float(s))

        no_drift = True
        if len(drift_scores) >= _MIN_SAMPLES * 2:
            mid = len(drift_scores) // 2
            psi_result = compute_psi(drift_scores[:mid], drift_scores[mid:])
            no_drift = psi_result.level == "ok"

        conditions.append(PhaseTransitionCondition(
            description="No score drift detected",
            met=no_drift,
            detail="ok" if no_drift else "drift detected",
        ))

        ready = days_met and no_drift
        if ready:
            recommendation = (
                "Phase D（継続的改善）への移行を推奨します。"
                f"運用期間 {operation_days} 日、ドリフト検知なし。"
            )
        else:
            parts_cd: list[str] = []
            if not days_met:
                parts_cd.append(
                    f"運用期間があと"
                    f" {cfg.min_operation_days - operation_days} 日必要です"
                )
            if not no_drift:
                parts_cd.append("スコア分布のドリフトが検知されています")
            recommendation = (
                "Phase D への移行条件が未達です。"
                + "。".join(parts_cd) + "。"
            )

    else:
        # Phase D: no further transition
        ready = False
        recommendation = (
            "Phase D（継続的改善）で運用中です。"
            "定期的な再学習とドリフト検知を継続してください。"
        )

    return PhaseAdvice(
        current_phase=current_phase,
        conditions=tuple(conditions),
        ready=ready,
        recommendation=recommendation,
    )


def analyze(
    records: list[dict[str, object]],
    period_days: int | None = None,
    *,
    current_phase: DeploymentPhase | None = None,
    phase_config: PhaseTransitionConfig | None = None,
) -> AnalysisReport:
    """Run a complete analysis on prediction log records.

    Args:
        records: List of prediction log records (dicts with ``timestamp``,
            ``score``, ``label``, etc.).
        period_days: If set, only analyse records from the last N days.
        current_phase: Current deployment phase. If set, phase transition
            advice is included in the report.
        phase_config: Custom thresholds for phase transition conditions.

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

    # Phase transition advice
    phase_advice: PhaseAdvice | None = None
    if current_phase is not None:
        phase_advice = _advise_phase_transition(
            filtered, current_phase, phase_config,
        )

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
        phase_advice=phase_advice,
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


def notify_analysis_results(
    report: AnalysisReport,
    dispatcher: object,
) -> list[bool]:
    """Send notifications for noteworthy findings in an analysis report.

    Sends notifications for drift alerts, threshold suggestions, and phase
    transition readiness.  Only fires when ``report.warnings`` contains
    alert-level messages or when the phase transition is ready.

    Args:
        report: The analysis report to inspect.
        dispatcher: A :class:`~tobira.monitoring.notifier.NotificationDispatcher`.

    Returns:
        Aggregated list of send results across all notifications fired.
    """
    from tobira.monitoring.notifier import NotificationDispatcher

    if not isinstance(dispatcher, NotificationDispatcher):
        return []

    results: list[bool] = []

    # Drift alerts
    if report.psi is not None and report.psi.level == "alert":
        results.extend(dispatcher.notify(
            "Score distribution drift detected",
            f"PSI={report.psi.psi:.4f} — significant shift in prediction "
            f"score distribution. Investigate model performance.",
            severity="critical",
        ))
    elif report.psi is not None and report.psi.level == "warning":
        results.extend(dispatcher.notify(
            "Score distribution shift warning",
            f"PSI={report.psi.psi:.4f} — moderate shift detected in "
            f"prediction scores. Monitor closely.",
            severity="warning",
        ))

    # Phase transition readiness
    if report.phase_advice is not None and report.phase_advice.ready:
        results.extend(dispatcher.notify(
            "Phase transition recommended",
            report.phase_advice.recommendation,
            severity="info",
        ))

    return results
