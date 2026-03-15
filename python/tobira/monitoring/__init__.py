"""tobira.monitoring - prediction metrics logging and drift detection."""

__all__ = [
    "DeploymentPhase",
    "PhaseAdvice",
    "PhaseTransitionConfig",
    "PredictionCollector",
    "analyze",
    "analyze_drift_from_redis",
    "append_record",
    "detect_drift",
    "read_records",
    "RedisScoreStore",
]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "PredictionCollector":
        from tobira.monitoring.collector import PredictionCollector

        return PredictionCollector
    if name in ("DeploymentPhase", "PhaseAdvice", "PhaseTransitionConfig"):
        from tobira.monitoring import analyzer

        return getattr(analyzer, name)
    if name == "analyze":
        from tobira.monitoring.analyzer import analyze

        return analyze
    if name == "analyze_drift_from_redis":
        from tobira.monitoring.analyzer import analyze_drift_from_redis

        return analyze_drift_from_redis
    if name == "detect_drift":
        from tobira.monitoring.drift import detect_drift

        return detect_drift
    if name in ("append_record", "read_records"):
        from tobira.monitoring import store

        return getattr(store, name)
    if name == "RedisScoreStore":
        from tobira.monitoring.store import RedisScoreStore

        return RedisScoreStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
