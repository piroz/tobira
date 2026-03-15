"""tobira.monitoring - prediction metrics logging and drift detection."""

__all__ = [
    "DeploymentPhase",
    "PhaseAdvice",
    "PhaseTransitionConfig",
    "PredictionCollector",
    "RetrainConfig",
    "RetrainEvent",
    "analyze",
    "analyze_drift_from_redis",
    "append_record",
    "check_retrain_needed",
    "detect_drift",
    "load_retrain_config",
    "read_records",
    "RedisScoreStore",
    "trigger_retrain",
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
    if name in ("RetrainConfig", "RetrainEvent", "check_retrain_needed",
                "load_retrain_config", "trigger_retrain"):
        from tobira.monitoring import retrain

        return getattr(retrain, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
