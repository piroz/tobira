"""tobira.monitoring - prediction metrics logging."""

__all__ = ["PredictionCollector", "analyze", "append_record", "read_records"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "PredictionCollector":
        from tobira.monitoring.collector import PredictionCollector

        return PredictionCollector
    if name == "analyze":
        from tobira.monitoring.analyzer import analyze

        return analyze
    if name in ("append_record", "read_records"):
        from tobira.monitoring import store

        return getattr(store, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
