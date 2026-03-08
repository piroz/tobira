"""tobira.serving - HTTP API server for spam prediction."""

__all__ = ["create_app"]


def __getattr__(name: str):  # type: ignore[no-untyped-def]
    if name == "create_app":
        from tobira.serving.server import create_app

        return create_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
