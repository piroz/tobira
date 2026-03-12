"""FastAPI application for spam prediction."""

from typing import Any

from tobira.backends.factory import create_backend
from tobira.backends.protocol import BackendProtocol
from tobira.config import load_toml


def _import_deps() -> tuple[Any, Any]:
    """Lazy-import FastAPI and uvicorn."""
    try:
        import fastapi
        import uvicorn
    except ImportError:
        raise ImportError(
            "serving dependencies are not installed. "
            "Install them with: pip install tobira[serving]"
        ) from None
    return fastapi, uvicorn


def create_app(backend: BackendProtocol) -> Any:
    """Create a FastAPI application with the given backend.

    Args:
        backend: A backend instance implementing BackendProtocol.

    Returns:
        A FastAPI application.
    """
    fastapi, _ = _import_deps()

    import tobira
    from tobira.serving.schemas import HealthResponse, PredictRequest, PredictResponse

    app = fastapi.FastAPI(
        title="tobira",
        description="Spam prediction API powered by tobira.",
        version=tobira.__version__,
    )
    app.state.backend = backend

    @app.post("/predict", response_model=PredictResponse, tags=["prediction"])
    async def predict(req: PredictRequest) -> PredictResponse:
        result = app.state.backend.predict(req.text)
        return PredictResponse(
            label=result.label,
            score=result.score,
            labels=result.labels,
        )

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    return app


def main(config_path: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the server from a TOML config file.

    Args:
        config_path: Path to the TOML configuration file.
        host: Host to bind to.
        port: Port to bind to.
    """
    _, uvicorn = _import_deps()

    config = load_toml(config_path)
    backend_config = config["backend"]
    backend = create_backend(backend_config)

    app = create_app(backend)
    uvicorn.run(app, host=host, port=port, access_log=False)
