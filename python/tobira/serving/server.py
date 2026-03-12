"""FastAPI application for spam prediction."""

import sys
from pathlib import Path
from typing import Any

from tobira.backends.factory import create_backend
from tobira.backends.protocol import BackendProtocol


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


def _load_config(config_path: str) -> dict[str, Any]:
    """Load a TOML configuration file.

    Args:
        config_path: Path to the TOML config file.

    Returns:
        Parsed configuration dict.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")

    if sys.version_info >= (3, 11):
        import tomllib  # type: ignore[import-not-found]
    else:
        try:
            import tomllib  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            import tomli as tomllib

    with open(path, "rb") as f:
        result: dict[str, Any] = tomllib.load(f)
        return result


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

    config = _load_config(config_path)
    backend_config = config["backend"]
    backend = create_backend(backend_config)

    app = create_app(backend)
    uvicorn.run(app, host=host, port=port, access_log=False)
