"""Factory for creating backend instances from configuration."""

from __future__ import annotations

from typing import Any

from tobira.backends.protocol import BackendProtocol


def create_backend(config: dict[str, Any]) -> BackendProtocol:
    """Create a backend instance from a configuration dict.

    Args:
        config: Must contain a "type" key (e.g. "fasttext") and any
            backend-specific options.

    Returns:
        A backend instance implementing BackendProtocol.

    Raises:
        ValueError: If the backend type is unknown.
        KeyError: If required configuration keys are missing.

    Example::

        backend = create_backend({
            "type": "fasttext",
            "model_path": "/path/to/model.bin",
        })
    """
    backend_type = config.get("type")
    if backend_type is None:
        raise KeyError("config must contain a 'type' key")

    if backend_type == "fasttext":
        from tobira.backends.fasttext import FastTextBackend

        model_path = config.get("model_path")
        if model_path is None:
            raise KeyError("fasttext config must contain a 'model_path' key")
        return FastTextBackend(model_path=model_path)

    raise ValueError(f"unknown backend type: {backend_type!r}")
