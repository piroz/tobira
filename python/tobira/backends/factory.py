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
    backend_type = config["type"]

    if backend_type == "fasttext":
        from tobira.backends.fasttext import FastTextBackend

        return FastTextBackend(model_path=config["model_path"])

    if backend_type == "bert":
        from tobira.backends.bert import BertBackend

        model_name = config.get("model_name")
        device = config.get("device")
        kwargs: dict[str, Any] = {}
        if model_name is not None:
            kwargs["model_name"] = model_name
        if device is not None:
            kwargs["device"] = device
        return BertBackend(**kwargs)

    if backend_type == "onnx":
        from tobira.backends.onnx import OnnxBackend

        onnx_kwargs: dict[str, Any] = {"model_path": config["model_path"]}
        model_name = config.get("model_name")
        if model_name is not None:
            onnx_kwargs["model_name"] = model_name
        return OnnxBackend(**onnx_kwargs)

    raise ValueError(f"unknown backend type: {backend_type!r}")
