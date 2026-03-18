"""Factory for creating backend instances from configuration."""

from __future__ import annotations

import logging
import sys
from typing import Any

from tobira.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)

_BUILTIN_BACKENDS = frozenset(
    {"fasttext", "bert", "onnx", "ollama", "llm_api", "two_stage", "ensemble"}
)

ENTRY_POINT_GROUP = "tobira.backends"


def _load_plugin_backends() -> dict[str, Any]:
    """Discover backend plugins registered via entry_points."""
    from importlib.metadata import entry_points

    if sys.version_info >= (3, 10):
        eps = entry_points(group=ENTRY_POINT_GROUP)
    else:
        # Python 3.9: entry_points() returns a dict
        all_eps = entry_points()
        eps = all_eps.get(ENTRY_POINT_GROUP, [])

    plugins: dict[str, Any] = {}
    for ep in eps:
        if ep.name in _BUILTIN_BACKENDS:
            logger.warning(
                "Plugin %r provides backend %r which conflicts with a "
                "built-in backend; the built-in will be used instead.",
                ep.value,
                ep.name,
            )
            continue
        if ep.name in plugins:
            logger.warning(
                "Multiple plugins provide backend %r; using the first one found.",
                ep.name,
            )
            continue
        plugins[ep.name] = ep
    return plugins


def list_backends() -> list[str]:
    """Return a sorted list of all available backend type names.

    Includes both built-in backends and any discovered plugins.
    """
    plugins = _load_plugin_backends()
    return sorted(_BUILTIN_BACKENDS | set(plugins.keys()))


def create_backend(config: dict[str, Any]) -> BackendProtocol:
    """Create a backend instance from a configuration dict.

    Args:
        config: Must contain a "type" key (e.g. "fasttext") and any
            backend-specific options.  For plugin backends, all keys
            except "type" are forwarded to the backend constructor.

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

    if backend_type == "ollama":
        from tobira.backends.ollama import OllamaBackend

        ollama_kwargs: dict[str, Any] = {}
        model = config.get("model")
        if model is not None:
            ollama_kwargs["model"] = model
        base_url = config.get("base_url")
        if base_url is not None:
            ollama_kwargs["base_url"] = base_url
        timeout = config.get("timeout")
        if timeout is not None:
            ollama_kwargs["timeout"] = timeout
        return OllamaBackend(**ollama_kwargs)

    if backend_type == "llm_api":
        from tobira.backends.llm_api import LlmApiBackend

        llm_kwargs: dict[str, Any] = {}
        model = config.get("model")
        if model is not None:
            llm_kwargs["model"] = model
        base_url = config.get("base_url")
        if base_url is not None:
            llm_kwargs["base_url"] = base_url
        api_key = config.get("api_key")
        if api_key is not None:
            llm_kwargs["api_key"] = api_key
        timeout = config.get("timeout")
        if timeout is not None:
            llm_kwargs["timeout"] = timeout
        return LlmApiBackend(**llm_kwargs)

    if backend_type == "two_stage":
        from tobira.backends.two_stage import TwoStageBackend

        first_stage = create_backend(config["first_stage"])
        second_stage = create_backend(config["second_stage"])
        ts_kwargs: dict[str, Any] = {}
        grey_zone = config.get("grey_zone")
        if grey_zone is not None:
            ts_kwargs["grey_zone"] = tuple(grey_zone)
        return TwoStageBackend(
            first_stage=first_stage, second_stage=second_stage, **ts_kwargs
        )

    if backend_type == "ensemble":
        from tobira.backends.ensemble import EnsembleBackend

        child_backends = [create_backend(c) for c in config["backends"]]
        ens_kwargs: dict[str, Any] = {}
        weights = config.get("weights")
        if weights is not None:
            ens_kwargs["weights"] = list(weights)
        strategy = config.get("strategy")
        if strategy is not None:
            ens_kwargs["strategy"] = strategy
        return EnsembleBackend(backends=child_backends, **ens_kwargs)

    # Try plugin backends discovered via entry_points
    plugins = _load_plugin_backends()
    if backend_type in plugins:
        ep = plugins[backend_type]
        backend_class = ep.load()
        plugin_config = {k: v for k, v in config.items() if k != "type"}
        return backend_class(**plugin_config)  # type: ignore[no-any-return]

    raise ValueError(f"unknown backend type: {backend_type!r}")
