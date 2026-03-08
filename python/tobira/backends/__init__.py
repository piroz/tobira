"""tobira.backends - ML backend abstractions and implementations."""

from tobira.backends.factory import create_backend
from tobira.backends.protocol import BackendProtocol, PredictionResult

__all__ = ["BackendProtocol", "PredictionResult", "create_backend"]
