"""Bearer token authentication for the tobira API."""

from __future__ import annotations

import os
from typing import Any

_HEADER_NAME = "Authorization"
_SCHEME = "Bearer"
_ENV_VAR = "TOBIRA_API_KEY"


def get_api_key(serving_config: dict[str, Any] | None = None) -> str | None:
    """Resolve the API key from config or environment.

    Priority: environment variable ``TOBIRA_API_KEY`` > config value.

    Args:
        serving_config: Optional ``[serving]`` section from the TOML config.

    Returns:
        The API key string, or ``None`` if authentication is disabled.
    """
    env_key = os.environ.get(_ENV_VAR)
    if env_key:
        return env_key
    if serving_config:
        return serving_config.get("api_key")
    return None


def create_auth_dependency(api_key: str) -> Any:
    """Create a FastAPI dependency that enforces Bearer token auth.

    Args:
        api_key: The expected API key.

    Returns:
        A FastAPI dependency callable.
    """
    import fastapi
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    _bearer_scheme = HTTPBearer(auto_error=False)

    async def _verify_api_key(
        credentials: HTTPAuthorizationCredentials | None = fastapi.Depends(
            _bearer_scheme
        ),
    ) -> None:
        if credentials is None or credentials.credentials != api_key:
            raise fastapi.HTTPException(
                status_code=401,
                detail="Invalid or missing API key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    return _verify_api_key
