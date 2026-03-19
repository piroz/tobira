"""Tests for tobira.errors module."""

from __future__ import annotations

import pytest

from tobira.errors import (
    BACKEND_UNKNOWN_TYPE,
    CONFIG_NOT_FOUND,
    SERVING_NOT_READY,
    TobiraError,
    format_cli_error,
)


class TestTobiraError:
    def test_basic(self) -> None:
        err = TobiraError(BACKEND_UNKNOWN_TYPE, "Unknown backend: 'foo'")
        assert err.code == BACKEND_UNKNOWN_TYPE
        assert err.detail == "Unknown backend: 'foo'"
        assert err.hint is None
        assert "Unknown backend: 'foo'" in str(err)

    def test_with_hint(self) -> None:
        err = TobiraError(
            CONFIG_NOT_FOUND,
            "Config not found: /tmp/c.toml",
            hint="Check the file path.",
        )
        assert err.code == CONFIG_NOT_FOUND
        assert err.hint == "Check the file path."
        assert "Hint:" in str(err)

    def test_inherits_from_exception(self) -> None:
        err = TobiraError(SERVING_NOT_READY, "Not ready")
        assert isinstance(err, Exception)

    def test_raise_and_catch(self) -> None:
        with pytest.raises(TobiraError, match="Not ready"):
            raise TobiraError(SERVING_NOT_READY, "Not ready")


class TestFormatCliError:
    def test_basic(self) -> None:
        msg = format_cli_error(BACKEND_UNKNOWN_TYPE, "Unknown backend: 'foo'")
        assert msg == "Error [BACKEND_UNKNOWN_TYPE]: Unknown backend: 'foo'"

    def test_with_hint(self) -> None:
        msg = format_cli_error(
            CONFIG_NOT_FOUND,
            "Config not found.",
            hint="Check the path.",
        )
        assert "Error [CONFIG_NOT_FOUND]: Config not found." in msg
        assert "Hint: Check the path." in msg

    def test_without_hint(self) -> None:
        msg = format_cli_error(SERVING_NOT_READY, "Service not ready.")
        assert "Hint:" not in msg


class TestErrorCodes:
    def test_codes_are_strings(self) -> None:
        from tobira import errors

        code_names = [
            name for name in dir(errors)
            if name.isupper() and not name.startswith("_")
        ]
        for name in code_names:
            value = getattr(errors, name)
            if isinstance(value, str):
                assert value == name, f"Error code {name} value should equal its name"
