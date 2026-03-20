"""Tests for tobira.cli.doctor infrastructure checks."""

from __future__ import annotations

import builtins
import os
import textwrap
from unittest.mock import MagicMock, patch

_real_import = builtins.__import__


def _make_blocker(*blocked: str):  # type: ignore[no-untyped-def]
    """Return an import side-effect that blocks specific top-level packages."""

    def _import(name: str, *args: object, **kwargs: object) -> object:
        if name in blocked:
            raise ImportError(f"mocked: no module named {name!r}")
        return _real_import(name, *args, **kwargs)

    return _import


class TestCheckServingDeps:
    def test_returns_ok_when_installed(self) -> None:
        from tobira.cli.doctor import _check_serving_deps

        ok, msg = _check_serving_deps()
        assert ok is True
        assert "fastapi" in msg
        assert "uvicorn" in msg

    def test_returns_fail_when_both_missing(self) -> None:
        from tobira.cli.doctor import _check_serving_deps

        blocker = _make_blocker("fastapi", "uvicorn")
        with patch("builtins.__import__", side_effect=blocker):
            ok, msg = _check_serving_deps()
        assert ok is False
        assert "fastapi" in msg
        assert "uvicorn" in msg

    def test_partial_missing(self) -> None:
        from tobira.cli.doctor import _check_serving_deps

        with patch("builtins.__import__", side_effect=_make_blocker("uvicorn")):
            ok, msg = _check_serving_deps()
        assert ok is False
        assert "uvicorn" in msg
        assert "fastapi" not in msg


class TestCheckPathWritable:
    def test_writable_path(self, tmp_path: os.PathLike[str]) -> None:
        from tobira.cli.doctor import _check_path_writable

        target = str(tmp_path / "subdir" / "test.jsonl")  # type: ignore[operator]
        # Parent is tmp_path/subdir which doesn't exist yet
        ok, msg = _check_path_writable(target, "test")
        assert ok is False
        assert "does not exist" in msg

    def test_existing_writable_parent(self, tmp_path: os.PathLike[str]) -> None:
        from tobira.cli.doctor import _check_path_writable

        target = str(tmp_path / "test.jsonl")  # type: ignore[operator]
        ok, msg = _check_path_writable(target, "test")
        assert ok is True
        assert "writable" in msg

    def test_non_writable_parent(self, tmp_path: os.PathLike[str]) -> None:
        from pathlib import Path

        from tobira.cli.doctor import _check_path_writable

        readonly_dir = Path(str(tmp_path)) / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)

        try:
            target = str(readonly_dir / "test.jsonl")
            ok, msg = _check_path_writable(target, "test")
            assert ok is False
            assert "not writable" in msg
        finally:
            readonly_dir.chmod(0o755)

    def test_nonexistent_parent(self) -> None:
        from tobira.cli.doctor import _check_path_writable

        ok, msg = _check_path_writable(
            "/nonexistent/deep/path/file.jsonl", "test"
        )
        assert ok is False
        assert "does not exist" in msg


class TestCheckStore:
    def test_jsonl_returns_empty(self) -> None:
        from tobira.cli.doctor import _check_store

        results = _check_store({"type": "jsonl"})
        assert results == []

    def test_default_type_returns_empty(self) -> None:
        from tobira.cli.doctor import _check_store

        results = _check_store({})
        assert results == []

    def test_postgres_deps_available(self) -> None:
        from tobira.cli.doctor import _check_store

        # psycopg is installed in test env; no DSN so only import check
        results = _check_store({"type": "postgres"})
        assert len(results) == 1
        assert results[0][0] is True
        assert "psycopg" in results[0][1]

    def test_postgres_deps_missing(self) -> None:
        from tobira.cli.doctor import _check_store

        blocker = _make_blocker("psycopg", "psycopg_pool")
        with patch("builtins.__import__", side_effect=blocker):
            results = _check_store({"type": "postgres"})
        assert len(results) == 1
        assert results[0][0] is False
        assert "psycopg" in results[0][1]

    def test_postgres_connection_failure(self) -> None:
        from tobira.cli.doctor import _check_store

        results = _check_store({
            "type": "postgres",
            "dsn": "postgresql://invalid:invalid@localhost:1/nonexistent",
        })
        # First result: deps OK, second: connection failed
        assert len(results) == 2
        assert results[0][0] is True
        assert results[1][0] is False
        assert "connection failed" in results[1][1]

    def test_redis_deps_available(self) -> None:
        from tobira.cli.doctor import _check_store

        # redis is installed in test env; no redis_url so only import check
        results = _check_store({"type": "redis"})
        assert len(results) == 1
        assert results[0][0] is True
        assert "redis" in results[0][1]

    def test_redis_deps_missing(self) -> None:
        from tobira.cli.doctor import _check_store

        with patch("builtins.__import__", side_effect=_make_blocker("redis")):
            results = _check_store({"type": "redis"})
        assert len(results) == 1
        assert results[0][0] is False
        assert "redis" in results[0][1]

    def test_redis_connection_failure(self) -> None:
        from tobira.cli.doctor import _check_store

        results = _check_store({
            "type": "redis",
            "redis_url": "redis://localhost:1/0",
        })
        assert len(results) == 2
        assert results[0][0] is True
        assert results[1][0] is False
        assert "connection failed" in results[1][1]


class TestCheckMonitoringRedis:
    def test_deps_available(self) -> None:
        from tobira.cli.doctor import _check_monitoring_redis

        # redis installed; connection will fail with invalid URL
        results = _check_monitoring_redis("redis://localhost:1/0")
        assert len(results) == 2
        assert results[0][0] is True
        assert results[1][0] is False

    def test_deps_missing(self) -> None:
        from tobira.cli.doctor import _check_monitoring_redis

        with patch("builtins.__import__", side_effect=_make_blocker("redis")):
            results = _check_monitoring_redis("redis://localhost:6379/0")
        assert len(results) == 1
        assert results[0][0] is False
        assert "redis" in results[0][1]


class TestRunChecksInfra:
    """Integration tests for infrastructure checks within run_checks."""

    @patch("tobira.backends.factory.create_backend")
    def test_monitoring_log_path_checked_when_enabled(
        self, mock_create: MagicMock, tmp_path: os.PathLike[str]
    ) -> None:
        from tobira.cli.doctor import run_checks

        log_file = str(tmp_path / "predictions.jsonl")  # type: ignore[operator]
        toml_file = tmp_path / "config.toml"  # type: ignore[operator]
        toml_content = textwrap.dedent(f"""\
            [backend]
            type = "fasttext"
            model_path = "/tmp/model.bin"

            [monitoring]
            enabled = true
            log_path = "{log_file}"
        """)
        toml_file.write_text(toml_content, encoding="utf-8")
        mock_create.return_value = MagicMock()

        results = run_checks(config_path=str(toml_file))
        log_results = [(ok, msg) for ok, msg in results if "monitoring log_path" in msg]
        assert len(log_results) == 1
        assert log_results[0][0] is True

    @patch("tobira.backends.factory.create_backend")
    def test_monitoring_not_checked_when_disabled(
        self, mock_create: MagicMock, tmp_path: os.PathLike[str]
    ) -> None:
        from tobira.cli.doctor import run_checks

        toml_file = tmp_path / "config.toml"  # type: ignore[operator]
        toml_content = textwrap.dedent("""\
            [backend]
            type = "fasttext"
            model_path = "/tmp/model.bin"
        """)
        toml_file.write_text(toml_content, encoding="utf-8")
        mock_create.return_value = MagicMock()

        results = run_checks(config_path=str(toml_file))
        log_results = [(ok, msg) for ok, msg in results if "monitoring log_path" in msg]
        assert len(log_results) == 0

    @patch("tobira.backends.factory.create_backend")
    def test_feedback_store_path_checked_when_enabled(
        self, mock_create: MagicMock, tmp_path: os.PathLike[str]
    ) -> None:
        from tobira.cli.doctor import run_checks

        store_file = str(tmp_path / "feedback.jsonl")  # type: ignore[operator]
        toml_file = tmp_path / "config.toml"  # type: ignore[operator]
        toml_content = textwrap.dedent(f"""\
            [backend]
            type = "fasttext"
            model_path = "/tmp/model.bin"

            [feedback]
            enabled = true
            store_path = "{store_file}"
        """)
        toml_file.write_text(toml_content, encoding="utf-8")
        mock_create.return_value = MagicMock()

        results = run_checks(config_path=str(toml_file))
        fb_results = [(ok, msg) for ok, msg in results if "feedback store_path" in msg]
        assert len(fb_results) == 1
        assert fb_results[0][0] is True

    @patch("tobira.backends.factory.create_backend")
    def test_store_checked_when_configured(
        self, mock_create: MagicMock, tmp_path: os.PathLike[str]
    ) -> None:
        from tobira.cli.doctor import run_checks

        toml_file = tmp_path / "config.toml"  # type: ignore[operator]
        toml_content = textwrap.dedent("""\
            [backend]
            type = "fasttext"
            model_path = "/tmp/model.bin"

            [store]
            type = "postgres"
        """)
        toml_file.write_text(toml_content, encoding="utf-8")
        mock_create.return_value = MagicMock()

        results = run_checks(config_path=str(toml_file))
        store_results = [
            (ok, msg) for ok, msg in results
            if "store" in msg.lower() and "psycopg" in msg
        ]
        assert len(store_results) == 1
        assert store_results[0][0] is True

    def test_serving_deps_always_checked(self) -> None:
        from tobira.cli.doctor import run_checks

        results = run_checks(config_path="/nonexistent/config.toml")
        serving_results = [
            (ok, msg) for ok, msg in results if "serving dependencies" in msg
        ]
        assert len(serving_results) == 1
