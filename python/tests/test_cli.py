"""Tests for tobira.cli."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestBuildParser:
    def test_parser_has_serve_subcommand(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["serve", "--config", "/tmp/c.toml"])
        assert args.command == "serve"

    def test_parser_serve_defaults(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["serve", "--config", "/tmp/c.toml"])
        assert args.config == "/tmp/c.toml"
        assert args.host == "127.0.0.1"
        assert args.port == 8000

    def test_parser_serve_custom_options(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["serve", "--config", "/tmp/c.toml", "--host", "0.0.0.0", "--port", "9000"]
        )
        assert args.host == "0.0.0.0"
        assert args.port == 9000

    def test_parser_serve_missing_config_exits(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["serve"])


class TestMain:
    def test_no_command_returns_1(self) -> None:
        from tobira.cli import main

        assert main([]) == 1

    @patch("tobira.cli.serve._run")
    def test_serve_dispatches_to_run(self, mock_run: MagicMock) -> None:
        from tobira.cli import main

        mock_run.return_value = 0
        result = main(["serve", "--config", "/tmp/c.toml"])

        assert result == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.config == "/tmp/c.toml"


class TestServeRun:
    @patch("tobira.serving.server.main")
    def test_run_calls_server_main(self, mock_server_main: MagicMock) -> None:
        from tobira.cli import main

        main([
            "serve", "--config", "/tmp/c.toml",
            "--host", "0.0.0.0", "--port", "9000",
        ])

        mock_server_main.assert_called_once_with(
            config_path="/tmp/c.toml",
            host="0.0.0.0",
            port=9000,
        )

    @patch("tobira.serving.server.main")
    def test_run_returns_0(self, mock_server_main: MagicMock) -> None:
        from tobira.cli import main

        result = main(["serve", "--config", "/tmp/c.toml"])
        assert result == 0


class TestHelpOutput:
    def test_help_exits_with_0(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["--help"])
        assert exc_info.value.code == 0

    def test_serve_help_exits_with_0(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["serve", "--help"])
        assert exc_info.value.code == 0
