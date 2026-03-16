"""Tests for tobira.milter and tobira.cli.milter_cmd."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# MilterConfig tests
# ---------------------------------------------------------------------------


class TestMilterConfig:
    def test_load_defaults(self, tmp_path: Path) -> None:
        """Loading a config with only [milter] section uses defaults."""
        from tobira.milter.config import MilterConfig, load_config

        conf = tmp_path / "milter.conf"
        conf.write_text("[milter]\n", encoding="utf-8")

        cfg = load_config(conf)

        assert cfg == MilterConfig()
        assert cfg.api_url == "http://127.0.0.1:8000"
        assert cfg.socket == "unix:/var/run/tobira/milter.sock"
        assert cfg.timeout == 10
        assert cfg.fail_action == "accept"
        assert cfg.reject_threshold == 0.9
        assert cfg.add_headers is True

    def test_load_custom_values(self, tmp_path: Path) -> None:
        from tobira.milter.config import load_config

        conf = tmp_path / "milter.conf"
        conf.write_text(
            "[milter]\n"
            "api_url = http://localhost:9000\n"
            "socket = inet:9900@127.0.0.1\n"
            "timeout = 5\n"
            "fail_action = tempfail\n"
            "reject_threshold = 0.8\n"
            "add_headers = false\n",
            encoding="utf-8",
        )

        cfg = load_config(conf)

        assert cfg.api_url == "http://localhost:9000"
        assert cfg.socket == "inet:9900@127.0.0.1"
        assert cfg.timeout == 5
        assert cfg.fail_action == "tempfail"
        assert cfg.reject_threshold == 0.8
        assert cfg.add_headers is False

    def test_load_missing_file(self) -> None:
        from tobira.milter.config import load_config

        with pytest.raises(FileNotFoundError, match="configuration file not found"):
            load_config("/nonexistent/milter.conf")

    def test_load_invalid_fail_action(self, tmp_path: Path) -> None:
        from tobira.milter.config import load_config

        conf = tmp_path / "milter.conf"
        conf.write_text(
            "[milter]\nfail_action = reject\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="fail_action must be"):
            load_config(conf)

    def test_load_invalid_reject_threshold(self, tmp_path: Path) -> None:
        from tobira.milter.config import load_config

        conf = tmp_path / "milter.conf"
        conf.write_text(
            "[milter]\nreject_threshold = 1.5\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="reject_threshold must be"):
            load_config(conf)

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """A file with no [milter] section returns defaults."""
        from tobira.milter.config import MilterConfig, load_config

        conf = tmp_path / "milter.conf"
        conf.write_text("# empty\n", encoding="utf-8")

        cfg = load_config(conf)
        assert cfg == MilterConfig()


# ---------------------------------------------------------------------------
# TobiraFilter tests
# ---------------------------------------------------------------------------


class TestTobiraFilter:
    def test_configure_sets_class_attrs(self) -> None:
        from tobira.milter.filter import TobiraFilter

        TobiraFilter.configure(
            api_url="http://test:8080",
            timeout=5,
            fail_action="tempfail",
            reject_threshold=0.7,
            add_headers=False,
        )

        assert TobiraFilter._api_url == "http://test:8080"
        assert TobiraFilter._timeout == 5
        assert TobiraFilter._fail_action == "tempfail"
        assert TobiraFilter._reject_threshold == 0.7
        assert TobiraFilter._add_headers is False

        # Reset to defaults
        TobiraFilter.configure(
            api_url="http://127.0.0.1:8000",
            timeout=10,
            fail_action="accept",
            reject_threshold=0.9,
            add_headers=True,
        )

    def test_extract_text_plain(self) -> None:
        from tobira.milter.filter import TobiraFilter

        f = TobiraFilter()
        text = f._extract_text(b"Hello, this is plain text")
        assert "Hello, this is plain text" in text

    def test_extract_text_multipart(self) -> None:
        from tobira.milter.filter import TobiraFilter

        mime = (
            b"MIME-Version: 1.0\r\n"
            b"Content-Type: multipart/alternative; boundary=BOUNDARY\r\n"
            b"\r\n"
            b"--BOUNDARY\r\n"
            b"Content-Type: text/plain; charset=utf-8\r\n"
            b"\r\n"
            b"Plain text body\r\n"
            b"--BOUNDARY\r\n"
            b"Content-Type: text/html; charset=utf-8\r\n"
            b"\r\n"
            b"<p>HTML body</p>\r\n"
            b"--BOUNDARY--\r\n"
        )

        f = TobiraFilter()
        text = f._extract_text(mime)
        assert "Plain text body" in text
        assert "<p>HTML body</p>" not in text

    @patch("tobira.milter.filter._import_milter")
    def test_header_captures_subject(self, mock_import: MagicMock) -> None:
        mock_milter = MagicMock()
        mock_milter.CONTINUE = 0
        mock_import.return_value = mock_milter

        from tobira.milter.filter import TobiraFilter

        f = TobiraFilter()
        result = f.header("Subject", "Test Email")

        assert result == 0
        assert f._subject == "Test Email"

    @patch("tobira.milter.filter._import_milter")
    def test_header_ignores_other(self, mock_import: MagicMock) -> None:
        mock_milter = MagicMock()
        mock_milter.CONTINUE = 0
        mock_import.return_value = mock_milter

        from tobira.milter.filter import TobiraFilter

        f = TobiraFilter()
        f.header("From", "test@example.com")

        assert f._subject == ""

    @patch("tobira.milter.filter._import_milter")
    def test_body_collects_chunks(self, mock_import: MagicMock) -> None:
        mock_milter = MagicMock()
        mock_milter.CONTINUE = 0
        mock_import.return_value = mock_milter

        from tobira.milter.filter import TobiraFilter

        f = TobiraFilter()
        f.body(b"chunk1")
        f.body(b"chunk2")

        assert f._body_chunks == [b"chunk1", b"chunk2"]

    @patch("tobira.milter.filter._import_milter")
    def test_close_resets_state(self, mock_import: MagicMock) -> None:
        mock_milter = MagicMock()
        mock_milter.CONTINUE = 0
        mock_import.return_value = mock_milter

        from tobira.milter.filter import TobiraFilter

        f = TobiraFilter()
        f._subject = "test"
        f._body_chunks = [b"data"]

        f.close()

        assert f._subject == ""
        assert f._body_chunks == []


# ---------------------------------------------------------------------------
# TobiraFilter.eom tests (API interaction)
# ---------------------------------------------------------------------------


class TestTobiraFilterEom:
    @patch("tobira.milter.filter._import_urllib")
    @patch("tobira.milter.filter._import_milter")
    def test_eom_accept_ham(
        self, mock_import_milter: MagicMock, mock_import_urllib: MagicMock
    ) -> None:
        mock_milter = MagicMock()
        mock_milter.CONTINUE = 0
        mock_milter.ACCEPT = 1
        mock_milter.REJECT = 2
        mock_milter.TEMPFAIL = 3
        mock_import_milter.return_value = mock_milter

        mock_json = MagicMock()
        mock_json.dumps.return_value = '{"text": "hello"}'
        mock_json.loads.return_value = {
            "label": "ham",
            "score": 0.1,
            "labels": {"spam": 0.1, "ham": 0.9},
        }

        mock_urllib_request = MagicMock()
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = b'{"label": "ham", "score": 0.1}'
        mock_urllib_request.urlopen.return_value = mock_response

        mock_urllib_error = MagicMock()
        mock_import_urllib.return_value = (
            mock_urllib_request, mock_urllib_error, mock_json,
        )

        from tobira.milter.filter import TobiraFilter

        TobiraFilter.configure(
            api_url="http://127.0.0.1:8000",
            timeout=10,
            fail_action="accept",
            reject_threshold=0.9,
            add_headers=True,
        )

        f = TobiraFilter()
        f.addheader = MagicMock()  # type: ignore[attr-defined]
        f.setreply = MagicMock()  # type: ignore[attr-defined]
        f._body_chunks = [b"Hello, this is a test email"]

        result = f.eom()

        assert result == 1  # ACCEPT
        f.addheader.assert_any_call("X-Tobira-Score", "0.1000")
        f.addheader.assert_any_call("X-Tobira-Label", "ham")

    @patch("tobira.milter.filter._import_urllib")
    @patch("tobira.milter.filter._import_milter")
    def test_eom_reject_spam(
        self, mock_import_milter: MagicMock, mock_import_urllib: MagicMock
    ) -> None:
        mock_milter = MagicMock()
        mock_milter.CONTINUE = 0
        mock_milter.ACCEPT = 1
        mock_milter.REJECT = 2
        mock_milter.TEMPFAIL = 3
        mock_import_milter.return_value = mock_milter

        mock_json = MagicMock()
        mock_json.dumps.return_value = '{"text": "spam"}'
        mock_json.loads.return_value = {
            "label": "spam",
            "score": 0.95,
            "labels": {"spam": 0.95, "ham": 0.05},
        }

        mock_urllib_request = MagicMock()
        mock_response = MagicMock()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_response.read.return_value = b'{"label": "spam", "score": 0.95}'
        mock_urllib_request.urlopen.return_value = mock_response

        mock_urllib_error = MagicMock()
        mock_import_urllib.return_value = (
            mock_urllib_request, mock_urllib_error, mock_json,
        )

        from tobira.milter.filter import TobiraFilter

        TobiraFilter.configure(
            api_url="http://127.0.0.1:8000",
            timeout=10,
            fail_action="accept",
            reject_threshold=0.9,
            add_headers=True,
        )

        f = TobiraFilter()
        f.addheader = MagicMock()  # type: ignore[attr-defined]
        f.setreply = MagicMock()  # type: ignore[attr-defined]
        f._body_chunks = [b"Buy cheap pills now!!!"]

        result = f.eom()

        assert result == 2  # REJECT
        f.setreply.assert_called_once_with("550", "5.7.1", "Message rejected as spam")

    @patch("tobira.milter.filter._import_urllib")
    @patch("tobira.milter.filter._import_milter")
    def test_eom_api_failure_accept(
        self, mock_import_milter: MagicMock, mock_import_urllib: MagicMock
    ) -> None:
        mock_milter = MagicMock()
        mock_milter.CONTINUE = 0
        mock_milter.ACCEPT = 1
        mock_milter.REJECT = 2
        mock_milter.TEMPFAIL = 3
        mock_import_milter.return_value = mock_milter

        mock_json = MagicMock()
        mock_json.dumps.return_value = '{"text": "test"}'

        mock_urllib_request = MagicMock()
        mock_urllib_request.urlopen.side_effect = Exception("connection refused")

        mock_urllib_error = MagicMock()
        mock_import_urllib.return_value = (
            mock_urllib_request, mock_urllib_error, mock_json,
        )

        from tobira.milter.filter import TobiraFilter

        TobiraFilter.configure(
            api_url="http://127.0.0.1:8000",
            timeout=10,
            fail_action="accept",
            reject_threshold=0.9,
            add_headers=True,
        )

        f = TobiraFilter()
        f.addheader = MagicMock()  # type: ignore[attr-defined]
        f.setreply = MagicMock()  # type: ignore[attr-defined]
        f._body_chunks = [b"Test email"]

        result = f.eom()

        assert result == 1  # ACCEPT (fail-open)

    @patch("tobira.milter.filter._import_urllib")
    @patch("tobira.milter.filter._import_milter")
    def test_eom_api_failure_tempfail(
        self, mock_import_milter: MagicMock, mock_import_urllib: MagicMock
    ) -> None:
        mock_milter = MagicMock()
        mock_milter.CONTINUE = 0
        mock_milter.ACCEPT = 1
        mock_milter.REJECT = 2
        mock_milter.TEMPFAIL = 3
        mock_import_milter.return_value = mock_milter

        mock_json = MagicMock()
        mock_json.dumps.return_value = '{"text": "test"}'

        mock_urllib_request = MagicMock()
        mock_urllib_request.urlopen.side_effect = Exception("timeout")

        mock_urllib_error = MagicMock()
        mock_import_urllib.return_value = (
            mock_urllib_request, mock_urllib_error, mock_json,
        )

        from tobira.milter.filter import TobiraFilter

        TobiraFilter.configure(
            api_url="http://127.0.0.1:8000",
            timeout=10,
            fail_action="tempfail",
            reject_threshold=0.9,
            add_headers=True,
        )

        f = TobiraFilter()
        f.addheader = MagicMock()  # type: ignore[attr-defined]
        f.setreply = MagicMock()  # type: ignore[attr-defined]
        f._body_chunks = [b"Test email"]

        result = f.eom()

        assert result == 3  # TEMPFAIL

    @patch("tobira.milter.filter._import_urllib")
    @patch("tobira.milter.filter._import_milter")
    def test_eom_empty_body_accept(
        self, mock_import_milter: MagicMock, mock_import_urllib: MagicMock
    ) -> None:
        mock_milter = MagicMock()
        mock_milter.CONTINUE = 0
        mock_milter.ACCEPT = 1
        mock_import_milter.return_value = mock_milter

        mock_json = MagicMock()
        mock_urllib_request = MagicMock()
        mock_urllib_error = MagicMock()
        mock_import_urllib.return_value = (
            mock_urllib_request, mock_urllib_error, mock_json,
        )

        from tobira.milter.filter import TobiraFilter

        f = TobiraFilter()
        f._body_chunks = []

        result = f.eom()

        assert result == 1  # ACCEPT


# ---------------------------------------------------------------------------
# run_milter tests
# ---------------------------------------------------------------------------


class TestRunMilter:
    @patch("tobira.milter.filter._import_milter")
    def test_invalid_socket_spec(self, mock_import: MagicMock) -> None:
        mock_milter = MagicMock()
        mock_import.return_value = mock_milter

        from tobira.milter.filter import run_milter

        with pytest.raises(ValueError, match="invalid socket spec"):
            run_milter(socket_spec="tcp:9900")


# ---------------------------------------------------------------------------
# CLI milter_cmd tests
# ---------------------------------------------------------------------------


class TestMilterCli:
    def test_parser_has_milter_subcommand(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["milter", "--config", "/tmp/milter.conf"])
        assert args.command == "milter"

    def test_parser_milter_missing_config_exits(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["milter"])

    def test_milter_run_missing_config_file(self) -> None:
        from tobira.cli import main

        result = main(["milter", "--config", "/nonexistent/milter.conf"])
        assert result == 1


# ---------------------------------------------------------------------------
# MTA detector tests for Postfix
# ---------------------------------------------------------------------------


class TestPostfixDetection:
    @patch("tobira.cli.mta_detectors._check_config_dir")
    @patch("tobira.cli.mta_detectors._check_systemd")
    @patch("tobira.cli.mta_detectors._check_process")
    def test_detect_postfix_by_process(
        self,
        mock_proc: MagicMock,
        mock_systemd: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        from tobira.cli.mta_detectors import detect_mtas

        def proc_side_effect(mta: str) -> bool:
            return mta == "postfix"

        mock_proc.side_effect = proc_side_effect
        mock_systemd.return_value = False
        mock_config.return_value = False

        result = detect_mtas()
        names = [d.name for d in result]
        assert "postfix" in names

    @patch("tobira.cli.mta_detectors._check_config_dir")
    @patch("tobira.cli.mta_detectors._check_systemd")
    @patch("tobira.cli.mta_detectors._check_process")
    def test_detect_postfix_by_systemd(
        self,
        mock_proc: MagicMock,
        mock_systemd: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        from tobira.cli.mta_detectors import detect_mtas

        mock_proc.return_value = False

        def systemd_side_effect(mta: str) -> bool:
            return mta == "postfix"

        mock_systemd.side_effect = systemd_side_effect
        mock_config.return_value = False

        result = detect_mtas()
        names = [d.name for d in result]
        assert "postfix" in names

    @patch("tobira.cli.mta_detectors._check_config_dir")
    @patch("tobira.cli.mta_detectors._check_systemd")
    @patch("tobira.cli.mta_detectors._check_process")
    def test_detect_postfix_by_config(
        self,
        mock_proc: MagicMock,
        mock_systemd: MagicMock,
        mock_config: MagicMock,
    ) -> None:
        from tobira.cli.mta_detectors import detect_mtas

        mock_proc.return_value = False
        mock_systemd.return_value = False

        def config_side_effect(mta: str) -> bool:
            return mta == "postfix"

        mock_config.side_effect = config_side_effect

        result = detect_mtas()
        names = [d.name for d in result]
        assert "postfix" in names


# ---------------------------------------------------------------------------
# Config generator tests for Postfix
# ---------------------------------------------------------------------------


class TestPostfixConfigGeneration:
    def test_generate_postfix_configs(self, tmp_path: Path) -> None:
        from tobira.cli.generators import generate_configs

        files = generate_configs("postfix", tmp_path)

        assert len(files) == 2
        names = [f.name for f in files]
        assert "tobira-milter.conf" in names
        assert "tobira-milter.service" in names

    def test_generate_postfix_custom_url(self, tmp_path: Path) -> None:
        from tobira.cli.generators import generate_configs

        files = generate_configs("postfix", tmp_path, api_url="http://10.0.0.1:9000")

        conf_file = next(f for f in files if f.name == "tobira-milter.conf")
        content = conf_file.read_text(encoding="utf-8")
        assert "http://10.0.0.1:9000" in content
        assert "http://127.0.0.1:8000" not in content

    def test_get_install_instructions_postfix(self, tmp_path: Path) -> None:
        from tobira.cli.generators import get_install_instructions

        instructions = get_install_instructions("postfix", tmp_path)

        joined = "\n".join(instructions)
        assert "milter.conf" in joined
        assert "tobira-milter.service" in joined
        assert "postfix reload" in joined
        assert "milter_default_action" in joined
