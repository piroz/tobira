"""Tests for tobira.cli."""

from __future__ import annotations

import os
import textwrap
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

    def test_doctor_help_exits_with_0(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["doctor", "--help"])
        assert exc_info.value.code == 0


# --- doctor subcommand tests ---


class TestDoctorParser:
    def test_parser_has_doctor_subcommand(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["doctor", "--config", "/tmp/c.toml"])
        assert args.command == "doctor"

    def test_parser_doctor_defaults(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["doctor", "--config", "/tmp/c.toml"])
        assert args.config == "/tmp/c.toml"
        assert args.api_url is None

    def test_parser_doctor_with_api_url(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "doctor", "--config", "/tmp/c.toml",
            "--api-url", "http://localhost:8000",
        ])
        assert args.api_url == "http://localhost:8000"

    def test_parser_doctor_missing_config_exits(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["doctor"])


class TestDoctorDispatch:
    @patch("tobira.cli.doctor._run")
    def test_doctor_dispatches_to_run(self, mock_run: MagicMock) -> None:
        from tobira.cli import main

        mock_run.return_value = 0
        result = main(["doctor", "--config", "/tmp/c.toml"])

        assert result == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.config == "/tmp/c.toml"


class TestDoctorChecks:
    def test_check_config_file_not_found(self) -> None:
        from tobira.cli.doctor import run_checks

        results = run_checks(config_path="/nonexistent/config.toml")
        assert results[0] == (False, "config file not found: /nonexistent/config.toml")

    def test_check_config_invalid_toml(self, tmp_path: "os.PathLike[str]") -> None:
        from tobira.cli.doctor import run_checks

        bad_toml = tmp_path / "bad.toml"  # type: ignore[operator]
        bad_toml.write_text("[[invalid toml", encoding="utf-8")

        results = run_checks(config_path=str(bad_toml))
        ok, msg = results[0]
        assert ok is False
        assert "invalid TOML syntax" in msg

    def test_check_config_valid_but_no_backend(
        self, tmp_path: "os.PathLike[str]"
    ) -> None:
        from tobira.cli.doctor import run_checks

        toml_file = tmp_path / "ok.toml"  # type: ignore[operator]
        toml_file.write_text("[server]\nhost = '0.0.0.0'\n", encoding="utf-8")

        results = run_checks(config_path=str(toml_file))
        # Config check passes
        assert results[0][0] is True
        # Backend check fails (no [backend] section)
        assert results[1] == (False, "missing [backend] section in config")

    @patch("tobira.backends.factory.create_backend")
    def test_check_backend_success(
        self, mock_create: MagicMock, tmp_path: "os.PathLike[str]"
    ) -> None:
        from tobira.cli.doctor import run_checks

        toml_file = tmp_path / "ok.toml"  # type: ignore[operator]
        toml_content = textwrap.dedent("""\
            [backend]
            type = "fasttext"
            model_path = "/tmp/model.bin"
        """)
        toml_file.write_text(toml_content, encoding="utf-8")

        mock_create.return_value = MagicMock()
        results = run_checks(config_path=str(toml_file))

        assert results[0][0] is True  # config OK
        assert results[1] == (True, "backend loaded successfully")

    @patch("tobira.backends.factory.create_backend", side_effect=ValueError("bad type"))
    def test_check_backend_failure(
        self, mock_create: MagicMock, tmp_path: "os.PathLike[str]"
    ) -> None:
        from tobira.cli.doctor import run_checks

        toml_file = tmp_path / "ok.toml"  # type: ignore[operator]
        toml_content = textwrap.dedent("""\
            [backend]
            type = "unknown"
        """)
        toml_file.write_text(toml_content, encoding="utf-8")

        results = run_checks(config_path=str(toml_file))
        ok, msg = results[1]
        assert ok is False
        assert "backend initialisation failed" in msg

    @patch("tobira.cli.doctor.urlopen")
    def test_check_api_success(
        self, mock_urlopen: MagicMock, tmp_path: "os.PathLike[str]"
    ) -> None:
        from tobira.cli.doctor import run_checks

        toml_file = tmp_path / "ok.toml"  # type: ignore[operator]
        toml_file.write_text("[server]\n", encoding="utf-8")

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        results = run_checks(
            config_path=str(toml_file), api_url="http://localhost:8000"
        )
        # Find the API check result (after config + backend-missing)
        api_results = [(ok, msg) for ok, msg in results if "API server" in msg]
        assert len(api_results) == 1
        assert api_results[0][0] is True

    def test_check_api_unreachable(self, tmp_path: "os.PathLike[str]") -> None:
        from tobira.cli.doctor import run_checks

        toml_file = tmp_path / "ok.toml"  # type: ignore[operator]
        toml_file.write_text("[server]\n", encoding="utf-8")

        results = run_checks(
            config_path=str(toml_file),
            api_url="http://localhost:19999",
        )
        api_results = [(ok, msg) for ok, msg in results if "API server" in msg]
        assert len(api_results) == 1
        assert api_results[0][0] is False

    def test_mta_plugins_not_found(self) -> None:
        from tobira.cli.doctor import _check_mta_plugins

        results = _check_mta_plugins()
        # In CI/test environments MTA configs won't exist
        mta_names = [msg.split(" ")[0] for _, msg in results]
        assert "rspamd" in mta_names
        assert "spamassassin" in mta_names
        assert "haraka" in mta_names
        assert len(results) == 3


class TestDoctorRun:
    @patch("tobira.cli.doctor.run_checks")
    def test_run_returns_0_when_all_pass(self, mock_checks: MagicMock) -> None:
        from tobira.cli import main

        mock_checks.return_value = [(True, "all good")]
        result = main(["doctor", "--config", "/tmp/c.toml"])
        assert result == 0

    @patch("tobira.cli.doctor.run_checks")
    def test_run_returns_1_when_check_fails(self, mock_checks: MagicMock) -> None:
        from tobira.cli import main

        mock_checks.return_value = [(True, "config OK"), (False, "backend failed")]
        result = main(["doctor", "--config", "/tmp/c.toml"])
        assert result == 1


# --- init subcommand tests ---


class TestInitParser:
    def test_parser_has_init_subcommand(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["init"])
        assert args.command == "init"

    def test_parser_init_defaults(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["init"])
        assert args.output_dir is None
        assert args.api_url is None

    def test_parser_init_with_options(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "init",
            "--output-dir", "/tmp/out",
            "--api-url", "http://localhost:9000",
        ])
        assert args.output_dir == "/tmp/out"
        assert args.api_url == "http://localhost:9000"

    def test_init_help_exits_with_0(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["init", "--help"])
        assert exc_info.value.code == 0


class TestInitDispatch:
    @patch("tobira.cli.setup._run")
    def test_init_dispatches_to_run(self, mock_run: MagicMock) -> None:
        from tobira.cli import main

        mock_run.return_value = 0
        result = main(["init"])

        assert result == 0
        mock_run.assert_called_once()


class TestMTADetectors:
    @patch("tobira.cli.mta_detectors._check_config_dir", return_value=False)
    @patch("tobira.cli.mta_detectors._check_systemd", return_value=False)
    @patch("tobira.cli.mta_detectors._check_process", return_value=False)
    def test_detect_no_mtas(
        self,
        mock_proc: MagicMock,
        mock_sysd: MagicMock,
        mock_conf: MagicMock,
    ) -> None:
        from tobira.cli.mta_detectors import detect_mtas

        result = detect_mtas()
        assert result == []

    @patch("tobira.cli.mta_detectors._check_config_dir", return_value=False)
    @patch("tobira.cli.mta_detectors._check_systemd", return_value=False)
    @patch("tobira.cli.mta_detectors._check_process")
    def test_detect_single_mta_by_process(
        self,
        mock_proc: MagicMock,
        mock_sysd: MagicMock,
        mock_conf: MagicMock,
    ) -> None:
        from tobira.cli.mta_detectors import detect_mtas

        mock_proc.side_effect = lambda mta: mta == "rspamd"
        result = detect_mtas()
        assert len(result) == 1
        assert result[0].name == "rspamd"
        assert result[0].source == "process"

    @patch("tobira.cli.mta_detectors._check_config_dir")
    @patch("tobira.cli.mta_detectors._check_systemd")
    @patch("tobira.cli.mta_detectors._check_process", return_value=False)
    def test_detect_mta_by_systemd(
        self,
        mock_proc: MagicMock,
        mock_sysd: MagicMock,
        mock_conf: MagicMock,
    ) -> None:
        from tobira.cli.mta_detectors import detect_mtas

        mock_sysd.side_effect = lambda mta: mta == "haraka"
        mock_conf.return_value = False
        result = detect_mtas()
        assert len(result) == 1
        assert result[0].name == "haraka"
        assert result[0].source == "systemd"

    @patch("tobira.cli.mta_detectors._check_config_dir")
    @patch("tobira.cli.mta_detectors._check_systemd", return_value=False)
    @patch("tobira.cli.mta_detectors._check_process", return_value=False)
    def test_detect_mta_by_config_dir(
        self,
        mock_proc: MagicMock,
        mock_sysd: MagicMock,
        mock_conf: MagicMock,
    ) -> None:
        from tobira.cli.mta_detectors import detect_mtas

        mock_conf.side_effect = lambda mta: mta == "spamassassin"
        result = detect_mtas()
        assert len(result) == 1
        assert result[0].name == "spamassassin"
        assert result[0].source == "config_path"

    @patch("tobira.cli.mta_detectors._check_config_dir", return_value=True)
    @patch("tobira.cli.mta_detectors._check_systemd", return_value=False)
    @patch("tobira.cli.mta_detectors._check_process", return_value=False)
    def test_detect_multiple_mtas(
        self,
        mock_proc: MagicMock,
        mock_sysd: MagicMock,
        mock_conf: MagicMock,
    ) -> None:
        from tobira.cli.mta_detectors import detect_mtas

        result = detect_mtas()
        assert len(result) == 3
        names = {d.name for d in result}
        assert names == {"rspamd", "spamassassin", "haraka"}


class TestGenerators:
    def test_generate_rspamd_configs(self, tmp_path: "os.PathLike[str]") -> None:
        from pathlib import Path

        from tobira.cli.generators import generate_configs

        output = Path(str(tmp_path))
        generated = generate_configs(mta="rspamd", output_dir=output)

        assert len(generated) == 2
        filenames = {p.name for p in generated}
        assert "tobira.conf" in filenames
        assert "tobira.lua" in filenames

        # Check content contains default URL
        conf_content = (output / "tobira.conf").read_text(encoding="utf-8")
        assert "http://127.0.0.1:8000" in conf_content

    def test_generate_spamassassin_configs(
        self, tmp_path: "os.PathLike[str]"
    ) -> None:
        from pathlib import Path

        from tobira.cli.generators import generate_configs

        output = Path(str(tmp_path))
        generated = generate_configs(mta="spamassassin", output_dir=output)

        assert len(generated) == 2
        filenames = {p.name for p in generated}
        assert "tobira.cf" in filenames
        assert "Tobira.pm" in filenames

    def test_generate_haraka_configs(self, tmp_path: "os.PathLike[str]") -> None:
        from pathlib import Path

        from tobira.cli.generators import generate_configs

        output = Path(str(tmp_path))
        generated = generate_configs(mta="haraka", output_dir=output)

        assert len(generated) == 2
        filenames = {p.name for p in generated}
        assert "tobira.ini" in filenames
        assert "tobira.js" in filenames

    def test_generate_with_custom_api_url(
        self, tmp_path: "os.PathLike[str]"
    ) -> None:
        from pathlib import Path

        from tobira.cli.generators import generate_configs

        output = Path(str(tmp_path))
        generate_configs(
            mta="rspamd",
            output_dir=output,
            api_url="http://10.0.0.1:9000",
        )

        conf_content = (output / "tobira.conf").read_text(encoding="utf-8")
        assert "http://10.0.0.1:9000" in conf_content
        assert "http://127.0.0.1:8000" not in conf_content

    def test_generate_unsupported_mta_raises(
        self, tmp_path: "os.PathLike[str]"
    ) -> None:
        from pathlib import Path

        from tobira.cli.generators import generate_configs

        output = Path(str(tmp_path))
        with pytest.raises(ValueError, match="unsupported MTA"):
            generate_configs(mta="postfix", output_dir=output)

    def test_get_install_instructions(self, tmp_path: "os.PathLike[str]") -> None:
        from pathlib import Path

        from tobira.cli.generators import get_install_instructions

        output = Path(str(tmp_path))
        instructions = get_install_instructions(mta="rspamd", output_dir=output)

        assert len(instructions) > 0
        assert any("sudo cp" in line for line in instructions)
        assert any("rspamd" in line for line in instructions)

    def test_get_install_instructions_unsupported(self) -> None:
        from pathlib import Path

        from tobira.cli.generators import get_install_instructions

        instructions = get_install_instructions(mta="postfix", output_dir=Path("/tmp"))
        assert instructions == []


class TestInitRunIntegration:
    @patch("tobira.cli.setup.detect_mtas")
    def test_run_init_single_mta_detected(
        self,
        mock_detect: MagicMock,
        tmp_path: "os.PathLike[str]",
    ) -> None:
        from pathlib import Path

        from tobira.cli.mta_detectors import DetectedMTA
        from tobira.cli.setup import run_init

        mock_detect.return_value = [DetectedMTA(name="rspamd", source="process")]

        output = Path(str(tmp_path))
        result = run_init(output_dir=output, api_url="http://127.0.0.1:8000")

        assert result == 0
        assert (output / "tobira.conf").exists()
        assert (output / "tobira.lua").exists()

    @patch("tobira.cli.setup._prompt_select_mta", return_value="haraka")
    @patch("tobira.cli.setup.detect_mtas")
    def test_run_init_multiple_mtas_user_selects(
        self,
        mock_detect: MagicMock,
        mock_prompt: MagicMock,
        tmp_path: "os.PathLike[str]",
    ) -> None:
        from pathlib import Path

        from tobira.cli.mta_detectors import DetectedMTA
        from tobira.cli.setup import run_init

        mock_detect.return_value = [
            DetectedMTA(name="rspamd", source="process"),
            DetectedMTA(name="haraka", source="systemd"),
        ]

        output = Path(str(tmp_path))
        result = run_init(output_dir=output, api_url="http://127.0.0.1:8000")

        assert result == 0
        assert (output / "tobira.ini").exists()
        assert (output / "tobira.js").exists()
        mock_prompt.assert_called_once()

    @patch("tobira.cli.setup._prompt_select_mta", return_value=None)
    @patch("tobira.cli.setup.detect_mtas")
    def test_run_init_cancelled(
        self,
        mock_detect: MagicMock,
        mock_prompt: MagicMock,
    ) -> None:
        from tobira.cli.mta_detectors import DetectedMTA
        from tobira.cli.setup import run_init

        mock_detect.return_value = [
            DetectedMTA(name="rspamd", source="process"),
            DetectedMTA(name="haraka", source="systemd"),
        ]

        result = run_init(api_url="http://127.0.0.1:8000")
        assert result == 1

    @patch("tobira.cli.setup._prompt_select_mta", return_value="rspamd")
    @patch("tobira.cli.setup.detect_mtas")
    def test_run_init_no_mtas_manual_select(
        self,
        mock_detect: MagicMock,
        mock_prompt: MagicMock,
        tmp_path: "os.PathLike[str]",
    ) -> None:
        from pathlib import Path

        from tobira.cli.setup import run_init

        mock_detect.return_value = []

        output = Path(str(tmp_path))
        result = run_init(output_dir=output, api_url="http://127.0.0.1:8000")

        assert result == 0
        assert (output / "tobira.conf").exists()
