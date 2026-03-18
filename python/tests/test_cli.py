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
        assert len(result) == 4
        names = {d.name for d in result}
        assert names == {"rspamd", "spamassassin", "haraka", "postfix"}


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
            generate_configs(mta="unknown_mta", output_dir=output)

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

        instructions = get_install_instructions(
            mta="unknown_mta", output_dir=Path("/tmp"),
        )
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


# --- evaluate subcommand tests ---


class TestEvaluateParser:
    def test_parser_has_evaluate_subcommand(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["evaluate", "/tmp/data.csv"])
        assert args.command == "evaluate"

    def test_parser_evaluate_defaults(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["evaluate", "/tmp/data.csv"])
        assert args.dataset == "/tmp/data.csv"
        assert args.config is None
        assert args.output_format == "text"
        assert args.plot is None
        assert args.tune_threshold is False

    def test_parser_evaluate_all_options(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "evaluate", "/tmp/data.jsonl",
            "--config", "/tmp/c.toml",
            "--format", "json",
            "--plot", "/tmp/pr.png",
            "--tune-threshold",
        ])
        assert args.dataset == "/tmp/data.jsonl"
        assert args.config == "/tmp/c.toml"
        assert args.output_format == "json"
        assert args.plot == "/tmp/pr.png"
        assert args.tune_threshold is True

    def test_parser_evaluate_missing_dataset_exits(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["evaluate"])

    def test_evaluate_help_exits_with_0(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["evaluate", "--help"])
        assert exc_info.value.code == 0


class TestEvaluateDispatch:
    @patch("tobira.cli.evaluate._run")
    def test_evaluate_dispatches_to_run(self, mock_run: MagicMock) -> None:
        from tobira.cli import main

        mock_run.return_value = 0
        result = main(["evaluate", "/tmp/data.csv"])

        assert result == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.dataset == "/tmp/data.csv"


class TestEvaluateRun:
    def test_missing_dataset_file(self) -> None:
        from tobira.cli import main

        result = main(["evaluate", "/nonexistent/data.csv"])
        assert result == 1

    def test_empty_dataset(self, tmp_path: "os.PathLike[str]") -> None:
        from pathlib import Path

        from tobira.cli import main

        csv_file = Path(str(tmp_path)) / "empty.csv"
        csv_file.write_text("text,label,score\n", encoding="utf-8")

        result = main(["evaluate", str(csv_file)])
        assert result == 1

    def test_missing_columns(self, tmp_path: "os.PathLike[str]") -> None:
        from pathlib import Path

        from tobira.cli import main

        csv_file = Path(str(tmp_path)) / "bad.csv"
        csv_file.write_text("foo,bar\na,b\n", encoding="utf-8")

        result = main(["evaluate", str(csv_file)])
        assert result == 1

    def test_unsupported_format(self, tmp_path: "os.PathLike[str]") -> None:
        from pathlib import Path

        from tobira.cli import main

        txt_file = Path(str(tmp_path)) / "data.txt"
        txt_file.write_text("text,label\nhello,1\n", encoding="utf-8")

        result = main(["evaluate", str(txt_file)])
        assert result == 1

    def test_no_score_no_config_returns_error(
        self, tmp_path: "os.PathLike[str]"
    ) -> None:
        from pathlib import Path

        from tobira.cli import main

        csv_file = Path(str(tmp_path)) / "data.csv"
        csv_file.write_text("text,label\nhello,1\nworld,0\n", encoding="utf-8")

        result = main(["evaluate", str(csv_file)])
        assert result == 1

    def test_csv_with_scores_text_format(
        self, tmp_path: "os.PathLike[str]", capsys: pytest.CaptureFixture[str]
    ) -> None:
        from pathlib import Path

        from tobira.cli import main

        csv_file = Path(str(tmp_path)) / "data.csv"
        lines = [
            "text,label,score",
            "spam msg,1,0.9",
            "ham msg,0,0.1",
            "another spam,1,0.8",
            "another ham,0,0.2",
        ]
        csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = main(["evaluate", str(csv_file)])
        assert result == 0

        captured = capsys.readouterr()
        assert "Evaluation Report" in captured.out
        assert "F1:" in captured.out

    def test_csv_with_scores_json_format(
        self, tmp_path: "os.PathLike[str]", capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json
        from pathlib import Path

        from tobira.cli import main

        csv_file = Path(str(tmp_path)) / "data.csv"
        lines = [
            "text,label,score",
            "spam msg,1,0.9",
            "ham msg,0,0.1",
        ]
        csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = main(["evaluate", str(csv_file), "--format", "json"])
        assert result == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "accuracy" in data
        assert "f1" in data

    def test_jsonl_with_scores(
        self, tmp_path: "os.PathLike[str]", capsys: pytest.CaptureFixture[str]
    ) -> None:
        import json
        from pathlib import Path

        from tobira.cli import main

        jsonl_file = Path(str(tmp_path)) / "data.jsonl"
        rows = [
            {"text": "spam msg", "label": "1", "score": "0.9"},
            {"text": "ham msg", "label": "0", "score": "0.1"},
        ]
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

        result = main(["evaluate", str(jsonl_file)])
        assert result == 0

        captured = capsys.readouterr()
        assert "Evaluation Report" in captured.out

    def test_tune_threshold(
        self, tmp_path: "os.PathLike[str]", capsys: pytest.CaptureFixture[str]
    ) -> None:
        from pathlib import Path

        from tobira.cli import main

        csv_file = Path(str(tmp_path)) / "data.csv"
        lines = [
            "text,label,score",
            "spam,1,0.9",
            "ham,0,0.1",
            "spam2,1,0.7",
            "ham2,0,0.3",
        ]
        csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = main(["evaluate", str(csv_file), "--tune-threshold"])
        assert result == 0

        captured = capsys.readouterr()
        assert "Optimal threshold" in captured.out

    @patch("tobira.backends.factory.create_backend")
    def test_backend_inference_mode(
        self,
        mock_create: MagicMock,
        tmp_path: "os.PathLike[str]",
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from pathlib import Path

        from tobira.backends.protocol import PredictionResult
        from tobira.cli import main

        # Create config
        config_file = Path(str(tmp_path)) / "config.toml"
        config_file.write_text(
            '[backend]\ntype = "fasttext"\nmodel_path = "/tmp/m.bin"\n',
            encoding="utf-8",
        )

        # Create dataset without score column
        csv_file = Path(str(tmp_path)) / "data.csv"
        lines = ["text,label", "spam msg,1", "ham msg,0"]
        csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Mock backend
        mock_backend = MagicMock()
        mock_backend.predict.side_effect = [
            PredictionResult(label="spam", score=0.9, labels={"spam": 0.9, "ham": 0.1}),
            PredictionResult(label="ham", score=0.1, labels={"spam": 0.1, "ham": 0.9}),
        ]
        mock_create.return_value = mock_backend

        result = main(["evaluate", str(csv_file), "--config", str(config_file)])
        assert result == 0

        captured = capsys.readouterr()
        assert "Evaluation Report" in captured.out
        assert mock_backend.predict.call_count == 2

    def test_config_file_not_found(
        self, tmp_path: "os.PathLike[str]"
    ) -> None:
        from pathlib import Path

        from tobira.cli import main

        csv_file = Path(str(tmp_path)) / "data.csv"
        lines = ["text,label", "spam,1", "ham,0"]
        csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = main([
            "evaluate", str(csv_file), "--config", "/nonexistent/config.toml"
        ])
        assert result == 1

    def test_config_missing_backend_section(
        self, tmp_path: "os.PathLike[str]"
    ) -> None:
        from pathlib import Path

        from tobira.cli import main

        config_file = Path(str(tmp_path)) / "config.toml"
        config_file.write_text("[server]\nhost = '0.0.0.0'\n", encoding="utf-8")

        csv_file = Path(str(tmp_path)) / "data.csv"
        lines = ["text,label", "spam,1", "ham,0"]
        csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        result = main(["evaluate", str(csv_file), "--config", str(config_file)])
        assert result == 1


# --- demo subcommand tests ---


class TestDemoParser:
    def test_parser_has_demo_subcommand(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["demo"])
        assert args.command == "demo"

    def test_parser_demo_defaults(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["demo"])
        assert args.down is False
        assert args.docker_dir is None
        assert args.timeout == 60

    def test_parser_demo_down_flag(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["demo", "--down"])
        assert args.down is True

    def test_parser_demo_custom_options(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "demo", "--docker-dir", "/tmp/docker", "--timeout", "30",
        ])
        assert args.docker_dir == "/tmp/docker"
        assert args.timeout == 30

    def test_demo_help_exits_with_0(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["demo", "--help"])
        assert exc_info.value.code == 0


class TestDemoDispatch:
    @patch("tobira.cli.demo._run")
    def test_demo_dispatches_to_run(self, mock_run: MagicMock) -> None:
        from tobira.cli import main

        mock_run.return_value = 0
        result = main(["demo"])

        assert result == 0
        mock_run.assert_called_once()


class TestDemoPrerequisites:
    @patch("tobira.cli.demo.shutil.which", return_value=None)
    def test_docker_not_found(self, mock_which: MagicMock) -> None:
        from tobira.cli.demo import _check_prerequisites

        ok, msg = _check_prerequisites()
        assert ok is False
        assert "docker command not found" in msg

    @patch("tobira.cli.demo.subprocess.run")
    @patch("tobira.cli.demo.shutil.which", return_value="/usr/bin/docker")
    def test_docker_compose_available(
        self, mock_which: MagicMock, mock_run: MagicMock
    ) -> None:
        from tobira.cli.demo import _check_prerequisites

        mock_run.return_value = MagicMock(returncode=0)
        ok, msg = _check_prerequisites()
        assert ok is True
        assert "available" in msg

    @patch("tobira.cli.demo.subprocess.run")
    @patch("tobira.cli.demo.shutil.which", return_value="/usr/bin/docker")
    def test_docker_compose_not_available(
        self, mock_which: MagicMock, mock_run: MagicMock
    ) -> None:
        from tobira.cli.demo import _check_prerequisites

        mock_run.return_value = MagicMock(returncode=1)
        ok, msg = _check_prerequisites()
        assert ok is False
        assert "docker compose not available" in msg


class TestDemoFindDir:
    def test_find_docker_dir_from_source_tree(self) -> None:
        from tobira.cli.demo import _find_docker_compose_dir

        result = _find_docker_compose_dir()
        # In the source tree, docker/docker-compose.yml should exist
        if result is not None:
            assert (result / "docker-compose.yml").exists()


class TestDemoRun:
    def test_run_missing_docker_dir(self) -> None:
        from tobira.cli import main

        result = main(["demo", "--docker-dir", "/nonexistent/docker"])
        assert result == 1

    @patch("tobira.cli.demo._check_prerequisites", return_value=(False, "no docker"))
    def test_run_no_docker(self, mock_prereq: MagicMock) -> None:
        from tobira.cli import main

        result = main(["demo", "--docker-dir", "/tmp"])
        # Will fail at docker-compose.yml check before prerequisites
        assert result == 1

    @patch("tobira.cli.demo._compose_down", return_value=0)
    @patch(
        "tobira.cli.demo._check_prerequisites",
        return_value=(True, "ok"),
    )
    @patch(
        "tobira.cli.demo._find_docker_compose_dir",
    )
    def test_run_down(
        self,
        mock_find: MagicMock,
        mock_prereq: MagicMock,
        mock_down: MagicMock,
        tmp_path: "os.PathLike[str]",
    ) -> None:
        from pathlib import Path

        from tobira.cli import main

        docker_dir = Path(str(tmp_path))
        (docker_dir / "docker-compose.yml").write_text("", encoding="utf-8")

        result = main(["demo", "--down", "--docker-dir", str(docker_dir)])
        assert result == 0
        mock_down.assert_called_once_with(docker_dir)

    @patch("tobira.cli.demo._wait_for_health", return_value=True)
    @patch("tobira.cli.demo._compose_up", return_value=0)
    @patch(
        "tobira.cli.demo._check_prerequisites",
        return_value=(True, "ok"),
    )
    def test_run_up_success(
        self,
        mock_prereq: MagicMock,
        mock_up: MagicMock,
        mock_health: MagicMock,
        tmp_path: "os.PathLike[str]",
    ) -> None:
        from pathlib import Path

        from tobira.cli import main

        docker_dir = Path(str(tmp_path))
        (docker_dir / "docker-compose.yml").write_text("", encoding="utf-8")

        result = main(["demo", "--docker-dir", str(docker_dir)])
        assert result == 0
        mock_up.assert_called_once()

    @patch("tobira.cli.demo._compose_up", return_value=1)
    @patch(
        "tobira.cli.demo._check_prerequisites",
        return_value=(True, "ok"),
    )
    def test_run_up_failure(
        self,
        mock_prereq: MagicMock,
        mock_up: MagicMock,
        tmp_path: "os.PathLike[str]",
    ) -> None:
        from pathlib import Path

        from tobira.cli import main

        docker_dir = Path(str(tmp_path))
        (docker_dir / "docker-compose.yml").write_text("", encoding="utf-8")

        result = main(["demo", "--docker-dir", str(docker_dir)])
        assert result == 1

    @patch("tobira.cli.demo._wait_for_health", return_value=False)
    @patch("tobira.cli.demo._compose_up", return_value=0)
    @patch(
        "tobira.cli.demo._check_prerequisites",
        return_value=(True, "ok"),
    )
    def test_run_health_timeout(
        self,
        mock_prereq: MagicMock,
        mock_up: MagicMock,
        mock_health: MagicMock,
        tmp_path: "os.PathLike[str]",
    ) -> None:
        from pathlib import Path

        from tobira.cli import main

        docker_dir = Path(str(tmp_path))
        (docker_dir / "docker-compose.yml").write_text("", encoding="utf-8")

        result = main(["demo", "--docker-dir", str(docker_dir)])
        assert result == 1
