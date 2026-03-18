"""Tests for tobira.cli.train subcommand."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# --- helpers ---


@dataclass
class _MockTrainingConfig:
    model_name: str = "bert-base-uncased"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 5e-5
    max_length: int = 512
    device: str | None = None
    label_names: list[str] | None = None


@dataclass
class _MockTrainingResult:
    output_path: Path = Path("/tmp/model")
    onnx_path: Path | None = None


def _make_trainer_module(
    train_return: Any = None,
    train_side_effect: Exception | None = None,
) -> ModuleType:
    """Create a mock ``tobira.core.trainer`` module."""
    mod = ModuleType("tobira.core.trainer")
    mod.TrainingConfig = _MockTrainingConfig  # type: ignore[attr-defined]
    mock_train = MagicMock()
    if train_side_effect is not None:
        mock_train.side_effect = train_side_effect
    else:
        mock_train.return_value = (
            train_return if train_return is not None else _MockTrainingResult()
        )
    mod.train = mock_train  # type: ignore[attr-defined]
    return mod


def _make_pipeline_module(
    run_side_effect: Any = None,
) -> ModuleType:
    """Create a mock ``tobira.preprocessing.pipeline`` module."""
    mod = ModuleType("tobira.preprocessing.pipeline")
    mock_result = MagicMock()
    mock_result.text = "preprocessed"
    mock_pipeline_instance = MagicMock()
    if run_side_effect is not None:
        mock_pipeline_instance.run.side_effect = run_side_effect
    else:
        mock_pipeline_instance.run.return_value = mock_result
    mock_pipeline_cls = MagicMock(return_value=mock_pipeline_instance)
    mod.PreprocessingPipeline = mock_pipeline_cls  # type: ignore[attr-defined]
    return mod


def _write_training_config(tmp_path: Path, extra: str = "") -> Path:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        f'[training]\nmodel_name = "bert-base-uncased"\nepochs = 3\n{extra}',
        encoding="utf-8",
    )
    return config_file


def _write_csv_data(tmp_path: Path, n: int = 20) -> Path:
    csv_file = tmp_path / "data.csv"
    lines = ["text,label"] + [f"text{i},{i % 2}" for i in range(n)]
    csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_file


# --- parser tests ---


class TestTrainParser:
    def test_parser_has_train_subcommand(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train", "--config", "/tmp/c.toml",
            "--data", "/tmp/data.csv", "--output", "/tmp/out",
        ])
        assert args.command == "train"

    def test_parser_train_defaults(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train", "--config", "/tmp/c.toml",
            "--data", "/tmp/data.csv", "--output", "/tmp/out",
        ])
        assert args.config == "/tmp/c.toml"
        assert args.data == "/tmp/data.csv"
        assert args.output == "/tmp/out"
        assert args.skip_export is False
        assert args.split_ratio == "0.8,0.1,0.1"

    def test_parser_train_with_skip_export(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train", "--config", "/tmp/c.toml",
            "--data", "/tmp/data.csv", "--output", "/tmp/out",
            "--skip-export",
        ])
        assert args.skip_export is True

    def test_parser_train_with_split_ratio(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "train", "--config", "/tmp/c.toml",
            "--data", "/tmp/data.csv", "--output", "/tmp/out",
            "--split-ratio", "0.7,0.15,0.15",
        ])
        assert args.split_ratio == "0.7,0.15,0.15"

    def test_parser_train_missing_required_args(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["train"])

    def test_train_help_exits_with_0(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["train", "--help"])
        assert exc_info.value.code == 0


# --- dispatch test ---


class TestTrainDispatch:
    @patch("tobira.cli.train._run")
    def test_train_dispatches_to_run(self, mock_run: MagicMock) -> None:
        from tobira.cli import main

        mock_run.return_value = 0
        result = main([
            "train", "--config", "/tmp/c.toml",
            "--data", "/tmp/data.csv", "--output", "/tmp/out",
        ])

        assert result == 0
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args.config == "/tmp/c.toml"
        assert args.data == "/tmp/data.csv"


# --- unit tests for helpers ---


class TestLoadLabelledData:
    def test_load_csv(self, tmp_path: "os.PathLike[str]") -> None:
        from tobira.cli.train import _load_labelled_data

        p = Path(str(tmp_path))
        csv_file = p / "data.csv"
        csv_file.write_text(
            "text,label\nspam msg,1\nham msg,0\n", encoding="utf-8"
        )

        rows = _load_labelled_data(csv_file)
        assert len(rows) == 2
        assert rows[0]["text"] == "spam msg"
        assert rows[0]["label"] == "1"

    def test_load_jsonl(self, tmp_path: "os.PathLike[str]") -> None:
        from tobira.cli.train import _load_labelled_data

        p = Path(str(tmp_path))
        jsonl_file = p / "data.jsonl"
        records = [
            {"text": "spam msg", "label": "1"},
            {"text": "ham msg", "label": "0"},
        ]
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        rows = _load_labelled_data(jsonl_file)
        assert len(rows) == 2

    def test_load_csv_missing_columns(self, tmp_path: "os.PathLike[str]") -> None:
        from tobira.cli.train import _load_labelled_data

        p = Path(str(tmp_path))
        csv_file = p / "bad.csv"
        csv_file.write_text("foo,bar\na,b\n", encoding="utf-8")

        with pytest.raises(ValueError, match="'text' and 'label'"):
            _load_labelled_data(csv_file)

    def test_load_unsupported_format(self, tmp_path: "os.PathLike[str]") -> None:
        from tobira.cli.train import _load_labelled_data

        p = Path(str(tmp_path))
        txt_file = p / "data.txt"
        txt_file.write_text("text,label\nhello,1\n", encoding="utf-8")

        with pytest.raises(ValueError, match="unsupported file format"):
            _load_labelled_data(txt_file)


class TestParseSplitRatio:
    def test_valid_ratio(self) -> None:
        from tobira.cli.train import _parse_split_ratio

        train_r, val_r, test_r = _parse_split_ratio("0.8,0.1,0.1")
        assert train_r == pytest.approx(0.8)
        assert val_r == pytest.approx(0.1)
        assert test_r == pytest.approx(0.1)

    def test_invalid_parts_count(self) -> None:
        from tobira.cli.train import _parse_split_ratio

        with pytest.raises(ValueError, match="3 parts"):
            _parse_split_ratio("0.8,0.2")

    def test_invalid_sum(self) -> None:
        from tobira.cli.train import _parse_split_ratio

        with pytest.raises(ValueError, match="sum to 1.0"):
            _parse_split_ratio("0.5,0.1,0.1")


class TestSplitData:
    def test_split_data(self) -> None:
        from tobira.cli.train import _split_data

        rows = [{"text": str(i), "label": "1"} for i in range(10)]
        train_set, val_set, test_set = _split_data(rows, 0.8, 0.1)
        assert len(train_set) == 8
        assert len(val_set) == 1
        assert len(test_set) == 1

    def test_split_data_small(self) -> None:
        from tobira.cli.train import _split_data

        rows = [{"text": "a", "label": "1"}, {"text": "b", "label": "0"}]
        train_set, val_set, test_set = _split_data(rows, 0.8, 0.1)
        assert len(train_set) == 1
        assert len(val_set) == 0
        assert len(test_set) == 1


# --- _run integration tests ---


class TestTrainRun:
    def test_config_not_found(self) -> None:
        from tobira.cli import main

        result = main([
            "train", "--config", "/nonexistent/config.toml",
            "--data", "/tmp/data.csv", "--output", "/tmp/out",
        ])
        assert result == 1

    def test_missing_training_section(self, tmp_path: "os.PathLike[str]") -> None:
        from tobira.cli import main

        p = Path(str(tmp_path))
        config_file = p / "config.toml"
        config_file.write_text("[server]\nhost = '0.0.0.0'\n", encoding="utf-8")

        csv_file = p / "data.csv"
        csv_file.write_text("text,label\nspam,1\nham,0\n", encoding="utf-8")

        result = main([
            "train", "--config", str(config_file),
            "--data", str(csv_file), "--output", str(p / "out"),
        ])
        assert result == 1

    def test_data_file_not_found(self, tmp_path: "os.PathLike[str]") -> None:
        from tobira.cli import main

        p = Path(str(tmp_path))
        config_file = _write_training_config(p)

        result = main([
            "train", "--config", str(config_file),
            "--data", "/nonexistent/data.csv",
            "--output", str(p / "out"),
        ])
        assert result == 1

    def test_empty_data_file(self, tmp_path: "os.PathLike[str]") -> None:
        from tobira.cli import main

        p = Path(str(tmp_path))
        config_file = _write_training_config(p)
        csv_file = p / "data.csv"
        csv_file.write_text("text,label\n", encoding="utf-8")

        result = main([
            "train", "--config", str(config_file),
            "--data", str(csv_file), "--output", str(p / "out"),
        ])
        assert result == 1

    def test_invalid_split_ratio(self, tmp_path: "os.PathLike[str]") -> None:
        from tobira.cli import main

        p = Path(str(tmp_path))
        config_file = _write_training_config(p)
        csv_file = _write_csv_data(p)

        result = main([
            "train", "--config", str(config_file),
            "--data", str(csv_file), "--output", str(p / "out"),
            "--split-ratio", "0.5,0.1",
        ])
        assert result == 1

    def test_full_pipeline_skip_export(
        self, tmp_path: "os.PathLike[str]", capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tobira.cli import main

        p = Path(str(tmp_path))
        config_file = _write_training_config(p)
        csv_file = _write_csv_data(p)
        output_dir = p / "out"

        trainer_mod = _make_trainer_module(
            train_return=_MockTrainingResult(output_path=output_dir / "model"),
        )

        with patch.dict(sys.modules, {"tobira.core.trainer": trainer_mod}):
            result = main([
                "train", "--config", str(config_file),
                "--data", str(csv_file), "--output", str(output_dir),
                "--skip-export",
            ])

        assert result == 0
        trainer_mod.train.assert_called_once()  # type: ignore[union-attr]

        captured = capsys.readouterr()
        assert "Training pipeline complete" in captured.out
        assert "ONNX export skipped" in captured.out

    @patch("tobira.core.export.export_onnx")
    @patch("tobira.core.export.quantize_dynamic")
    @patch("tobira.backends.factory.create_backend")
    def test_full_pipeline_with_export(
        self,
        mock_create_backend: MagicMock,
        mock_quantize: MagicMock,
        mock_export: MagicMock,
        tmp_path: "os.PathLike[str]",
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from tobira.backends.protocol import PredictionResult
        from tobira.cli import main

        p = Path(str(tmp_path))
        config_file = _write_training_config(p)
        csv_file = _write_csv_data(p)
        output_dir = p / "out"

        trainer_mod = _make_trainer_module(
            train_return=_MockTrainingResult(output_path=output_dir / "model"),
        )

        mock_backend = MagicMock()
        mock_backend.predict.return_value = PredictionResult(
            label="spam", score=0.9, labels={"spam": 0.9, "ham": 0.1}
        )
        mock_create_backend.return_value = mock_backend

        mock_export.return_value = output_dir / "model.onnx"
        mock_quantize.return_value = output_dir / "model_int8.onnx"

        with patch.dict(sys.modules, {"tobira.core.trainer": trainer_mod}):
            result = main([
                "train", "--config", str(config_file),
                "--data", str(csv_file), "--output", str(output_dir),
            ])

        assert result == 0
        mock_export.assert_called_once()

        captured = capsys.readouterr()
        assert "Training pipeline complete" in captured.out

    def test_training_error(
        self, tmp_path: "os.PathLike[str]"
    ) -> None:
        from tobira.cli import main

        p = Path(str(tmp_path))
        config_file = _write_training_config(p)
        csv_file = _write_csv_data(p)

        trainer_mod = _make_trainer_module(
            train_side_effect=RuntimeError("CUDA out of memory"),
        )

        with patch.dict(sys.modules, {"tobira.core.trainer": trainer_mod}):
            result = main([
                "train", "--config", str(config_file),
                "--data", str(csv_file), "--output", str(p / "out"),
                "--skip-export",
            ])

        assert result == 1

    def test_trainer_not_installed(
        self, tmp_path: "os.PathLike[str]"
    ) -> None:
        """When tobira.core.trainer is not importable, return 1."""
        from tobira.cli import main

        p = Path(str(tmp_path))
        config_file = _write_training_config(p)
        csv_file = _write_csv_data(p)

        # Ensure trainer module is not importable
        saved = sys.modules.pop("tobira.core.trainer", None)
        try:
            with patch.dict(sys.modules, {"tobira.core.trainer": None}):
                result = main([
                    "train", "--config", str(config_file),
                    "--data", str(csv_file), "--output", str(p / "out"),
                    "--skip-export",
                ])
            assert result == 1
        finally:
            if saved is not None:
                sys.modules["tobira.core.trainer"] = saved

    def test_with_preprocessing(
        self, tmp_path: "os.PathLike[str]"
    ) -> None:
        from tobira.cli import main

        p = Path(str(tmp_path))
        config_file = _write_training_config(
            p, extra="[preprocessing]\nanonymize = true\n"
        )
        csv_file = _write_csv_data(p)
        output_dir = p / "out"

        trainer_mod = _make_trainer_module(
            train_return=_MockTrainingResult(output_path=output_dir / "model"),
        )
        pipeline_mod = _make_pipeline_module()

        with patch.dict(sys.modules, {
            "tobira.core.trainer": trainer_mod,
            "tobira.preprocessing.pipeline": pipeline_mod,
        }):
            result = main([
                "train", "--config", str(config_file),
                "--data", str(csv_file), "--output", str(output_dir),
                "--skip-export",
            ])

        assert result == 0
        assert pipeline_mod.PreprocessingPipeline.call_count >= 1  # type: ignore[union-attr]

    def test_preprocessing_not_installed(
        self, tmp_path: "os.PathLike[str]"
    ) -> None:
        from tobira.cli import main

        p = Path(str(tmp_path))
        config_file = _write_training_config(
            p, extra="[preprocessing]\nanonymize = true\n"
        )
        csv_file = _write_csv_data(p)

        with patch.dict(sys.modules, {"tobira.preprocessing.pipeline": None}):
            result = main([
                "train", "--config", str(config_file),
                "--data", str(csv_file), "--output", str(p / "out"),
                "--skip-export",
            ])

        assert result == 1
