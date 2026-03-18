"""Tests for tobira.core.trainer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tobira.core.trainer import (
    TrainingConfig,
    TrainingResult,
    _split_data,
    load_training_data,
)


class TestTrainingResult:
    def test_frozen(self) -> None:
        result = TrainingResult(
            model_name="bert",
            output_path="/tmp/out",
            num_samples=100,
            num_labels=2,
            epochs=3,
            final_loss=0.1,
        )
        with pytest.raises(AttributeError):
            result.epochs = 10  # type: ignore[misc]

    def test_attributes(self) -> None:
        result = TrainingResult(
            model_name="bert",
            output_path="/tmp/out",
            num_samples=100,
            num_labels=2,
            epochs=3,
            final_loss=0.1,
        )
        assert result.model_name == "bert"
        assert result.num_samples == 100
        assert result.final_loss == pytest.approx(0.1)
        assert result.onnx_path is None

    def test_with_onnx_path(self) -> None:
        result = TrainingResult(
            model_name="bert",
            output_path="/tmp/out",
            num_samples=50,
            num_labels=2,
            epochs=1,
            final_loss=0.5,
            onnx_path="/tmp/model.onnx",
        )
        assert result.onnx_path == "/tmp/model.onnx"


class TestTrainingConfig:
    def test_defaults(self) -> None:
        config = TrainingConfig()
        assert config.model_name == "tohoku-nlp/bert-base-japanese-v3"
        assert config.epochs == 3
        assert config.batch_size == 16
        assert config.learning_rate == pytest.approx(2e-5)
        assert config.weight_decay == pytest.approx(0.01)
        assert config.warmup_ratio == pytest.approx(0.1)
        assert config.max_length == 512
        assert config.device is None
        assert config.label_names == ["ham", "spam"]
        assert config.eval_split == pytest.approx(0.1)
        assert config.seed == 42
        assert config.export_onnx is False
        assert config.checkpoint_dir is None

    def test_custom_values(self) -> None:
        config = TrainingConfig(
            model_name="bert-tiny",
            epochs=10,
            batch_size=32,
            learning_rate=1e-4,
            device="cpu",
            label_names=["ham", "spam", "phishing"],
        )
        assert config.model_name == "bert-tiny"
        assert config.epochs == 10
        assert config.batch_size == 32
        assert len(config.label_names) == 3


class TestLoadTrainingData:
    def test_csv(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("text,label\nhello,ham\nfree money,spam\n")
        records = load_training_data(csv_file)
        assert len(records) == 2
        assert records[0] == {"text": "hello", "label": "ham"}
        assert records[1] == {"text": "free money", "label": "spam"}

    def test_jsonl(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            json.dumps({"text": "hello", "label": "ham"}) + "\n"
            + json.dumps({"text": "buy now", "label": "spam"}) + "\n"
        )
        records = load_training_data(jsonl_file)
        assert len(records) == 2
        assert records[0]["label"] == "ham"
        assert records[1]["label"] == "spam"

    def test_jsonl_skips_blank_lines(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "data.jsonl"
        content = (
            json.dumps({"text": "a", "label": "ham"}) + "\n"
            "\n"
            + json.dumps({"text": "b", "label": "spam"}) + "\n"
        )
        jsonl_file.write_text(content)
        records = load_training_data(jsonl_file)
        assert len(records) == 2

    def test_custom_columns(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("content,category\nhello,ham\n")
        records = load_training_data(
            csv_file, text_column="content", label_column="category"
        )
        assert records[0] == {"text": "hello", "label": "ham"}

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_training_data("/nonexistent/path.csv")

    def test_unsupported_format(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "data.txt"
        txt_file.write_text("hello\n")
        with pytest.raises(ValueError, match="Unsupported"):
            load_training_data(txt_file)

    def test_csv_missing_text_column(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("content,label\nhello,ham\n")
        with pytest.raises(ValueError, match="text"):
            load_training_data(csv_file)

    def test_csv_missing_label_column(self, tmp_path: Path) -> None:
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("text,category\nhello,ham\n")
        with pytest.raises(ValueError, match="label"):
            load_training_data(csv_file)

    def test_jsonl_missing_text_field(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(json.dumps({"content": "hi", "label": "ham"}) + "\n")
        with pytest.raises(ValueError, match="text"):
            load_training_data(jsonl_file)

    def test_jsonl_missing_label_field(self, tmp_path: Path) -> None:
        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(json.dumps({"text": "hi", "category": "ham"}) + "\n")
        with pytest.raises(ValueError, match="label"):
            load_training_data(jsonl_file)


class TestSplitData:
    def test_basic_split(self) -> None:
        records = [{"text": f"t{i}", "label": "ham"} for i in range(10)]
        train, eval_ = _split_data(records, eval_split=0.2, seed=42)
        assert len(train) == 8
        assert len(eval_) == 2
        assert len(train) + len(eval_) == 10

    def test_deterministic(self) -> None:
        records = [{"text": f"t{i}", "label": "ham"} for i in range(10)]
        train1, eval1 = _split_data(records, eval_split=0.2, seed=42)
        train2, eval2 = _split_data(records, eval_split=0.2, seed=42)
        assert train1 == train2
        assert eval1 == eval2

    def test_small_dataset(self) -> None:
        records = [{"text": "t0", "label": "ham"}]
        train, eval_ = _split_data(records, eval_split=0.1, seed=42)
        # Dataset too small: all goes to training
        assert len(train) == 1
        assert len(eval_) == 0

    def test_zero_split(self) -> None:
        records = [{"text": f"t{i}", "label": "ham"} for i in range(5)]
        train, eval_ = _split_data(records, eval_split=0.0, seed=42)
        assert len(train) == 5
        assert len(eval_) == 0


class TestTrain:
    def test_import_error(self) -> None:
        from tobira.core.trainer import train

        data = [{"text": "test", "label": "spam"}]
        with patch(
            "tobira.core.trainer._import_deps",
            side_effect=ImportError("torch not found"),
        ):
            with pytest.raises(ImportError):
                train(data, "/tmp/out")

    def test_empty_data_raises(self) -> None:
        from tobira.core.trainer import train

        with pytest.raises(ValueError, match="must not be empty"):
            train([], "/tmp/out")

    def test_unknown_label_raises(self) -> None:
        from tobira.core.trainer import train

        data = [{"text": "test", "label": "phishing"}]
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch(
            "tobira.core.trainer._import_deps",
            return_value=(mock_torch, MagicMock(), MagicMock()),
        ):
            with pytest.raises(ValueError, match="unknown labels"):
                train(data, "/tmp/out")

    def test_train_with_mock_torch(self, tmp_path: Path) -> None:
        from tobira.core.trainer import train

        data = [
            {"text": "spam email", "label": "spam"},
            {"text": "good email", "label": "ham"},
            {"text": "another spam", "label": "spam"},
            {"text": "normal email", "label": "ham"},
        ]

        # Create mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()
        mock_torch.long = "long"

        mock_tensor = MagicMock()
        mock_tensor.__len__ = lambda self: 4
        mock_tensor.__getitem__ = lambda self, idx: MagicMock(
            to=lambda device: MagicMock()
        )
        mock_tensor.to = lambda self, device: self
        mock_torch.tensor.return_value = mock_tensor
        mock_torch.randperm.return_value = MagicMock(
            __getitem__=lambda self, s: MagicMock()
        )
        mock_torch.manual_seed = MagicMock()

        # Mock loss
        mock_loss = MagicMock()
        mock_loss.__float__ = lambda self: 0.5
        mock_loss.backward = MagicMock()

        mock_outputs = MagicMock()
        mock_outputs.loss = mock_loss
        mock_outputs.logits = MagicMock()

        mock_model = MagicMock()
        mock_model.return_value = mock_outputs
        mock_model.__call__ = lambda self, **kwargs: mock_outputs
        mock_model.parameters.return_value = []
        mock_model.named_parameters.return_value = []

        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_torch.nn.utils.clip_grad_norm_ = MagicMock()
        mock_torch.optim.AdamW.return_value = MagicMock()
        mock_torch.optim.lr_scheduler.LambdaLR.return_value = MagicMock()
        mock_torch.argmax.return_value = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        with patch(
            "tobira.core.trainer._import_deps",
            return_value=(
                mock_torch,
                mock_auto_model,
                mock_auto_tokenizer,
            ),
        ):
            config = TrainingConfig(
                model_name="test-model",
                epochs=1,
                batch_size=4,
                device="cpu",
                eval_split=0.0,
            )
            result = train(
                data=data,
                output_path=str(tmp_path / "trained"),
                config=config,
            )

        assert result.model_name == "test-model"
        assert result.num_samples == 4
        assert result.num_labels == 2
        assert result.epochs == 1
        assert result.onnx_path is None


class TestTrainerImports:
    def test_core_module_exports(self) -> None:
        from tobira.core import TrainingConfig, TrainingResult, train

        assert TrainingConfig is not None
        assert TrainingResult is not None
        assert callable(train)
