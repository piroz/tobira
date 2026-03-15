"""Tests for tobira.core.distillation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tobira.backends.protocol import PredictionResult
from tobira.core.distillation import (
    DistillationConfig,
    DistillationResult,
    SoftLabel,
    generate_soft_labels,
    load_soft_labels,
    save_soft_labels,
)


class _MockTeacher:
    """A mock teacher backend for testing."""

    def __init__(self, spam_score: float = 0.9) -> None:
        self.spam_score = spam_score
        self.call_count = 0

    def predict(self, text: str) -> PredictionResult:
        self.call_count += 1
        return PredictionResult(
            label="spam" if self.spam_score >= 0.5 else "ham",
            score=self.spam_score,
            labels={"spam": self.spam_score, "ham": 1.0 - self.spam_score},
        )


class TestSoftLabel:
    def test_frozen(self) -> None:
        sl = SoftLabel(text="hello", labels={"spam": 0.9, "ham": 0.1})
        with pytest.raises(AttributeError):
            sl.text = "other"  # type: ignore[misc]

    def test_attributes(self) -> None:
        sl = SoftLabel(text="test email", labels={"spam": 0.8, "ham": 0.2})
        assert sl.text == "test email"
        assert sl.labels["spam"] == pytest.approx(0.8)
        assert sl.labels["ham"] == pytest.approx(0.2)


class TestDistillationResult:
    def test_frozen(self) -> None:
        result = DistillationResult(
            teacher_name="bert",
            student_model_name="distilbert",
            output_path="/tmp/out",
            num_samples=100,
            temperature=3.0,
            alpha=0.7,
            epochs=5,
        )
        with pytest.raises(AttributeError):
            result.epochs = 10  # type: ignore[misc]

    def test_attributes(self) -> None:
        result = DistillationResult(
            teacher_name="bert",
            student_model_name="distilbert",
            output_path="/tmp/out",
            num_samples=100,
            temperature=3.0,
            alpha=0.7,
            epochs=5,
        )
        assert result.teacher_name == "bert"
        assert result.num_samples == 100
        assert result.temperature == pytest.approx(3.0)


class TestDistillationConfig:
    def test_defaults(self) -> None:
        config = DistillationConfig()
        assert config.student_model_name == "distilbert-base-uncased"
        assert config.temperature == pytest.approx(3.0)
        assert config.alpha == pytest.approx(0.7)
        assert config.epochs == 5
        assert config.batch_size == 16
        assert config.learning_rate == pytest.approx(5e-5)
        assert config.max_length == 512
        assert config.device is None
        assert config.label_names == ["ham", "spam"]

    def test_custom_values(self) -> None:
        config = DistillationConfig(
            student_model_name="bert-tiny",
            temperature=5.0,
            alpha=0.5,
            epochs=10,
            device="cpu",
        )
        assert config.student_model_name == "bert-tiny"
        assert config.temperature == pytest.approx(5.0)
        assert config.epochs == 10


class TestGenerateSoftLabels:
    def test_basic(self) -> None:
        teacher = _MockTeacher(spam_score=0.85)
        texts = ["spam email 1", "spam email 2", "spam email 3"]
        result = generate_soft_labels(teacher, texts)

        assert len(result) == 3
        assert teacher.call_count == 3
        for sl in result:
            assert sl.labels["spam"] == pytest.approx(0.85)
            assert sl.labels["ham"] == pytest.approx(0.15)

    def test_preserves_text(self) -> None:
        teacher = _MockTeacher()
        texts = ["hello", "world"]
        result = generate_soft_labels(teacher, texts)
        assert result[0].text == "hello"
        assert result[1].text == "world"

    def test_empty_raises(self) -> None:
        teacher = _MockTeacher()
        with pytest.raises(ValueError, match="must not be empty"):
            generate_soft_labels(teacher, [])

    def test_single_sample(self) -> None:
        teacher = _MockTeacher(spam_score=0.3)
        result = generate_soft_labels(teacher, ["one email"])
        assert len(result) == 1
        assert result[0].labels["spam"] == pytest.approx(0.3)


class TestSaveSoftLabels:
    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        labels = [
            SoftLabel(text="hello", labels={"spam": 0.9, "ham": 0.1}),
            SoftLabel(text="world", labels={"spam": 0.2, "ham": 0.8}),
        ]
        out = tmp_path / "labels.jsonl"
        save_soft_labels(labels, out)

        assert out.exists()
        loaded = load_soft_labels(out)
        assert len(loaded) == 2
        assert loaded[0].text == "hello"
        assert loaded[0].labels["spam"] == pytest.approx(0.9)
        assert loaded[1].text == "world"
        assert loaded[1].labels["ham"] == pytest.approx(0.8)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "sub" / "dir" / "labels.jsonl"
        save_soft_labels([SoftLabel(text="x", labels={"spam": 0.5, "ham": 0.5})], out)
        assert out.exists()

    def test_jsonl_format(self, tmp_path: Path) -> None:
        labels = [SoftLabel(text="test", labels={"spam": 0.7, "ham": 0.3})]
        out = tmp_path / "labels.jsonl"
        save_soft_labels(labels, out)

        with open(out) as f:
            lines = f.readlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["text"] == "test"
        assert record["labels"]["spam"] == pytest.approx(0.7)


class TestLoadSoftLabels:
    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_soft_labels("/nonexistent/path.jsonl")

    def test_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        result = load_soft_labels(empty)
        assert result == []

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        content = (
            json.dumps({"text": "a", "labels": {"spam": 0.5, "ham": 0.5}}) + "\n"
            "\n"
            + json.dumps({"text": "b", "labels": {"spam": 0.6, "ham": 0.4}}) + "\n"
        )
        f = tmp_path / "labels.jsonl"
        f.write_text(content)
        result = load_soft_labels(f)
        assert len(result) == 2


class TestDistill:
    def test_import_error(self) -> None:
        """Distill should raise ImportError when torch is not available."""
        from tobira.core.distillation import distill

        labels = [SoftLabel(text="test", labels={"ham": 0.1, "spam": 0.9})]
        with patch(
            "tobira.core.distillation._import_deps",
            side_effect=ImportError("torch not found"),
        ):
            with pytest.raises(ImportError):
                distill(labels, "/tmp/out")

    def test_empty_labels_raises(self) -> None:
        """Distill should raise ValueError for empty soft labels."""
        from tobira.core.distillation import distill

        with pytest.raises(ValueError, match="must not be empty"):
            distill([], "/tmp/out")

    def test_distill_with_mock_torch(self, tmp_path: Path) -> None:
        """Test distillation flow with mocked torch and transformers."""
        from tobira.core.distillation import distill

        soft_labels = [
            SoftLabel(text="spam email", labels={"ham": 0.1, "spam": 0.9}),
            SoftLabel(text="good email", labels={"ham": 0.8, "spam": 0.2}),
        ]

        # Create mock torch module
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = MagicMock()
        mock_torch.tensor.side_effect = lambda data, dtype=None: MagicMock(
            __len__=lambda self: len(data),
            __getitem__=lambda self, idx: MagicMock(to=lambda device: MagicMock()),
            to=lambda self, device: self,
        )
        mock_torch.randperm.return_value = MagicMock(
            __getitem__=lambda self, s: MagicMock()
        )
        mock_torch.float32 = "float32"
        mock_torch.long = "long"
        mock_torch.nn.functional.log_softmax = MagicMock(return_value=MagicMock())
        mock_torch.nn.functional.softmax = MagicMock(return_value=MagicMock())
        mock_torch.nn.functional.kl_div = MagicMock(
            return_value=MagicMock(
                __mul__=lambda self, x: MagicMock(
                    __add__=lambda self, x: MagicMock(
                        backward=MagicMock(),
                        __float__=lambda self: 0.5,
                    )
                ),
                __rmul__=lambda self, x: MagicMock(
                    __add__=lambda self, x: MagicMock(
                        backward=MagicMock(),
                        __float__=lambda self: 0.5,
                    )
                ),
            )
        )
        mock_torch.nn.functional.cross_entropy = MagicMock(
            return_value=MagicMock(
                __mul__=lambda self, x: MagicMock(),
                __rmul__=lambda self, x: MagicMock(),
            )
        )
        mock_torch.optim.AdamW.return_value = MagicMock()

        mock_model = MagicMock()
        mock_model.return_value.logits = MagicMock()
        mock_model.parameters.return_value = []

        mock_auto_model = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        mock_auto_tokenizer = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_training_args = MagicMock()

        with patch(
            "tobira.core.distillation._import_deps",
            return_value=(
                mock_torch,
                mock_auto_model,
                mock_auto_tokenizer,
                mock_training_args,
            ),
        ):
            config = DistillationConfig(
                student_model_name="test-model",
                epochs=1,
                batch_size=2,
                device="cpu",
            )
            result = distill(
                soft_labels=soft_labels,
                output_path=str(tmp_path / "distilled"),
                config=config,
                teacher_name="bert",
            )

        assert result.teacher_name == "bert"
        assert result.student_model_name == "test-model"
        assert result.num_samples == 2
        assert result.epochs == 1


class TestDistillCLI:
    def test_register(self) -> None:
        """Test that distill subcommand is registered."""
        from tobira.cli import build_parser

        parser = build_parser()
        # Verify distill is in the subcommands by checking --help exits
        with pytest.raises(SystemExit):
            parser.parse_args(["distill", "--help"])

    def test_register_exists(self) -> None:
        """Test that the distill command can be imported."""
        from tobira.cli.distill import register

        assert callable(register)

    def test_load_texts_csv(self, tmp_path: Path) -> None:
        """Test loading texts from CSV."""
        from tobira.cli.distill import _load_texts

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("text,other\nhello,1\nworld,2\n")
        texts = _load_texts(csv_file)
        assert texts == ["hello", "world"]

    def test_load_texts_jsonl(self, tmp_path: Path) -> None:
        """Test loading texts from JSONL."""
        from tobira.cli.distill import _load_texts

        jsonl_file = tmp_path / "data.jsonl"
        jsonl_file.write_text(
            json.dumps({"text": "hello"}) + "\n"
            + json.dumps({"text": "world"}) + "\n"
        )
        texts = _load_texts(jsonl_file)
        assert texts == ["hello", "world"]

    def test_load_texts_unsupported(self, tmp_path: Path) -> None:
        """Test unsupported format raises ValueError."""
        from tobira.cli.distill import _load_texts

        txt_file = tmp_path / "data.txt"
        txt_file.write_text("hello\n")
        with pytest.raises(ValueError, match="unsupported"):
            _load_texts(txt_file)

    def test_load_texts_missing_column(self, tmp_path: Path) -> None:
        """Test missing text column raises ValueError."""
        from tobira.cli.distill import _load_texts

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("content,label\nhello,spam\n")
        with pytest.raises(ValueError, match="text"):
            _load_texts(csv_file)
