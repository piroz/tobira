"""Tests for tobira.backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tobira.backends.factory import create_backend
from tobira.backends.protocol import BackendProtocol, PredictionResult

_has_torch = True
try:
    import torch as _torch
except ImportError:
    _has_torch = False

requires_torch = pytest.mark.skipif(not _has_torch, reason="torch not installed")


class TestPredictionResult:
    def test_frozen(self) -> None:
        result = PredictionResult(
            label="spam", score=0.95, labels={"spam": 0.95, "ham": 0.05}
        )
        with pytest.raises(AttributeError):
            result.label = "ham"  # type: ignore[misc]

    def test_attributes(self) -> None:
        labels = {"spam": 0.9, "ham": 0.1}
        result = PredictionResult(label="spam", score=0.9, labels=labels)
        assert result.label == "spam"
        assert result.score == 0.9
        assert result.labels == labels


class TestBackendProtocol:
    """Runtime-checkable protocol tests.

    Note: runtime_checkable only verifies method existence, not signatures.
    Use mypy for full static type checking.
    """

    def test_fasttext_implements_protocol(self) -> None:
        from tobira.backends.fasttext import FastTextBackend

        assert issubclass(FastTextBackend, BackendProtocol)

    def test_bert_implements_protocol(self) -> None:
        from tobira.backends.bert import BertBackend

        assert issubclass(BertBackend, BackendProtocol)

    def test_custom_class_implements_protocol(self) -> None:
        class DummyBackend:
            def predict(self, text: str) -> PredictionResult:
                return PredictionResult(label="ham", score=1.0, labels={"ham": 1.0})

        assert isinstance(DummyBackend(), BackendProtocol)

    def test_non_conforming_class_rejected(self) -> None:
        class NotABackend:
            pass

        assert not isinstance(NotABackend(), BackendProtocol)


def _make_mock_fasttext() -> MagicMock:
    """Create a mock fasttext module with a mock model."""
    mock_ft = MagicMock()
    mock_model = MagicMock()
    mock_model.predict.return_value = (
        ["__label__spam", "__label__ham"],
        [0.95, 0.05],
    )
    mock_ft.load_model.return_value = mock_model
    return mock_ft


class TestFastTextBackend:
    @patch("tobira.backends.fasttext.Path.exists", return_value=True)
    @patch("tobira.backends.fasttext._import_fasttext")
    def test_predict(self, mock_import: MagicMock, _mock_exists: MagicMock) -> None:
        mock_ft = _make_mock_fasttext()
        mock_import.return_value = mock_ft

        from tobira.backends.fasttext import FastTextBackend

        backend = FastTextBackend(model_path="/tmp/model.bin")
        result = backend.predict("buy now!!!")

        assert result.label == "spam"
        assert result.score == pytest.approx(0.95)
        assert result.labels == {
            "spam": pytest.approx(0.95),
            "ham": pytest.approx(0.05),
        }
        mock_ft.load_model.return_value.predict.assert_called_once_with(
            "buy now!!!", k=-1
        )

    @patch("tobira.backends.fasttext.Path.exists", return_value=True)
    @patch("tobira.backends.fasttext._import_fasttext")
    def test_loads_model(self, mock_import: MagicMock, _mock_exists: MagicMock) -> None:
        mock_ft = _make_mock_fasttext()
        mock_import.return_value = mock_ft

        from tobira.backends.fasttext import FastTextBackend

        FastTextBackend(model_path="/tmp/model.bin")
        mock_ft.load_model.assert_called_once_with("/tmp/model.bin")

    def test_missing_model_file_raises(self) -> None:
        from tobira.backends.fasttext import FastTextBackend

        with pytest.raises(FileNotFoundError, match="model file not found"):
            FastTextBackend(model_path="/nonexistent/model.bin")


class TestBertBackend:
    @requires_torch
    @patch("tobira.backends.bert._import_deps")
    def test_predict(self, mock_import: MagicMock) -> None:
        mock_model = MagicMock()
        mock_logits = _torch.tensor([[2.0, 0.5]])
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_model.return_value = mock_output
        mock_model.config.id2label = {0: "spam", 1: "ham"}

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": _torch.tensor([[1, 2, 3]])}

        MockAutoModel = MagicMock()
        MockAutoModel.from_pretrained.return_value = mock_model
        MockAutoTokenizer = MagicMock()
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_import.return_value = (_torch, MockAutoModel, MockAutoTokenizer)

        from tobira.backends.bert import BertBackend

        backend = BertBackend(model_name="test-model", device="cpu")
        result = backend.predict("buy now!!!")

        assert result.label == "spam"
        assert result.score > 0.5
        assert "spam" in result.labels
        assert "ham" in result.labels
        assert result.labels["spam"] + result.labels["ham"] == pytest.approx(1.0)

    @requires_torch
    @patch("tobira.backends.bert._import_deps")
    def test_loads_model(self, mock_import: MagicMock) -> None:
        MockAutoModel = MagicMock()
        MockAutoTokenizer = MagicMock()
        mock_import.return_value = (_torch, MockAutoModel, MockAutoTokenizer)

        from tobira.backends.bert import BertBackend

        BertBackend(model_name="test-model", device="cpu")
        MockAutoModel.from_pretrained.assert_called_once_with("test-model")
        MockAutoTokenizer.from_pretrained.assert_called_once_with("test-model")

    @patch("tobira.backends.bert._import_deps")
    def test_auto_device_detection(self, mock_import: MagicMock) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda"

        mock_import.return_value = (mock_torch, MagicMock(), MagicMock())

        from tobira.backends.bert import BertBackend

        BertBackend(model_name="test-model")
        mock_torch.device.assert_called_with("cuda")


class TestFactory:
    def test_missing_type_raises(self) -> None:
        with pytest.raises(KeyError, match="type"):
            create_backend({})

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown backend type"):
            create_backend({"type": "nonexistent"})

    def test_fasttext_missing_model_path_raises(self) -> None:
        with pytest.raises(KeyError, match="model_path"):
            create_backend({"type": "fasttext"})

    @patch("tobira.backends.fasttext._import_fasttext")
    @patch("tobira.backends.fasttext.Path.exists", return_value=True)
    def test_fasttext_creation(
        self, _mock_exists: MagicMock, mock_import: MagicMock
    ) -> None:
        mock_import.return_value = _make_mock_fasttext()

        from tobira.backends.fasttext import FastTextBackend

        backend = create_backend({"type": "fasttext", "model_path": "/tmp/m.bin"})
        assert isinstance(backend, FastTextBackend)

    @patch("tobira.backends.bert._import_deps")
    def test_bert_creation(self, mock_import: MagicMock) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        mock_import.return_value = (mock_torch, MagicMock(), MagicMock())

        from tobira.backends.bert import BertBackend

        backend = create_backend(
            {"type": "bert", "model_name": "test-model", "device": "cpu"}
        )
        assert isinstance(backend, BertBackend)

    @patch("tobira.backends.bert._import_deps")
    def test_bert_default_model(self, mock_import: MagicMock) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        MockAutoTokenizer = MagicMock()
        mock_import.return_value = (mock_torch, MagicMock(), MockAutoTokenizer)

        create_backend({"type": "bert"})
        MockAutoTokenizer.from_pretrained.assert_called_once_with(
            "tohoku-nlp/bert-base-japanese-v3"
        )
