"""Tests for tobira.backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tobira.backends.bert import BertBackend
from tobira.backends.factory import create_backend
from tobira.backends.fasttext import FastTextBackend
from tobira.backends.protocol import BackendProtocol, PredictionResult


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
    def test_fasttext_implements_protocol(self) -> None:
        assert issubclass(FastTextBackend, BackendProtocol)

    def test_bert_implements_protocol(self) -> None:
        assert issubclass(BertBackend, BackendProtocol)

    def test_custom_class_implements_protocol(self) -> None:
        class DummyBackend:
            def predict(self, text: str) -> PredictionResult:
                return PredictionResult(label="ham", score=1.0, labels={"ham": 1.0})

        assert isinstance(DummyBackend(), BackendProtocol)


class TestFastTextBackend:
    @patch("tobira.backends.fasttext.fasttext")
    def test_predict(self, mock_ft: MagicMock) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = (
            ["__label__spam", "__label__ham"],
            [0.95, 0.05],
        )
        mock_ft.load_model.return_value = mock_model

        backend = FastTextBackend(model_path="/tmp/model.bin")
        result = backend.predict("buy now!!!")

        assert result.label == "spam"
        assert result.score == pytest.approx(0.95)
        assert result.labels == {"spam": pytest.approx(0.95), "ham": pytest.approx(0.05)}
        mock_model.predict.assert_called_once_with("buy now!!!", k=-1)

    @patch("tobira.backends.fasttext.fasttext")
    def test_loads_model(self, mock_ft: MagicMock) -> None:
        FastTextBackend(model_path="/tmp/model.bin")
        mock_ft.load_model.assert_called_once_with("/tmp/model.bin")


class TestBertBackend:
    @patch("tobira.backends.bert.AutoModelForSequenceClassification")
    @patch("tobira.backends.bert.AutoTokenizer")
    @patch("tobira.backends.bert.torch")
    def test_predict(
        self, mock_torch: MagicMock, mock_tokenizer_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        import torch

        mock_logits = torch.tensor([[2.0, 0.5]])
        mock_output = MagicMock()
        mock_output.logits = mock_logits
        mock_model = MagicMock()
        mock_model.return_value = mock_output
        mock_model.config.id2label = {0: "spam", 1: "ham"}
        mock_model_cls.from_pretrained.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        # Use real torch functions for softmax/argmax
        mock_torch.softmax = torch.softmax
        mock_torch.no_grad = torch.no_grad

        backend = BertBackend(model_name="test-model", device="cpu")
        result = backend.predict("buy now!!!")

        assert result.label == "spam"
        assert result.score > 0.5
        assert "spam" in result.labels
        assert "ham" in result.labels
        assert result.labels["spam"] + result.labels["ham"] == pytest.approx(1.0)

    @patch("tobira.backends.bert.AutoModelForSequenceClassification")
    @patch("tobira.backends.bert.AutoTokenizer")
    @patch("tobira.backends.bert.torch")
    def test_loads_model(
        self, mock_torch: MagicMock, mock_tokenizer_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"

        BertBackend(model_name="test-model", device="cpu")
        mock_model_cls.from_pretrained.assert_called_once_with("test-model")
        mock_tokenizer_cls.from_pretrained.assert_called_once_with("test-model")

    @patch("tobira.backends.bert.AutoModelForSequenceClassification")
    @patch("tobira.backends.bert.AutoTokenizer")
    @patch("tobira.backends.bert.torch")
    def test_auto_device_detection(
        self, mock_torch: MagicMock, mock_tokenizer_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda"

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

    @patch("tobira.backends.fasttext.fasttext")
    def test_fasttext_creation(self, mock_ft: MagicMock) -> None:
        backend = create_backend({"type": "fasttext", "model_path": "/tmp/m.bin"})
        assert isinstance(backend, FastTextBackend)

    @patch("tobira.backends.bert.AutoModelForSequenceClassification")
    @patch("tobira.backends.bert.AutoTokenizer")
    @patch("tobira.backends.bert.torch")
    def test_bert_creation(
        self, mock_torch: MagicMock, mock_tokenizer_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        backend = create_backend({"type": "bert", "model_name": "test-model", "device": "cpu"})
        assert isinstance(backend, BertBackend)

    @patch("tobira.backends.bert.AutoModelForSequenceClassification")
    @patch("tobira.backends.bert.AutoTokenizer")
    @patch("tobira.backends.bert.torch")
    def test_bert_default_model(
        self, mock_torch: MagicMock, mock_tokenizer_cls: MagicMock, mock_model_cls: MagicMock
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        create_backend({"type": "bert"})
        mock_tokenizer_cls.from_pretrained.assert_called_once_with(
            "tohoku-nlp/bert-base-japanese-v3"
        )
