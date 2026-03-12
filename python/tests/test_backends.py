"""Tests for tobira.backends."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
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

    def test_onnx_implements_protocol(self) -> None:
        from tobira.backends.onnx import OnnxBackend

        assert issubclass(OnnxBackend, BackendProtocol)

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


class TestFastTextImportError:
    @patch.dict("sys.modules", {"fasttext": None})
    def test_missing_fasttext_raises(self) -> None:
        from tobira.backends.fasttext import _import_fasttext

        with pytest.raises(ImportError, match="fasttext is required"):
            _import_fasttext()


class TestBertImportError:
    @patch.dict("sys.modules", {"torch": None})
    def test_missing_torch_raises(self) -> None:
        from tobira.backends.bert import _import_deps

        with pytest.raises(ImportError, match="torch and transformers are required"):
            _import_deps()


class TestBertCpuFallback:
    @patch("tobira.backends.bert._import_deps")
    def test_cpu_fallback_when_no_cuda(self, mock_import: MagicMock) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device.return_value = "cpu"
        mock_import.return_value = (mock_torch, MagicMock(), MagicMock())

        from tobira.backends.bert import BertBackend

        BertBackend(model_name="test-model")
        mock_torch.device.assert_called_with("cpu")


class TestOnnxBackend:
    @patch("tobira.backends.onnx.Path.exists", return_value=True)
    @patch("tobira.backends.onnx._import_deps")
    def test_predict(self, mock_import: MagicMock, _mock_exists: MagicMock) -> None:
        mock_ort = MagicMock()
        mock_session = MagicMock()
        mock_input_0 = MagicMock()
        mock_input_0.name = "input_ids"
        mock_input_1 = MagicMock()
        mock_input_1.name = "attention_mask"
        mock_session.get_inputs.return_value = [mock_input_0, mock_input_1]
        mock_session.run.return_value = [np.array([[2.0, 0.5]])]
        mock_ort.InferenceSession.return_value = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]]),
        }

        MockAutoConfig = MagicMock()
        mock_config = MagicMock()
        mock_config.id2label = {0: "spam", 1: "ham"}
        MockAutoConfig.from_pretrained.return_value = mock_config

        MockAutoTokenizer = MagicMock()
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_import.return_value = (mock_ort, np, MockAutoConfig, MockAutoTokenizer)

        from tobira.backends.onnx import OnnxBackend

        backend = OnnxBackend(model_path="/tmp/model.onnx", model_name="test-model")
        result = backend.predict("buy now!!!")

        assert result.label == "spam"
        assert result.score > 0.5
        assert "spam" in result.labels
        assert "ham" in result.labels
        assert result.labels["spam"] + result.labels["ham"] == pytest.approx(1.0)

    @patch("tobira.backends.onnx.Path.exists", return_value=True)
    @patch("tobira.backends.onnx._import_deps")
    def test_loads_session(
        self, mock_import: MagicMock, _mock_exists: MagicMock
    ) -> None:
        mock_ort = MagicMock()
        MockAutoConfig = MagicMock()
        MockAutoTokenizer = MagicMock()
        mock_import.return_value = (mock_ort, np, MockAutoConfig, MockAutoTokenizer)

        from tobira.backends.onnx import OnnxBackend

        OnnxBackend(model_path="/tmp/model.onnx", model_name="test-model")
        mock_ort.InferenceSession.assert_called_once_with(
            "/tmp/model.onnx", providers=["CPUExecutionProvider"]
        )
        MockAutoTokenizer.from_pretrained.assert_called_once_with("test-model")
        MockAutoConfig.from_pretrained.assert_called_once_with("test-model")

    @patch("tobira.backends.onnx._import_deps")
    def test_missing_model_file_raises(self, mock_import: MagicMock) -> None:
        mock_ort = MagicMock()
        mock_import.return_value = (mock_ort, np, MagicMock(), MagicMock())

        from tobira.backends.onnx import OnnxBackend

        with pytest.raises(FileNotFoundError, match="ONNX model file not found"):
            OnnxBackend(model_path="/nonexistent/model.onnx")


class TestOnnxImportError:
    @patch.dict("sys.modules", {"onnxruntime": None})
    def test_missing_onnxruntime_raises(self) -> None:
        from tobira.backends.onnx import _import_deps

        with pytest.raises(ImportError, match="onnxruntime is required"):
            _import_deps()


class TestOnnxFactory:
    def test_onnx_missing_model_path_raises(self) -> None:
        with pytest.raises(KeyError, match="model_path"):
            create_backend({"type": "onnx"})

    @patch("tobira.backends.onnx.Path.exists", return_value=True)
    @patch("tobira.backends.onnx._import_deps")
    def test_onnx_creation(
        self, mock_import: MagicMock, _mock_exists: MagicMock
    ) -> None:
        mock_ort = MagicMock()
        MockAutoConfig = MagicMock()
        MockAutoTokenizer = MagicMock()
        mock_import.return_value = (mock_ort, np, MockAutoConfig, MockAutoTokenizer)

        from tobira.backends.onnx import OnnxBackend

        backend = create_backend(
            {"type": "onnx", "model_path": "/tmp/m.onnx", "model_name": "test-model"}
        )
        assert isinstance(backend, OnnxBackend)

    @patch("tobira.backends.onnx.Path.exists", return_value=True)
    @patch("tobira.backends.onnx._import_deps")
    def test_onnx_default_model_name(
        self, mock_import: MagicMock, _mock_exists: MagicMock
    ) -> None:
        mock_ort = MagicMock()
        MockAutoConfig = MagicMock()
        MockAutoTokenizer = MagicMock()
        mock_import.return_value = (mock_ort, np, MockAutoConfig, MockAutoTokenizer)

        create_backend({"type": "onnx", "model_path": "/tmp/m.onnx"})
        MockAutoTokenizer.from_pretrained.assert_called_once_with(
            "tohoku-nlp/bert-base-japanese-v3"
        )


def _make_mock_ollama_response(label: str = "spam", score: float = 0.95) -> MagicMock:
    """Create a mock httpx response for Ollama API."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "response": json.dumps({"label": label, "score": score}),
    }
    mock_response.raise_for_status = MagicMock()
    return mock_response


def _make_mock_llm_api_response(
    label: str = "spam", score: float = 0.95
) -> MagicMock:
    """Create a mock httpx response for OpenAI-compatible API."""
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"label": label, "score": score}),
                }
            }
        ],
    }
    mock_response.raise_for_status = MagicMock()
    return mock_response


class TestOllamaBackend:
    @patch("tobira.backends.ollama._import_httpx")
    def test_predict_spam(self, mock_import: MagicMock) -> None:
        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_ollama_response("spam", 0.95)
        mock_httpx.Client.return_value = mock_client
        mock_import.return_value = mock_httpx

        from tobira.backends.ollama import OllamaBackend

        backend = OllamaBackend(model="gemma2:2b")
        result = backend.predict("buy now!!!")

        assert result.label == "spam"
        assert result.score == pytest.approx(0.95)
        assert result.labels == {
            "spam": pytest.approx(0.95),
            "ham": pytest.approx(0.05),
        }

    @patch("tobira.backends.ollama._import_httpx")
    def test_predict_ham(self, mock_import: MagicMock) -> None:
        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_ollama_response("ham", 0.9)
        mock_httpx.Client.return_value = mock_client
        mock_import.return_value = mock_httpx

        from tobira.backends.ollama import OllamaBackend

        backend = OllamaBackend()
        result = backend.predict("hello friend")

        assert result.label == "ham"
        assert result.score == pytest.approx(0.9)
        assert result.labels["spam"] == pytest.approx(0.1)

    @patch("tobira.backends.ollama._import_httpx")
    def test_calls_correct_endpoint(self, mock_import: MagicMock) -> None:
        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_ollama_response()
        mock_httpx.Client.return_value = mock_client
        mock_import.return_value = mock_httpx

        from tobira.backends.ollama import OllamaBackend

        backend = OllamaBackend(
            model="llama3", base_url="http://myhost:11434"
        )
        backend.predict("test")

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "http://myhost:11434/api/generate"
        body = call_args[1]["json"]
        assert body["model"] == "llama3"
        assert body["format"] == "json"
        assert body["stream"] is False

    @patch("tobira.backends.ollama._import_httpx")
    def test_invalid_label_raises(self, mock_import: MagicMock) -> None:
        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_ollama_response("unknown", 0.5)
        mock_httpx.Client.return_value = mock_client
        mock_import.return_value = mock_httpx

        from tobira.backends.ollama import OllamaBackend

        backend = OllamaBackend()
        with pytest.raises(ValueError, match="unexpected label"):
            backend.predict("test")

    @patch("tobira.backends.ollama._import_httpx")
    def test_score_out_of_range_raises(self, mock_import: MagicMock) -> None:
        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_ollama_response("spam", 1.5)
        mock_httpx.Client.return_value = mock_client
        mock_import.return_value = mock_httpx

        from tobira.backends.ollama import OllamaBackend

        backend = OllamaBackend()
        with pytest.raises(ValueError, match="score out of range"):
            backend.predict("test")

    def test_implements_protocol(self) -> None:
        from tobira.backends.ollama import OllamaBackend

        assert issubclass(OllamaBackend, BackendProtocol)

    @patch("tobira.backends.ollama._import_httpx")
    def test_connection_error(self, mock_import: MagicMock) -> None:
        import httpx

        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        mock_httpx.Client.return_value = mock_client
        mock_import.return_value = mock_httpx

        from tobira.backends.ollama import OllamaBackend

        backend = OllamaBackend()
        with pytest.raises(Exception, match="connection refused"):
            backend.predict("test")


class TestLlmApiBackend:
    @patch("tobira.backends.llm_api._import_httpx")
    def test_predict_spam(self, mock_import: MagicMock) -> None:
        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_llm_api_response("spam", 0.92)
        mock_httpx.Client.return_value = mock_client
        mock_import.return_value = mock_httpx

        from tobira.backends.llm_api import LlmApiBackend

        backend = LlmApiBackend(api_key="test-key")
        result = backend.predict("buy now!!!")

        assert result.label == "spam"
        assert result.score == pytest.approx(0.92)
        assert result.labels == {
            "spam": pytest.approx(0.92),
            "ham": pytest.approx(0.08),
        }

    @patch("tobira.backends.llm_api._import_httpx")
    def test_predict_ham(self, mock_import: MagicMock) -> None:
        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_llm_api_response("ham", 0.88)
        mock_httpx.Client.return_value = mock_client
        mock_import.return_value = mock_httpx

        from tobira.backends.llm_api import LlmApiBackend

        backend = LlmApiBackend(api_key="test-key")
        result = backend.predict("hello")

        assert result.label == "ham"
        assert result.score == pytest.approx(0.88)

    @patch("tobira.backends.llm_api._import_httpx")
    def test_calls_correct_endpoint(self, mock_import: MagicMock) -> None:
        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_llm_api_response()
        mock_httpx.Client.return_value = mock_client
        mock_import.return_value = mock_httpx

        from tobira.backends.llm_api import LlmApiBackend

        backend = LlmApiBackend(
            model="gpt-4o",
            base_url="https://custom.api.com/v1",
            api_key="sk-test",
        )
        backend.predict("test")

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://custom.api.com/v1/chat/completions"
        assert call_args[1]["headers"]["Authorization"] == "Bearer sk-test"
        body = call_args[1]["json"]
        assert body["model"] == "gpt-4o"
        assert body["response_format"] == {"type": "json_object"}

    @patch("tobira.backends.llm_api._import_httpx")
    def test_missing_api_key_raises(self, mock_import: MagicMock) -> None:
        mock_import.return_value = MagicMock()

        from tobira.backends.llm_api import LlmApiBackend

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key is required"):
                LlmApiBackend()

    @patch("tobira.backends.llm_api._import_httpx")
    def test_api_key_from_env(self, mock_import: MagicMock) -> None:
        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_llm_api_response()
        mock_httpx.Client.return_value = mock_client
        mock_import.return_value = mock_httpx

        from tobira.backends.llm_api import LlmApiBackend

        with patch.dict("os.environ", {"TOBIRA_LLM_API_KEY": "env-key"}):
            backend = LlmApiBackend()
            backend.predict("test")

        call_args = mock_client.post.call_args
        assert call_args[1]["headers"]["Authorization"] == "Bearer env-key"

    @patch("tobira.backends.llm_api._import_httpx")
    def test_invalid_label_raises(self, mock_import: MagicMock) -> None:
        mock_httpx = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = _make_mock_llm_api_response("bad", 0.5)
        mock_httpx.Client.return_value = mock_client
        mock_import.return_value = mock_httpx

        from tobira.backends.llm_api import LlmApiBackend

        backend = LlmApiBackend(api_key="test-key")
        with pytest.raises(ValueError, match="unexpected label"):
            backend.predict("test")

    def test_implements_protocol(self) -> None:
        from tobira.backends.llm_api import LlmApiBackend

        assert issubclass(LlmApiBackend, BackendProtocol)


class TestOllamaImportError:
    @patch.dict("sys.modules", {"httpx": None})
    def test_missing_httpx_raises(self) -> None:
        from tobira.backends.ollama import _import_httpx

        with pytest.raises(ImportError, match="httpx is required"):
            _import_httpx()


class TestLlmApiImportError:
    @patch.dict("sys.modules", {"httpx": None})
    def test_missing_httpx_raises(self) -> None:
        from tobira.backends.llm_api import _import_httpx

        with pytest.raises(ImportError, match="httpx is required"):
            _import_httpx()


class TestOllamaFactory:
    @patch("tobira.backends.ollama._import_httpx")
    def test_ollama_creation(self, mock_import: MagicMock) -> None:
        mock_import.return_value = MagicMock()

        from tobira.backends.ollama import OllamaBackend

        backend = create_backend({"type": "ollama", "model": "gemma2:2b"})
        assert isinstance(backend, OllamaBackend)

    @patch("tobira.backends.ollama._import_httpx")
    def test_ollama_default_model(self, mock_import: MagicMock) -> None:
        mock_import.return_value = MagicMock()

        from tobira.backends.ollama import OllamaBackend

        backend = create_backend({"type": "ollama"})
        assert isinstance(backend, OllamaBackend)


class TestLlmApiFactory:
    @patch("tobira.backends.llm_api._import_httpx")
    def test_llm_api_creation(self, mock_import: MagicMock) -> None:
        mock_import.return_value = MagicMock()

        from tobira.backends.llm_api import LlmApiBackend

        backend = create_backend(
            {"type": "llm_api", "model": "gpt-4o", "api_key": "sk-test"}
        )
        assert isinstance(backend, LlmApiBackend)


class TestPrompts:
    def test_build_prompt(self) -> None:
        from tobira.backends.prompts import build_prompt

        result = build_prompt("hello world")
        assert "hello world" in result

    def test_system_prompt_exists(self) -> None:
        from tobira.backends.prompts import SPAM_CLASSIFICATION_SYSTEM

        assert "spam" in SPAM_CLASSIFICATION_SYSTEM.lower()
        assert "JSON" in SPAM_CLASSIFICATION_SYSTEM


class TestBackendsPublicAPI:
    def test_imports_from_package(self) -> None:
        from tobira.backends import BackendProtocol, PredictionResult, create_backend

        assert BackendProtocol is not None
        assert PredictionResult is not None
        assert create_backend is not None
