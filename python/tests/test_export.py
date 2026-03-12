"""Tests for tobira.core.export."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestExportOnnx:
    @patch("tobira.core.export._import_export_deps")
    def test_export_creates_file(
        self, mock_import: MagicMock, tmp_path: Path
    ) -> None:
        mock_torch = MagicMock()
        MockAutoModel = MagicMock()
        MockAutoTokenizer = MagicMock()

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        mock_tokenizer.__contains__ = lambda self, key: key in [
            "input_ids",
            "attention_mask",
        ]
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer

        mock_import.return_value = (mock_torch, MockAutoModel, MockAutoTokenizer)

        from tobira.core.export import export_onnx

        output = tmp_path / "model.onnx"
        result = export_onnx("test-model", output)

        assert result == output
        mock_torch.onnx.export.assert_called_once()
        MockAutoModel.from_pretrained.assert_called_once_with("test-model")
        MockAutoTokenizer.from_pretrained.assert_called_once_with("test-model")

    @patch("tobira.core.export._import_export_deps")
    def test_export_creates_parent_dirs(
        self, mock_import: MagicMock, tmp_path: Path
    ) -> None:
        mock_torch = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }
        mock_tokenizer.__contains__ = lambda self, key: key in [
            "input_ids",
            "attention_mask",
        ]
        MockAutoTokenizer = MagicMock()
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_import.return_value = (mock_torch, MagicMock(), MockAutoTokenizer)

        from tobira.core.export import export_onnx

        output = tmp_path / "subdir" / "model.onnx"
        export_onnx("test-model", output)

        assert output.parent.exists()

    @patch("tobira.core.export._import_export_deps")
    def test_export_includes_token_type_ids(
        self, mock_import: MagicMock, tmp_path: Path
    ) -> None:
        mock_torch = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
            "token_type_ids": MagicMock(),
        }
        mock_tokenizer.__contains__ = lambda self, key: key in [
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ]
        MockAutoTokenizer = MagicMock()
        MockAutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_import.return_value = (mock_torch, MagicMock(), MockAutoTokenizer)

        from tobira.core.export import export_onnx

        output = tmp_path / "model.onnx"
        export_onnx("test-model", output)

        export_call = mock_torch.onnx.export.call_args
        input_names = export_call.kwargs.get(
            "input_names", export_call[1].get("input_names")
        )
        assert "token_type_ids" in input_names


class TestQuantizeDynamic:
    @patch("tobira.core.export._import_quantization_deps")
    def test_quantize_default_output(
        self, mock_import: MagicMock, tmp_path: Path
    ) -> None:
        mock_qd = MagicMock()
        mock_quant_type = MagicMock()
        mock_import.return_value = (mock_qd, mock_quant_type)

        input_path = tmp_path / "model.onnx"
        input_path.touch()

        from tobira.core.export import quantize_dynamic

        result = quantize_dynamic(input_path)

        expected = tmp_path / "model_quantized.onnx"
        assert result == expected
        mock_qd.assert_called_once_with(
            model_input=str(input_path),
            model_output=str(expected),
            weight_type=mock_quant_type.QInt8,
        )

    @patch("tobira.core.export._import_quantization_deps")
    def test_quantize_custom_output(
        self, mock_import: MagicMock, tmp_path: Path
    ) -> None:
        mock_qd = MagicMock()
        mock_quant_type = MagicMock()
        mock_import.return_value = (mock_qd, mock_quant_type)

        input_path = tmp_path / "model.onnx"
        input_path.touch()
        output_path = tmp_path / "custom.onnx"

        from tobira.core.export import quantize_dynamic

        result = quantize_dynamic(input_path, output_path)

        assert result == output_path
        mock_qd.assert_called_once()

    @patch("tobira.core.export._import_quantization_deps")
    def test_quantize_missing_input_raises(
        self, mock_import: MagicMock, tmp_path: Path
    ) -> None:
        mock_import.return_value = (MagicMock(), MagicMock())

        from tobira.core.export import quantize_dynamic

        with pytest.raises(FileNotFoundError, match="ONNX model not found"):
            quantize_dynamic(tmp_path / "nonexistent.onnx")


class TestExportImportError:
    @patch.dict("sys.modules", {"torch": None})
    def test_missing_torch_raises(self) -> None:
        from tobira.core.export import _import_export_deps

        with pytest.raises(ImportError, match="torch and transformers are required"):
            _import_export_deps()

    @patch.dict("sys.modules", {"onnxruntime": None, "onnxruntime.quantization": None})
    def test_missing_onnxruntime_raises(self) -> None:
        from tobira.core.export import _import_quantization_deps

        with pytest.raises(ImportError, match="onnxruntime is required"):
            _import_quantization_deps()


class TestCorePublicAPI:
    def test_imports_from_package(self) -> None:
        from tobira.core import export_onnx, quantize_dynamic

        assert export_onnx is not None
        assert quantize_dynamic is not None
