"""Tests for tobira.evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tobira.evaluation.metrics import (
    ConfusionMatrix,
    MetricsResult,
    compute_metrics,
    confusion_matrix,
)
from tobira.evaluation.plot import pr_curve_data
from tobira.evaluation.report import to_json, to_text
from tobira.evaluation.threshold import (
    ThresholdResult,
    find_best_threshold_f1,
    find_best_threshold_fpr,
)


class TestConfusionMatrix:
    def test_perfect_predictions(self) -> None:
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        cm = confusion_matrix(y_true, y_pred)
        assert cm == ConfusionMatrix(tp=2, fp=0, fn=0, tn=2)

    def test_all_wrong(self) -> None:
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 1, 1]
        cm = confusion_matrix(y_true, y_pred)
        assert cm == ConfusionMatrix(tp=0, fp=2, fn=2, tn=0)

    def test_mixed(self) -> None:
        y_true = [1, 0, 1, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        cm = confusion_matrix(y_true, y_pred)
        assert cm == ConfusionMatrix(tp=2, fp=1, fn=1, tn=1)

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            confusion_matrix([1, 0], [1])

    def test_empty_inputs(self) -> None:
        with pytest.raises(ValueError, match="not be empty"):
            confusion_matrix([], [])

    def test_frozen(self) -> None:
        cm = ConfusionMatrix(tp=1, fp=0, fn=0, tn=1)
        with pytest.raises(AttributeError):
            cm.tp = 5  # type: ignore[misc]


class TestComputeMetrics:
    def test_perfect_scores(self) -> None:
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]
        m = compute_metrics(y_true, y_pred)
        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_all_wrong(self) -> None:
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 1, 1]
        m = compute_metrics(y_true, y_pred)
        assert m.accuracy == 0.0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_known_values(self) -> None:
        # 3 TP, 1 FP, 1 FN, 1 TN
        y_true = [1, 1, 1, 0, 0, 1]
        y_pred = [1, 1, 1, 1, 0, 0]
        m = compute_metrics(y_true, y_pred)
        assert m.accuracy == pytest.approx(4 / 6)
        assert m.precision == pytest.approx(3 / 4)
        assert m.recall == pytest.approx(3 / 4)
        assert m.f1 == pytest.approx(3 / 4)

    def test_no_positives_predicted(self) -> None:
        y_true = [1, 1, 0, 0]
        y_pred = [0, 0, 0, 0]
        m = compute_metrics(y_true, y_pred)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_frozen(self) -> None:
        m = MetricsResult(
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            confusion_matrix=ConfusionMatrix(tp=1, fp=0, fn=0, tn=1),
        )
        with pytest.raises(AttributeError):
            m.accuracy = 0.5  # type: ignore[misc]


class TestReport:
    def _make_result(self) -> MetricsResult:
        return MetricsResult(
            accuracy=0.85,
            precision=0.9,
            recall=0.8,
            f1=0.8471,
            confusion_matrix=ConfusionMatrix(tp=36, fp=4, fn=9, tn=51),
        )

    def test_text_contains_metrics(self) -> None:
        text = to_text(self._make_result())
        assert "Accuracy:" in text
        assert "Precision:" in text
        assert "Recall:" in text
        assert "F1:" in text
        assert "TP=36" in text
        assert "TN=51" in text

    def test_json_valid(self) -> None:
        result = self._make_result()
        data = json.loads(to_json(result))
        assert data["accuracy"] == pytest.approx(0.85)
        assert data["precision"] == pytest.approx(0.9)
        assert data["confusion_matrix"]["tp"] == 36
        assert data["confusion_matrix"]["tn"] == 51


class TestThreshold:
    def _make_data(self) -> tuple[list[int], list[float]]:
        # Clear separation: positives have high scores, negatives low
        y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.85, 0.75, 0.1, 0.2, 0.15, 0.3, 0.05]
        return y_true, scores

    def test_find_best_f1(self) -> None:
        y_true, scores = self._make_data()
        result = find_best_threshold_f1(y_true, scores)
        assert result.f1 == pytest.approx(1.0)
        assert 0.3 < result.threshold < 0.7

    def test_find_best_fpr(self) -> None:
        y_true, scores = self._make_data()
        result = find_best_threshold_fpr(y_true, scores, max_fpr=0.0)
        assert result.f1 > 0.0
        # With max_fpr=0, no false positives allowed
        assert result.precision == pytest.approx(1.0)

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            find_best_threshold_f1([1, 0], [0.5])

    def test_empty_inputs(self) -> None:
        with pytest.raises(ValueError, match="not be empty"):
            find_best_threshold_f1([], [])


class TestPRCurveData:
    def test_returns_lists(self) -> None:
        y_true = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.3, 0.1]
        precisions, recalls, thresholds = pr_curve_data(y_true, scores, steps=10)
        assert len(precisions) == 11
        assert len(recalls) == 11
        assert len(thresholds) == 11

    def test_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            pr_curve_data([1, 0], [0.5], steps=10)

    def test_empty_inputs(self) -> None:
        with pytest.raises(ValueError, match="not be empty"):
            pr_curve_data([], [], steps=10)

    def test_values_in_range(self) -> None:
        y_true = [1, 1, 0, 0]
        scores = [0.9, 0.8, 0.3, 0.1]
        precisions, recalls, _ = pr_curve_data(y_true, scores, steps=10)
        for p in precisions:
            assert 0.0 <= p <= 1.0
        for r in recalls:
            assert 0.0 <= r <= 1.0


class TestComputeMetricsEdgeCases:
    def test_all_negatives(self) -> None:
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]
        m = compute_metrics(y_true, y_pred)
        assert m.accuracy == 1.0
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_all_positives(self) -> None:
        y_true = [1, 1, 1, 1]
        y_pred = [1, 1, 1, 1]
        m = compute_metrics(y_true, y_pred)
        assert m.accuracy == 1.0
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0

    def test_metrics_result_has_confusion_matrix(self) -> None:
        y_true = [1, 0, 1, 0]
        y_pred = [1, 0, 0, 1]
        m = compute_metrics(y_true, y_pred)
        assert m.confusion_matrix.tp == 1
        assert m.confusion_matrix.fp == 1
        assert m.confusion_matrix.fn == 1
        assert m.confusion_matrix.tn == 1


class TestThresholdEdgeCases:
    def test_threshold_result_frozen(self) -> None:
        result = ThresholdResult(threshold=0.5, f1=0.8, precision=0.9, recall=0.7)
        with pytest.raises(AttributeError):
            result.threshold = 0.6  # type: ignore[misc]

    def test_fpr_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            find_best_threshold_fpr([1, 0], [0.5])

    def test_fpr_empty_inputs(self) -> None:
        with pytest.raises(ValueError, match="not be empty"):
            find_best_threshold_fpr([], [])

    def test_fpr_all_exceed_constraint(self) -> None:
        # All positives have low scores, all negatives have high scores
        # Any threshold will produce false positives or no true positives
        y_true = [0, 0, 0, 0, 0]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        result = find_best_threshold_fpr(y_true, scores, max_fpr=0.0)
        assert result.f1 == 0.0
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_fpr_no_negatives(self) -> None:
        y_true = [1, 1, 1]
        scores = [0.9, 0.8, 0.7]
        result = find_best_threshold_fpr(y_true, scores, max_fpr=0.01)
        assert result.f1 > 0.0


class TestReportEdgeCases:
    def test_text_report_header(self) -> None:
        result = MetricsResult(
            accuracy=0.5,
            precision=0.5,
            recall=0.5,
            f1=0.5,
            confusion_matrix=ConfusionMatrix(tp=1, fp=1, fn=1, tn=1),
        )
        text = to_text(result)
        assert "=== Evaluation Report ===" in text
        assert "FP=1" in text
        assert "FN=1" in text

    def test_json_all_fields_present(self) -> None:
        result = MetricsResult(
            accuracy=0.75,
            precision=0.8,
            recall=0.7,
            f1=0.746,
            confusion_matrix=ConfusionMatrix(tp=7, fp=2, fn=3, tn=8),
        )
        data = json.loads(to_json(result))
        expected = {"accuracy", "precision", "recall", "f1", "confusion_matrix"}
        assert set(data.keys()) == expected
        assert set(data["confusion_matrix"].keys()) == {"tp", "fp", "fn", "tn"}


class TestPlotPRCurve:
    @patch("tobira.evaluation.plot._import_matplotlib")
    def test_plot_pr_curve_saves_to_file(
        self, mock_mpl: MagicMock, tmp_path: Path
    ) -> None:
        from tobira.evaluation.plot import plot_pr_curve

        mock_plt = MagicMock()
        mock_mpl.return_value = mock_plt
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        output = tmp_path / "pr.png"  # type: ignore[operator]
        plot_pr_curve([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.1], output_path=str(output))

        mock_fig.savefig.assert_called_once()
        mock_plt.close.assert_called_once_with(mock_fig)

    @patch("tobira.evaluation.plot._import_matplotlib")
    def test_plot_pr_curve_shows_plot(self, mock_mpl: MagicMock) -> None:
        from tobira.evaluation.plot import plot_pr_curve

        mock_plt = MagicMock()
        mock_mpl.return_value = mock_plt
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        plot_pr_curve([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.1])

        mock_plt.show.assert_called_once()


class TestPlotImportError:
    @patch.dict("sys.modules", {"matplotlib": None, "matplotlib.pyplot": None})
    def test_missing_matplotlib_raises(self) -> None:
        from tobira.evaluation.plot import _import_matplotlib

        with pytest.raises(ImportError, match="matplotlib is required"):
            _import_matplotlib()


class TestEvaluationPublicAPI:
    def test_imports_from_package(self) -> None:
        from tobira.evaluation import (
            ConfusionMatrix,
            MetricsResult,
            ThresholdResult,
            compute_metrics,
            confusion_matrix,
            find_best_threshold_f1,
            find_best_threshold_fpr,
            plot_pr_curve,
            pr_curve_data,
            to_json,
            to_text,
        )

        assert ConfusionMatrix is not None
        assert MetricsResult is not None
        assert ThresholdResult is not None
        assert compute_metrics is not None
        assert confusion_matrix is not None
        assert find_best_threshold_f1 is not None
        assert find_best_threshold_fpr is not None
        assert plot_pr_curve is not None
        assert pr_curve_data is not None
        assert to_json is not None
        assert to_text is not None
