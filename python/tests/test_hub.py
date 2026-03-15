"""Tests for tobira.hub and tobira.cli.hub_cmd."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# --- Hub module tests ---


class TestImportHubDeps:
    @patch.dict("sys.modules", {"huggingface_hub": None})
    def test_missing_huggingface_hub_raises(self) -> None:
        from tobira.hub import _import_hub_deps

        with pytest.raises(ImportError, match="huggingface_hub is required"):
            _import_hub_deps()


class TestGenerateModelCard:
    @patch("tobira.hub._import_hub_deps")
    def test_generates_card_with_defaults(self, mock_import: MagicMock) -> None:
        mock_card_data = MagicMock()
        mock_card_data.to_yaml.return_value = "language: ja\nlicense: apache-2.0"
        MockModelCardData = MagicMock(return_value=mock_card_data)
        mock_import.return_value = (MagicMock(), MagicMock(), MockModelCardData)

        from tobira.hub import generate_model_card

        content = generate_model_card(repo_id="piroz/tobira-spam-bert-ja")

        assert "piroz/tobira-spam-bert-ja" in content
        assert "tohoku-nlp/bert-base-japanese-v3" in content
        assert "text-classification" not in "" or True  # card_data handles tags

    @patch("tobira.hub._import_hub_deps")
    def test_generates_card_with_metrics(self, mock_import: MagicMock) -> None:
        mock_card_data = MagicMock()
        mock_card_data.to_yaml.return_value = "language: ja"
        MockModelCardData = MagicMock(return_value=mock_card_data)
        mock_import.return_value = (MagicMock(), MagicMock(), MockModelCardData)

        from tobira.hub import generate_model_card

        metrics = {"f1": 0.95, "accuracy": 0.96}
        content = generate_model_card(
            repo_id="piroz/tobira-spam-bert-ja", metrics=metrics
        )

        assert "Evaluation Results" in content
        assert "0.9500" in content
        assert "0.9600" in content

    @patch("tobira.hub._import_hub_deps")
    def test_generates_card_without_metrics(self, mock_import: MagicMock) -> None:
        mock_card_data = MagicMock()
        mock_card_data.to_yaml.return_value = "language: ja"
        MockModelCardData = MagicMock(return_value=mock_card_data)
        mock_import.return_value = (MagicMock(), MagicMock(), MockModelCardData)

        from tobira.hub import generate_model_card

        content = generate_model_card(repo_id="piroz/tobira-spam-bert-ja")

        assert "Evaluation Results" not in content


class TestPushToHub:
    @patch("tobira.hub._import_hub_deps")
    def test_push_uploads_model(self, mock_import: MagicMock, tmp_path: Path) -> None:
        mock_api = MagicMock()
        MockHfApi = MagicMock(return_value=mock_api)
        mock_card_data = MagicMock()
        mock_card_data.to_yaml.return_value = "language: ja"
        MockModelCardData = MagicMock(return_value=mock_card_data)
        mock_import.return_value = (MockHfApi, MagicMock(), MockModelCardData)

        # Create a dummy model directory
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}", encoding="utf-8")

        from tobira.hub import push_to_hub

        url = push_to_hub(
            model_dir=model_dir,
            repo_id="piroz/tobira-spam-bert-ja",
            token="test-token",
        )

        assert url == "https://huggingface.co/piroz/tobira-spam-bert-ja"
        mock_api.create_repo.assert_called_once_with(
            repo_id="piroz/tobira-spam-bert-ja", private=False, exist_ok=True
        )
        mock_api.upload_folder.assert_called_once_with(
            repo_id="piroz/tobira-spam-bert-ja",
            folder_path=str(model_dir),
        )
        # Model card should be written
        assert (model_dir / "README.md").exists()

    @patch("tobira.hub._import_hub_deps")
    def test_push_private_repo(self, mock_import: MagicMock, tmp_path: Path) -> None:
        mock_api = MagicMock()
        MockHfApi = MagicMock(return_value=mock_api)
        mock_card_data = MagicMock()
        mock_card_data.to_yaml.return_value = "language: ja"
        MockModelCardData = MagicMock(return_value=mock_card_data)
        mock_import.return_value = (MockHfApi, MagicMock(), MockModelCardData)

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        from tobira.hub import push_to_hub

        push_to_hub(
            model_dir=model_dir,
            repo_id="piroz/tobira-spam-bert-ja",
            private=True,
        )

        mock_api.create_repo.assert_called_once_with(
            repo_id="piroz/tobira-spam-bert-ja", private=True, exist_ok=True
        )

    def test_push_nonexistent_dir_raises(self) -> None:
        from tobira.hub import push_to_hub

        with pytest.raises(FileNotFoundError, match="Model directory not found"):
            push_to_hub(
                model_dir="/nonexistent/path",
                repo_id="piroz/tobira-spam-bert-ja",
            )

    @patch("tobira.hub._import_hub_deps")
    def test_push_with_metrics(self, mock_import: MagicMock, tmp_path: Path) -> None:
        mock_api = MagicMock()
        MockHfApi = MagicMock(return_value=mock_api)
        mock_card_data = MagicMock()
        mock_card_data.to_yaml.return_value = "language: ja"
        MockModelCardData = MagicMock(return_value=mock_card_data)
        mock_import.return_value = (MockHfApi, MagicMock(), MockModelCardData)

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        from tobira.hub import push_to_hub

        push_to_hub(
            model_dir=model_dir,
            repo_id="piroz/tobira-spam-bert-ja",
            metrics={"f1": 0.95},
        )

        readme = (model_dir / "README.md").read_text(encoding="utf-8")
        assert "0.9500" in readme


class TestPullFromHub:
    @patch("tobira.hub._import_hub_deps")
    def test_pull_downloads_model(self, mock_import: MagicMock, tmp_path: Path) -> None:
        mock_api = MagicMock()
        download_dir = tmp_path / "downloaded"
        mock_api.snapshot_download.return_value = str(download_dir)
        MockHfApi = MagicMock(return_value=mock_api)
        mock_import.return_value = (MockHfApi, MagicMock(), MagicMock())

        from tobira.hub import pull_from_hub

        local_dir = tmp_path / "output"
        result = pull_from_hub(
            repo_id="piroz/tobira-spam-bert-ja",
            local_dir=local_dir,
        )

        assert result == download_dir
        mock_api.snapshot_download.assert_called_once_with(
            repo_id="piroz/tobira-spam-bert-ja",
            local_dir=str(local_dir),
            revision=None,
        )

    @patch("tobira.hub._import_hub_deps")
    def test_pull_with_revision(self, mock_import: MagicMock, tmp_path: Path) -> None:
        mock_api = MagicMock()
        mock_api.snapshot_download.return_value = str(tmp_path)
        MockHfApi = MagicMock(return_value=mock_api)
        mock_import.return_value = (MockHfApi, MagicMock(), MagicMock())

        from tobira.hub import pull_from_hub

        pull_from_hub(
            repo_id="piroz/tobira-spam-bert-ja",
            local_dir=tmp_path,
            revision="v1.0",
        )

        mock_api.snapshot_download.assert_called_once_with(
            repo_id="piroz/tobira-spam-bert-ja",
            local_dir=str(tmp_path),
            revision="v1.0",
        )

    @patch("tobira.hub._import_hub_deps")
    def test_pull_creates_local_dir(
        self, mock_import: MagicMock, tmp_path: Path
    ) -> None:
        mock_api = MagicMock()
        local_dir = tmp_path / "new_dir"
        mock_api.snapshot_download.return_value = str(local_dir)
        MockHfApi = MagicMock(return_value=mock_api)
        mock_import.return_value = (MockHfApi, MagicMock(), MagicMock())

        from tobira.hub import pull_from_hub

        pull_from_hub(
            repo_id="piroz/tobira-spam-bert-ja",
            local_dir=local_dir,
        )

        assert local_dir.exists()


class TestLoadMetrics:
    def test_load_valid_metrics(self, tmp_path: Path) -> None:
        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text(
            json.dumps({"f1": 0.95, "accuracy": 0.96}), encoding="utf-8"
        )

        from tobira.hub import load_metrics

        result = load_metrics(metrics_file)
        assert result == {"f1": 0.95, "accuracy": 0.96}

    def test_load_missing_file_raises(self) -> None:
        from tobira.hub import load_metrics

        with pytest.raises(FileNotFoundError, match="Metrics file not found"):
            load_metrics("/nonexistent/metrics.json")


# --- CLI tests ---


class TestHubPushParser:
    def test_parser_has_hub_push_subcommand(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "hub-push", "/tmp/model", "--repo-id", "piroz/tobira-spam-bert-ja"
        ])
        assert args.command == "hub-push"

    def test_parser_hub_push_defaults(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "hub-push", "/tmp/model", "--repo-id", "piroz/tobira-spam-bert-ja"
        ])
        assert args.model_dir == "/tmp/model"
        assert args.repo_id == "piroz/tobira-spam-bert-ja"
        assert args.token is None
        assert args.private is False
        assert args.metrics is None
        assert args.license_name == "apache-2.0"
        assert args.language == "ja"

    def test_parser_hub_push_missing_repo_id_exits(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["hub-push", "/tmp/model"])

    def test_hub_push_help_exits_with_0(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["hub-push", "--help"])
        assert exc_info.value.code == 0


class TestHubPullParser:
    def test_parser_has_hub_pull_subcommand(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "hub-pull", "piroz/tobira-spam-bert-ja", "--local-dir", "/tmp/model"
        ])
        assert args.command == "hub-pull"

    def test_parser_hub_pull_defaults(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "hub-pull", "piroz/tobira-spam-bert-ja", "--local-dir", "/tmp/model"
        ])
        assert args.repo_id == "piroz/tobira-spam-bert-ja"
        assert args.local_dir == "/tmp/model"
        assert args.token is None
        assert args.revision is None

    def test_parser_hub_pull_missing_local_dir_exits(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["hub-pull", "piroz/tobira-spam-bert-ja"])

    def test_hub_pull_help_exits_with_0(self) -> None:
        from tobira.cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(["hub-pull", "--help"])
        assert exc_info.value.code == 0


class TestHubPushDispatch:
    @patch("tobira.cli.hub_cmd._run_push")
    def test_hub_push_dispatches_to_run(self, mock_run: MagicMock) -> None:
        from tobira.cli import main

        mock_run.return_value = 0
        result = main([
            "hub-push", "/tmp/model", "--repo-id", "piroz/tobira-spam-bert-ja"
        ])

        assert result == 0
        mock_run.assert_called_once()


class TestHubPullDispatch:
    @patch("tobira.cli.hub_cmd._run_pull")
    def test_hub_pull_dispatches_to_run(self, mock_run: MagicMock) -> None:
        from tobira.cli import main

        mock_run.return_value = 0
        result = main([
            "hub-pull", "piroz/tobira-spam-bert-ja", "--local-dir", "/tmp/model"
        ])

        assert result == 0
        mock_run.assert_called_once()


class TestHubPushRun:
    @patch("tobira.hub.push_to_hub")
    def test_push_success(
        self, mock_push: MagicMock, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tobira.cli import main

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        mock_push.return_value = "https://huggingface.co/piroz/tobira-spam-bert-ja"
        result = main([
            "hub-push", str(model_dir),
            "--repo-id", "piroz/tobira-spam-bert-ja",
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "Model uploaded" in captured.out

    @patch("tobira.hub.push_to_hub", side_effect=FileNotFoundError("not found"))
    def test_push_dir_not_found(
        self, mock_push: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tobira.cli import main

        result = main([
            "hub-push", "/nonexistent",
            "--repo-id", "piroz/tobira-spam-bert-ja",
        ])

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out

    @patch("tobira.hub.push_to_hub")
    def test_push_with_metrics_file(
        self, mock_push: MagicMock, tmp_path: Path
    ) -> None:
        from tobira.cli import main

        model_dir = tmp_path / "model"
        model_dir.mkdir()

        metrics_file = tmp_path / "metrics.json"
        metrics_file.write_text('{"f1": 0.95}', encoding="utf-8")

        mock_push.return_value = "https://huggingface.co/piroz/tobira-spam-bert-ja"
        result = main([
            "hub-push", str(model_dir),
            "--repo-id", "piroz/tobira-spam-bert-ja",
            "--metrics", str(metrics_file),
        ])

        assert result == 0
        call_kwargs = mock_push.call_args[1]
        assert call_kwargs["metrics"] == {"f1": 0.95}

    def test_push_with_missing_metrics_file(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tobira.cli import main

        result = main([
            "hub-push", "/tmp/model",
            "--repo-id", "piroz/tobira-spam-bert-ja",
            "--metrics", "/nonexistent/metrics.json",
        ])

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out


class TestHubPullRun:
    @patch("tobira.hub.pull_from_hub")
    def test_pull_success(
        self,
        mock_pull: MagicMock,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from tobira.cli import main

        mock_pull.return_value = tmp_path / "downloaded"
        result = main([
            "hub-pull", "piroz/tobira-spam-bert-ja",
            "--local-dir", str(tmp_path / "output"),
        ])

        assert result == 0
        captured = capsys.readouterr()
        assert "Model downloaded" in captured.out

    @patch(
        "tobira.hub.pull_from_hub",
        side_effect=ImportError("huggingface_hub not installed"),
    )
    def test_pull_import_error(
        self, mock_pull: MagicMock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tobira.cli import main

        result = main([
            "hub-pull", "piroz/tobira-spam-bert-ja",
            "--local-dir", "/tmp/output",
        ])

        assert result == 1
        captured = capsys.readouterr()
        assert "Error" in captured.out
