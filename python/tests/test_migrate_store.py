"""Tests for tobira migrate-store command."""

from __future__ import annotations

import json
from pathlib import Path


class TestMigrateStoreCommand:
    def _write_jsonl(self, path: Path, records: list[dict[str, object]]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def _write_config(self, path: Path, store_config: dict[str, object]) -> None:
        import sys

        if sys.version_info >= (3, 11):
            # Write TOML manually for simplicity
            pass

        lines = ["[store]\n"]
        for k, v in store_config.items():
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"\n')
            else:
                lines.append(f"{k} = {v}\n")
        path.write_text("".join(lines), encoding="utf-8")

    def test_dry_run(self, tmp_path: Path) -> None:
        from tobira.cli import main

        source = tmp_path / "predictions.jsonl"
        self._write_jsonl(source, [
            {"label": "spam", "score": 0.9},
            {"label": "ham", "score": 0.1},
        ])
        config_file = tmp_path / "tobira.toml"
        self._write_config(config_file, {
            "type": "jsonl",
            "base_dir": str(tmp_path / "target"),
        })

        exit_code = main([
            "migrate-store", str(source), "predictions",
            "--config", str(config_file), "--dry-run",
        ])
        assert exit_code == 0
        # Target should not exist in dry run
        assert not (tmp_path / "target" / "predictions.jsonl").exists()

    def test_migrate_to_jsonl(self, tmp_path: Path) -> None:
        from tobira.cli import main

        source = tmp_path / "old.jsonl"
        self._write_jsonl(source, [
            {"label": "spam", "score": 0.9},
            {"label": "ham", "score": 0.1},
        ])
        target_dir = tmp_path / "target"
        config_file = tmp_path / "tobira.toml"
        self._write_config(config_file, {
            "type": "jsonl",
            "base_dir": str(target_dir),
        })

        exit_code = main([
            "migrate-store", str(source), "predictions",
            "--config", str(config_file),
        ])
        assert exit_code == 0

        from tobira.monitoring.store import JsonlStore

        store = JsonlStore(base_dir=str(target_dir))
        records = store.read_all("predictions")
        assert len(records) == 2
        assert records[0]["label"] == "spam"

    def test_missing_source_file(self, tmp_path: Path) -> None:
        from tobira.cli import main

        config_file = tmp_path / "tobira.toml"
        self._write_config(config_file, {"type": "jsonl"})

        exit_code = main([
            "migrate-store", str(tmp_path / "missing.jsonl"), "logs",
            "--config", str(config_file),
        ])
        assert exit_code == 1

    def test_missing_config_file(self, tmp_path: Path) -> None:
        from tobira.cli import main

        source = tmp_path / "data.jsonl"
        self._write_jsonl(source, [{"a": 1}])

        exit_code = main([
            "migrate-store", str(source), "logs",
            "--config", str(tmp_path / "missing.toml"),
        ])
        assert exit_code == 1

    def test_missing_store_section(self, tmp_path: Path) -> None:
        from tobira.cli import main

        source = tmp_path / "data.jsonl"
        self._write_jsonl(source, [{"a": 1}])

        config_file = tmp_path / "tobira.toml"
        config_file.write_text("[backend]\ntype = 'fasttext'\n")

        exit_code = main([
            "migrate-store", str(source), "logs",
            "--config", str(config_file),
        ])
        assert exit_code == 1

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        from tobira.cli import main

        source = tmp_path / "data.jsonl"
        source.write_text(
            '{"label": "spam"}\n'
            'not valid json\n'
            '{"label": "ham"}\n',
            encoding="utf-8",
        )
        target_dir = tmp_path / "target"
        config_file = tmp_path / "tobira.toml"
        self._write_config(config_file, {
            "type": "jsonl",
            "base_dir": str(target_dir),
        })

        exit_code = main([
            "migrate-store", str(source), "logs",
            "--config", str(config_file),
        ])
        assert exit_code == 0

        from tobira.monitoring.store import JsonlStore

        store = JsonlStore(base_dir=str(target_dir))
        records = store.read_all("logs")
        assert len(records) == 2
