"""Tests for ConfigLoader."""

import sys
from pathlib import Path
import yaml
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from utils.config_loader import ConfigLoader


class TestConfigLoaderInit:

    def test_init_with_valid_directory(self, tmp_path):
        loader = ConfigLoader(str(tmp_path))
        assert loader.config_dir == tmp_path

    def test_init_with_missing_directory(self):
        with pytest.raises(FileNotFoundError):
            ConfigLoader("nonexistent_dir_xyz")


class TestLoadConfig:

    def test_load_config_success(self, tmp_path):
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml.dump({"key": "value"}))

        loader = ConfigLoader(str(tmp_path))
        result = loader.load_config("test.yaml")
        assert result == {"key": "value"}

    def test_load_config_missing_file(self, tmp_path):
        loader = ConfigLoader(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            loader.load_config("nonexistent.yaml")


class TestLoadAllConfigs:

    def test_merges_multiple_files(self, tmp_path):
        (tmp_path / "data_config.yaml").write_text(yaml.dump({"data": {"path": "x"}}))
        (tmp_path / "model_config.yaml").write_text(yaml.dump({"models": {"lr": {}}}))

        loader = ConfigLoader(str(tmp_path))
        result = loader.load_all_configs()
        assert "data" in result
        assert "models" in result

    def test_skips_missing_files(self, tmp_path):
        (tmp_path / "data_config.yaml").write_text(yaml.dump({"data": "test"}))

        loader = ConfigLoader(str(tmp_path))
        result = loader.load_all_configs()
        assert "data" in result

    def test_empty_dir_returns_empty(self, tmp_path):
        loader = ConfigLoader(str(tmp_path))
        result = loader.load_all_configs()
        assert result == {}


class TestSaveConfig:

    def test_save_and_read_back(self, tmp_path):
        config = {"key": "value", "nested": {"a": 1}}
        output = tmp_path / "out.yaml"

        ConfigLoader.save_config(config, str(output))

        with open(output) as f:
            loaded = yaml.safe_load(f)
        assert loaded == config

    def test_creates_parent_dirs(self, tmp_path):
        output = tmp_path / "nested" / "dir" / "config.yaml"
        ConfigLoader.save_config({"k": "v"}, str(output))
        assert output.exists()
