"""
Configuration loading utilities.
Handles loading and merging YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration files."""

    def __init__(self, config_dir: str = "config"):
        """
        Initialize ConfigLoader.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)

        if not self.config_dir.exists():
            raise FileNotFoundError(f"Config directory not found: {config_dir}")

    def load_config(self, config_file: str) -> Dict[str, Any]:
        """
        Load a single configuration file.

        Args:
            config_file: Name of the config file (e.g., 'data_config.yaml')

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = self.config_dir / config_file

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info(f"Loading configuration from {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    def load_all_configs(self) -> Dict[str, Any]:
        """
        Load all configuration files and merge them.

        Returns:
            Merged configuration dictionary
        """
        logger.info("Loading all configuration files")

        config_files = [
            'data_config.yaml',
            'feature_config.yaml',
            'model_config.yaml',
            'pipeline_config.yaml',
            'mlflow_config.yaml',
            'validation_config.yaml',
        ]

        merged_config = {}

        for config_file in config_files:
            try:
                config = self.load_config(config_file)
                merged_config.update(config)
            except FileNotFoundError:
                logger.warning(f"Config file not found: {config_file}, skipping")

        logger.info("All configurations loaded successfully")
        return merged_config

    @staticmethod
    def save_config(config: Dict[str, Any], output_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        logger.info(f"Configuration saved to {output_path}")
