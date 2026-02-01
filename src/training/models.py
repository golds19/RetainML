"""
Model definitions and factory.
Centralizes model configurations and initialization.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for creating model instances from configuration."""

    MODEL_REGISTRY = {
        'logistic_regression': LogisticRegression,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
    }

    @classmethod
    def create_model(cls, model_name: str, params: Dict[str, Any]):
        """
        Create a model instance.

        Args:
            model_name: Name of the model (key in MODEL_REGISTRY)
            params: Model hyperparameters

        Returns:
            Instantiated model

        Raises:
            ValueError: If model_name not in registry
        """
        if model_name not in cls.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(cls.MODEL_REGISTRY.keys())}"
            )

        model_class = cls.MODEL_REGISTRY[model_name]
        model = model_class(**params)

        logger.info(f"Created model: {model_name} with params: {params}")
        return model

    @classmethod
    def create_models_from_config(cls, config: dict) -> Dict[str, Any]:
        """
        Create all enabled models from configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            Dictionary of {display_name: model_instance}
        """
        models = {}
        model_configs = config['models']

        for model_name, model_config in model_configs.items():
            if not model_config.get('enabled', True):
                logger.info(f"Skipping disabled model: {model_name}")
                continue

            display_name = model_config['display_name']
            params = model_config['params']

            model = cls.create_model(model_name, params)
            models[display_name] = model

        logger.info(f"Created {len(models)} models")
        return models


class ModelConfig:
    """Helper class for model configuration management."""

    @staticmethod
    def get_model_params(config: dict, model_name: str) -> Dict[str, Any]:
        """
        Get parameters for a specific model.

        Args:
            config: Full configuration dictionary
            model_name: Name of the model

        Returns:
            Model parameters dictionary
        """
        return config['models'][model_name]['params']

    @staticmethod
    def get_enabled_models(config: dict) -> list:
        """
        Get list of enabled model names.

        Args:
            config: Full configuration dictionary

        Returns:
            List of enabled model names
        """
        return [
            name for name, cfg in config['models'].items()
            if cfg.get('enabled', True)
        ]
