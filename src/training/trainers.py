"""
Model training logic.
Handles training, cross-validation, and model management.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and manage multiple models."""

    def __init__(self, config: dict):
        """
        Initialize ModelTrainer.

        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.models = {}
        self.training_results = {}

    def train_single_model(
        self,
        model_name: str,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Any:
        """
        Train a single model.

        Args:
            model_name: Name of the model
            model: Model instance
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained model
        """
        logger.info(f"\n{'-'*80}")
        logger.info(f"Training {model_name}...")

        model.fit(X_train, y_train)
        self.models[model_name] = model

        logger.info(f"{model_name} training completed")
        return model

    def train_all_models(
        self,
        models: Dict[str, Any],
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Dict[str, Any]:
        """
        Train all provided models.

        Args:
            models: Dictionary of {model_name: model_instance}
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary of trained models
        """
        logger.info("\n" + "="*80)
        logger.info("TRAINING BASELINE MODELS")
        logger.info("="*80)

        for model_name, model in models.items():
            self.train_single_model(model_name, model, X_train, y_train)

        logger.info("\nAll models trained successfully")
        return self.models

    def cross_validate_model(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        scoring: str = 'f1'
    ) -> np.ndarray:
        """
        Perform cross-validation on a model.

        Args:
            model: Model instance
            X: Features
            y: Labels
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric

        Returns:
            Array of cross-validation scores
        """
        logger.info(f"Performing {cv_folds}-fold cross-validation")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        logger.info(f"CV Scores: {scores}")
        logger.info(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return scores

    def get_trained_model(self, model_name: str) -> Optional[Any]:
        """
        Retrieve a trained model.

        Args:
            model_name: Name of the model

        Returns:
            Trained model instance or None
        """
        return self.models.get(model_name)
