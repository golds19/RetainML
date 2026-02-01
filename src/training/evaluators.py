"""
Model evaluation and metrics calculation.
Handles comprehensive model evaluation and comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate and compare model performance."""

    def __init__(self, config: dict):
        """
        Initialize ModelEvaluator.

        Args:
            config: Evaluation configuration dictionary
        """
        self.config = config['evaluation']
        self.results = {}
        self.predictions = {}

    def evaluate_single_model(
        self,
        model_name: str,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate a single model.

        Args:
            model_name: Name of the model
            model: Trained model instance
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of metric scores
        """
        logger.info(f"\nEvaluating {model_name}...")

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        metrics['Model'] = model_name

        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        # Log metrics
        self._log_metrics(model_name, metrics)

        return metrics

    def evaluate_all_models(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate all models and create comparison dataframe.

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels

        Returns:
            DataFrame with evaluation results
        """
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)

        results_list = []

        for model_name, model in models.items():
            metrics = self.evaluate_single_model(model_name, model, X_test, y_test)
            results_list.append(metrics)

        # Create results dataframe
        results_df = pd.DataFrame(results_list)

        # Sort by primary metric
        primary_metric = self.config.get('primary_metric', 'f1_score')
        if primary_metric in results_df.columns:
            results_df = results_df.sort_values(primary_metric, ascending=False)

        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        logger.info(f"\n{results_df.to_string(index=False)}")

        return results_df

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray = None
    ) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        metrics = {}

        # Standard classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # ROC-AUC (requires probabilities)
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

        return metrics

    def _log_metrics(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Log model metrics."""
        logger.info(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
        logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"Recall:    {metrics.get('recall', 0):.4f}")
        logger.info(f"F1-Score:  {metrics.get('f1_score', 0):.4f}")
        logger.info(f"ROC-AUC:   {metrics.get('roc_auc', 0):.4f}")

    def get_detailed_evaluation(
        self,
        model_name: str,
        y_test: pd.Series
    ) -> Dict[str, Any]:
        """
        Get detailed evaluation for a specific model.

        Args:
            model_name: Name of the model
            y_test: Test labels

        Returns:
            Dictionary with detailed evaluation results
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} has not been evaluated")

        logger.info("\n" + "="*80)
        logger.info(f"DETAILED EVALUATION: {model_name}")
        logger.info("="*80)

        y_pred = self.results[model_name]['y_pred']

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        logger.info("\nConfusion Matrix:")
        logger.info(cm)

        # Classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

        return {
            'confusion_matrix': cm,
            'classification_report': classification_report(
                y_test, y_pred,
                target_names=['No Churn', 'Churn'],
                output_dict=True
            )
        }

    def get_feature_importance(
        self,
        model_name: str,
        model: Any,
        feature_names: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.

        Args:
            model_name: Name of the model
            model: Trained model instance
            feature_names: List of feature names
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"Model {model_name} does not have feature_importances_")
            return None

        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE IMPORTANCE: {model_name}")
        logger.info(f"{'='*80}")

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)

        logger.info(f"\nTop {top_n} Important Features:")
        logger.info(f"\n{importance_df.head(top_n).to_string(index=False)}")

        return importance_df

    def get_predictions(self, model_name: str) -> Dict[str, np.ndarray]:
        """
        Get predictions for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with predictions and probabilities
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} has not been evaluated")

        return {
            'y_pred': self.results[model_name]['y_pred'],
            'y_pred_proba': self.results[model_name]['y_pred_proba']
        }
