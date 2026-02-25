"""
Visualization utilities for model results.
Creates plots for model comparison, confusion matrices, and feature importance.
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be set before pyplot import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import logging

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """Create visualizations for model evaluation."""

    def __init__(self, config: dict):
        """
        Initialize ModelVisualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config['evaluation']['visualizations']
        self.save_path = Path(self.config['save_path'])
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style('whitegrid')

    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        metrics: List[str] = None
    ) -> None:
        """
        Create bar plots comparing model performance across metrics.

        Args:
            results_df: DataFrame with model results
            metrics: List of metrics to plot
        """
        if not self.config.get('create_model_comparison', True):
            return

        logger.info("\n" + "="*80)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("="*80)
        logger.info("\n1. Creating model comparison plot...")

        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

        # Filter metrics that exist in results
        available_metrics = [m for m in metrics if m in results_df.columns]

        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))

        # Handle single row case
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        axes = axes.ravel()

        for idx, metric in enumerate(available_metrics):
            if idx >= len(axes):
                break

            ax = axes[idx]

            bars = ax.bar(
                results_df['Model'],
                results_df[metric],
                color=['#3498db', '#e74c3c', '#2ecc71'][:len(results_df)]
            )

            ax.set_title(f'{metric.replace("_", " ").title()} Comparison',
                        fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45)
            ax.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)

        # Remove extra subplots
        for idx in range(len(available_metrics), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        save_file = self.save_path / 'model_comparison.png'
        plt.savefig(save_file, dpi=self.config.get('dpi', 300), bbox_inches='tight')
        logger.info(f"Saved: {save_file}")
        plt.close()

    def plot_confusion_matrices(
        self,
        models: Dict[str, Any],
        predictions: Dict[str, np.ndarray],
        y_test: pd.Series
    ) -> None:
        """
        Create confusion matrix heatmaps for all models.

        Args:
            models: Dictionary of model instances
            predictions: Dictionary of predictions for each model
            y_test: Test labels
        """
        if not self.config.get('create_confusion_matrix', True):
            return

        logger.info("2. Creating confusion matrices...")

        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))

        if n_models == 1:
            axes = [axes]

        for idx, (model_name, preds) in enumerate(predictions.items()):
            y_pred = preds['y_pred']
            cm = confusion_matrix(y_test, y_pred)

            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                ax=axes[idx], cbar=True, square=True
            )

            axes[idx].set_title(f'{model_name}\nConfusion Matrix',
                              fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_xticklabels(['No Churn', 'Churn'])
            axes[idx].set_yticklabels(['No Churn', 'Churn'])

        plt.tight_layout()
        save_file = self.save_path / 'confusion_matrices.png'
        plt.savefig(save_file, dpi=self.config.get('dpi', 300), bbox_inches='tight')
        logger.info(f"Saved: {save_file}")
        plt.close()

    def plot_roc_curves(
        self,
        predictions: Dict[str, np.ndarray],
        y_test: pd.Series
    ) -> None:
        """
        Create ROC curve comparison plot.

        Args:
            predictions: Dictionary of predictions for each model
            y_test: Test labels
        """
        if not self.config.get('create_roc_curve', True):
            return

        logger.info("3. Creating ROC curves...")

        plt.figure(figsize=(10, 8))

        for model_name, preds in predictions.items():
            y_pred_proba = preds['y_pred_proba']

            if y_pred_proba is None:
                continue

            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)

            plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')

        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)

        plt.tight_layout()
        save_file = self.save_path / 'roc_curves.png'
        plt.savefig(save_file, dpi=self.config.get('dpi', 300), bbox_inches='tight')
        logger.info(f"Saved: {save_file}")
        plt.close()

    def plot_feature_importance(
        self,
        importance_df: pd.DataFrame,
        model_name: str,
        top_n: Optional[int] = None
    ) -> None:
        """
        Create feature importance plot.

        Args:
            importance_df: DataFrame with feature importance
            model_name: Name of the model
            top_n: Number of top features to plot
        """
        if not self.config.get('create_feature_importance', True):
            return

        if importance_df is None:
            return

        if top_n is None:
            top_n = self.config.get('feature_importance_top_n', 20)

        top_features = importance_df.head(top_n)

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances - {model_name}',
                 fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        save_file = self.save_path / f'feature_importance_{model_name.replace(" ", "_").lower()}.png'
        plt.savefig(save_file, dpi=self.config.get('dpi', 300), bbox_inches='tight')
        logger.info("Saved feature importance plot")
        plt.close()

        logger.info("\nAll visualizations saved successfully!")
