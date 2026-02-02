"""
MLflow experiment tracking wrapper.
Provides a config-driven interface for logging experiments, metrics, and artifacts.
"""

import mlflow
import mlflow.sklearn
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import contextmanager
from datetime import datetime

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Config-driven MLflow experiment tracking.

    All methods are no-ops when MLflow is disabled via configuration.
    Errors are caught and logged when fail_silently is True.
    """

    def __init__(self, config: dict):
        """
        Initialize MLflowTracker from merged config dict.

        Args:
            config: Full merged configuration (same dict passed to all components)
        """
        self.mlflow_config = config.get('mlflow', {})
        self.enabled = self.mlflow_config.get('enabled', False)
        self.fail_silently = self.mlflow_config.get('connection', {}).get('fail_silently', True)
        self._initialized = False

        if self.enabled:
            self._initialize()

    def _initialize(self) -> None:
        """Set up MLflow tracking URI and experiment."""
        try:
            tracking_uri = self.mlflow_config.get('tracking_uri', './mlruns')
            mlflow.set_tracking_uri(tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")

            experiment_name = self.mlflow_config.get('experiment_name', 'churn_prediction')
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")

            self._initialized = True

        except Exception as e:
            logger.warning(f"MLflow initialization failed: {e}")
            if not self.fail_silently:
                raise
            self.enabled = False

    def _safe_execute(self, func, *args, **kwargs):
        """Execute an MLflow operation with error handling."""
        if not self.enabled:
            return None
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.warning(f"MLflow operation failed: {e}")
            if not self.fail_silently:
                raise
            return None

    # --- Run Management ---

    @contextmanager
    def start_pipeline_run(self, run_name: str = None):
        """
        Context manager for a parent pipeline run.

        Usage:
            with tracker.start_pipeline_run() as run:
                # ... pipeline steps ...
        """
        if not self.enabled:
            yield None
            return

        if run_name is None:
            prefix = self.mlflow_config.get('run_name_prefix', 'pipeline')
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            run_name = f"{prefix}_{timestamp}"

        tags = self.mlflow_config.get('default_tags', {})

        try:
            run = mlflow.start_run(run_name=run_name, tags=tags)
            logger.info(f"Started MLflow parent run: {run_name} (ID: {run.info.run_id})")
            yield run
        except Exception as e:
            logger.warning(f"MLflow parent run failed: {e}")
            if not self.fail_silently:
                raise
            yield None
        finally:
            self._safe_execute(mlflow.end_run)

    @contextmanager
    def start_model_run(self, model_name: str):
        """
        Context manager for a nested child run (one per model).

        Usage:
            with tracker.start_model_run("Random Forest") as run:
                # train, evaluate
        """
        if not self.enabled:
            yield None
            return

        try:
            child_run = mlflow.start_run(
                run_name=model_name,
                nested=True,
                tags={"model_name": model_name}
            )
            logger.info(f"Started MLflow child run: {model_name}")
            yield child_run
        except Exception as e:
            logger.warning(f"MLflow child run failed: {e}")
            if not self.fail_silently:
                raise
            yield None
        finally:
            self._safe_execute(mlflow.end_run)

    # --- Logging Methods ---

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dictionary of parameters (flattened if nested)."""

        def _flatten(d, prefix=''):
            flat = {}
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    flat.update(_flatten(v, key))
                else:
                    flat[key] = v
            return flat

        flat_params = _flatten(params)
        # MLflow has 500-char value limit; truncate
        for k, v in flat_params.items():
            self._safe_execute(mlflow.log_param, k, str(v)[:500])

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log a dictionary of numeric metrics."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._safe_execute(mlflow.log_metric, key, value)

    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """Log a local file as an artifact."""
        if Path(local_path).exists():
            self._safe_execute(mlflow.log_artifact, local_path, artifact_path)
        else:
            logger.warning(f"Artifact not found, skipping: {local_path}")

    def log_artifacts_dir(self, local_dir: str, artifact_path: str = None) -> None:
        """Log all files in a directory as artifacts."""
        if Path(local_dir).exists():
            self._safe_execute(mlflow.log_artifacts, local_dir, artifact_path)
        else:
            logger.warning(f"Artifact directory not found, skipping: {local_dir}")

    def log_model(self, model, model_name: str) -> None:
        """Log a scikit-learn model."""
        self._safe_execute(
            mlflow.sklearn.log_model,
            model,
            artifact_path=model_name,
        )

    def log_dataset_info(self, config: dict, n_rows: int, n_features: int,
                         class_distribution: Dict = None) -> None:
        """Log dataset metadata as parameters."""
        preprocessing = config.get('preprocessing', {})
        dataset_params = {
            "dataset.n_rows": n_rows,
            "dataset.n_features": n_features,
            "dataset.test_size": preprocessing.get('test_size', 'unknown'),
            "dataset.scaler": preprocessing.get('scaler_type', 'unknown'),
            "dataset.smote_applied": preprocessing.get('apply_smote', False),
        }
        if class_distribution:
            for label, count in class_distribution.items():
                dataset_params[f"dataset.class_{label}"] = count

        for k, v in dataset_params.items():
            self._safe_execute(mlflow.log_param, k, v)

    # --- Model Registry ---

    def register_best_model(self, model, model_name: str, metrics: Dict[str, float]) -> None:
        """Register the best model in MLflow Model Registry."""
        registry_config = self.mlflow_config.get('model_registry', {})
        if not registry_config.get('enabled', False):
            return
        if not registry_config.get('register_best_model', False):
            return

        registered_name = registry_config.get('model_name', 'churn_predictor')

        try:
            mlflow.sklearn.log_model(
                model,
                artifact_path="best_model",
                registered_model_name=registered_name,
            )
            logger.info(f"Registered model '{registered_name}' in MLflow Model Registry")

            # Tag with metrics
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self._safe_execute(mlflow.set_tag, f"best_model.{k}", f"{v:.4f}")

            # Tag with registry tags
            tags = registry_config.get('tags', {})
            for k, v in tags.items():
                self._safe_execute(mlflow.set_tag, f"registry.{k}", v)

        except Exception as e:
            logger.warning(f"Model registration failed: {e}")
            if not self.fail_silently:
                raise
