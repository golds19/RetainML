"""
Hyperparameter optimization using Optuna.
Provides config-driven search over model hyperparameters with cross-validation.
"""

import json
import numpy as np
import pandas as pd
import optuna
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from sklearn.model_selection import StratifiedKFold, cross_val_score
import logging

from training.models import ModelFactory

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose trial logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    """Optuna-based hyperparameter optimization."""

    def __init__(self, hpo_config: dict):
        """
        Initialize tuner.

        Args:
            hpo_config: HPO configuration with study settings and search spaces
        """
        self.hpo_config = hpo_config
        self.settings = hpo_config.get('hpo', {})
        self.search_spaces = hpo_config.get('search_spaces', {})
        self.studies: Dict[str, optuna.Study] = {}

    def optimize(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        n_trials: Optional[int] = None,
        scoring: Optional[str] = None,
        cv_folds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization for a model.

        Args:
            model_name: Key in MODEL_REGISTRY (e.g. 'random_forest')
            X: Training features
            y: Training labels
            n_trials: Number of trials (overrides config)
            scoring: Scoring metric (overrides config)
            cv_folds: CV folds (overrides config)

        Returns:
            Dict with best_params, best_score, n_trials, best_trial
        """
        if model_name not in self.search_spaces:
            raise ValueError(
                f"No search space defined for '{model_name}'. "
                f"Available: {list(self.search_spaces.keys())}"
            )

        n_trials = n_trials or self.settings.get('n_trials', 50)
        scoring = scoring or self.settings.get('scoring', 'f1')
        cv_folds = cv_folds or self.settings.get('cv_folds', 5)

        logger.info(f"Starting HPO for {model_name}: {n_trials} trials, "
                     f"{cv_folds}-fold CV, metric={scoring}")

        study = optuna.create_study(direction='maximize')

        def objective(trial):
            return self._objective(trial, model_name, X, y, scoring, cv_folds)

        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self.studies[model_name] = study

        best = study.best_trial
        logger.info(f"Best trial #{best.number}: score={best.value:.4f}, "
                     f"params={best.params}")

        return {
            'best_params': best.params,
            'best_score': best.value,
            'n_trials': len(study.trials),
            'best_trial': best.number,
        }

    def _objective(
        self,
        trial: optuna.Trial,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        scoring: str,
        cv_folds: int,
    ) -> float:
        """Optuna objective: suggest params, CV score, return mean."""
        params = self._suggest_params(trial, model_name)
        params['random_state'] = 42

        # Add class_weight=balanced for models that support it
        if model_name in ('logistic_regression', 'random_forest'):
            params['class_weight'] = 'balanced'

        model = ModelFactory.create_model(model_name, params)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

        return scores.mean()

    def _suggest_params(
        self, trial: optuna.Trial, model_name: str
    ) -> Dict[str, Any]:
        """Suggest hyperparameters from config-defined search space."""
        space = self.search_spaces[model_name]
        params = {}

        for param_name, spec in space.items():
            ptype = spec['type']

            if ptype == 'int':
                params[param_name] = trial.suggest_int(
                    param_name, spec['low'], spec['high'],
                    step=spec.get('step', 1),
                )
            elif ptype == 'float':
                params[param_name] = trial.suggest_float(
                    param_name, spec['low'], spec['high'],
                    log=spec.get('log', False),
                )
            elif ptype == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name, spec['choices'],
                )

        return params

    def save_results(self, model_name: str, output_path: str) -> None:
        """
        Save best parameters to JSON.

        Args:
            model_name: Model key
            output_path: Path for output JSON file
        """
        if model_name not in self.studies:
            raise ValueError(f"No study found for '{model_name}'. Run optimize() first.")

        study = self.studies[model_name]
        best = study.best_trial

        result = {
            'model_name': model_name,
            'best_params': best.params,
            'best_score': best.value,
            'n_trials': len(study.trials),
            'best_trial': best.number,
            'scoring': self.settings.get('scoring', 'f1'),
            'cv_folds': self.settings.get('cv_folds', 5),
            'timestamp': datetime.now().isoformat(),
        }

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w') as f:
            json.dump(result, f, indent=2, default=str)

        logger.info(f"Best params saved to {output_path}")
