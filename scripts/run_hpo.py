"""
Hyperparameter optimization CLI script.
Runs Optuna-based HPO for churn prediction models.

Usage:
    python scripts/run_hpo.py                              # all models, defaults
    python scripts/run_hpo.py --model random_forest        # single model
    python scripts/run_hpo.py --n-trials 20 --no-mlflow    # quick test
"""

import sys
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd

# Data pipeline imports
from data_pipeline.loaders import DataLoader
from data_pipeline.preprocessors import DataPreprocessor, TargetEncoder
from data_pipeline.feature_engineer import ChurnFeatureEngineer, FeatureEncoder
from data_pipeline.splitters import DataSplitter, FeatureScaler, ImbalanceHandler

# Training imports
from training.models import ModelFactory
from training.evaluators import ModelEvaluator
from training.hyperparameter_tuner import HyperparameterTuner

# Utility imports
from utils.config_loader import ConfigLoader
from utils.logger import setup_logger

# MLflow tracking
from monitoring.mlflow_tracker import MLflowTracker


MODEL_ALIASES = {
    'lr': 'logistic_regression',
    'rf': 'random_forest',
    'gb': 'gradient_boosting',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run hyperparameter optimization for churn prediction models"
    )
    parser.add_argument(
        '--model', type=str, default='all',
        help='Model to optimize: lr, rf, gb, or all (default: all)'
    )
    parser.add_argument(
        '--n-trials', type=int, default=None,
        help='Number of Optuna trials (overrides config)'
    )
    parser.add_argument(
        '--cv-folds', type=int, default=None,
        help='Number of CV folds (overrides config)'
    )
    parser.add_argument(
        '--no-mlflow', action='store_true',
        help='Disable MLflow tracking'
    )
    parser.add_argument(
        '--config-dir', type=str, default='config',
        help='Configuration directory (default: config)'
    )
    return parser.parse_args()


def resolve_models(model_arg, search_spaces):
    """Resolve model argument to list of model names."""
    if model_arg == 'all':
        return list(search_spaces.keys())
    name = MODEL_ALIASES.get(model_arg, model_arg)
    if name not in search_spaces:
        raise ValueError(f"Unknown model '{model_arg}'. Use: lr, rf, gb, or all")
    return [name]


def prepare_data(config, logger):
    """Run the same data pipeline as train_pipeline.py."""
    logger.info("Loading and preparing data...")

    # 1. Load
    data_loader = DataLoader(config)
    df = data_loader.load_csv()

    # 2. Preprocess
    preprocessor = DataPreprocessor(config)
    df = preprocessor.preprocess(df)

    # 3. Feature engineering
    feature_engineer = ChurnFeatureEngineer(config)
    df = feature_engineer.create_all_features(df)

    # 4. Encode target
    target_encoder = TargetEncoder(config)
    df = target_encoder.encode_target(df)

    # 5. Split (before encoding to prevent OHE leakage)
    splitter = DataSplitter(config)
    X_train, X_test, y_train, y_test = splitter.split_data(df)

    # 5a. Fix high_monthly_charge threshold from train data only
    charge_threshold = X_train['MonthlyCharges'].median()
    X_train['high_monthly_charge'] = (X_train['MonthlyCharges'] > charge_threshold).astype(int)
    X_test['high_monthly_charge'] = (X_test['MonthlyCharges'] > charge_threshold).astype(int)

    # 6. Encode features (fit on train, transform test)
    feature_encoder = FeatureEncoder(config)
    X_train, X_test = feature_encoder.fit_transform(X_train), feature_encoder.transform(X_test)

    # 7. Scale
    scaler = FeatureScaler(config)
    X_train_scaled, X_test_scaled = scaler.fit_transform(X_train, X_test)

    # 8. Handle imbalance
    imbalance_handler = ImbalanceHandler(config)
    X_train_balanced, y_train_balanced = imbalance_handler.apply_smote(
        X_train_scaled, y_train
    )

    logger.info(f"Data ready: train={X_train_balanced.shape}, test={X_test_scaled.shape}")

    return X_train_balanced, y_train_balanced, X_test_scaled, y_test


def main():
    args = parse_args()

    # Load configs
    config_loader = ConfigLoader(config_dir=args.config_dir)
    config = config_loader.load_all_configs()
    hpo_config = config_loader.load_config('hpo_config.yaml')

    # Disable MLflow if requested
    if args.no_mlflow:
        config['mlflow'] = config.get('mlflow', {})
        config['mlflow']['enabled'] = False

    # Setup logging
    pipeline_config = config.get('pipeline', {})
    logging_config = pipeline_config.get('logging', {})
    logger = setup_logger(
        name='retainml.hpo',
        level=logging_config.get('level', 'INFO'),
    )

    logger.info("=" * 80)
    logger.info("RETAINML - HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)

    # Resolve models
    models_to_optimize = resolve_models(
        args.model, hpo_config.get('search_spaces', {})
    )
    logger.info(f"Models to optimize: {models_to_optimize}")

    # Prepare data
    X_train, y_train, X_test, y_test = prepare_data(config, logger)

    # Initialize tracker and tuner
    tracker = MLflowTracker(config)
    tuner = HyperparameterTuner(hpo_config)
    evaluator = ModelEvaluator(config)

    output_dir = Path(hpo_config.get('hpo', {}).get('output_dir', 'results/hpo'))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    with tracker.start_pipeline_run(run_name="hpo_optimization"):

        for model_name in models_to_optimize:
            logger.info(f"\n{'=' * 80}")
            logger.info(f"OPTIMIZING: {model_name}")
            logger.info(f"{'=' * 80}")

            with tracker.start_model_run(f"HPO_{model_name}"):

                # Run optimization
                result = tuner.optimize(
                    model_name=model_name,
                    X=X_train,
                    y=y_train,
                    n_trials=args.n_trials,
                    cv_folds=args.cv_folds,
                )

                # Save best params JSON
                tuner.save_results(
                    model_name,
                    str(output_dir / f"{model_name}_best_params.json"),
                )

                # Retrain with best params and evaluate on holdout
                best_params = dict(result['best_params'])
                best_params['random_state'] = 42
                if model_name in ('logistic_regression', 'random_forest'):
                    best_params['class_weight'] = 'balanced'

                best_model = ModelFactory.create_model(model_name, best_params)
                best_model.fit(X_train, y_train)

                metrics = evaluator.evaluate_single_model(
                    model_name, best_model, X_test, y_test
                )

                logger.info(f"\nHoldout metrics for {model_name}:")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        logger.info(f"  {k}: {v:.4f}")

                # Log to MLflow
                tracker.log_params({
                    f"best.{k}": v for k, v in result['best_params'].items()
                })
                tracker.log_metrics({
                    "best_cv_score": result['best_score'],
                    "holdout_f1": metrics.get('f1_score', 0),
                    "holdout_roc_auc": metrics.get('roc_auc', 0),
                    "n_trials": result['n_trials'],
                })
                tracker.log_model(best_model, model_name)

                all_results[model_name] = {
                    'best_cv_score': result['best_score'],
                    'holdout_f1': metrics.get('f1_score', 0),
                    'holdout_roc_auc': metrics.get('roc_auc', 0),
                    'best_params': result['best_params'],
                }

        # Summary
        logger.info(f"\n{'=' * 80}")
        logger.info("HPO SUMMARY")
        logger.info(f"{'=' * 80}")

        summary_rows = []
        for name, res in all_results.items():
            summary_rows.append({
                'Model': name,
                'Best CV F1': f"{res['best_cv_score']:.4f}",
                'Holdout F1': f"{res['holdout_f1']:.4f}",
                'Holdout ROC-AUC': f"{res['holdout_roc_auc']:.4f}",
            })

        summary_df = pd.DataFrame(summary_rows)
        logger.info(f"\n{summary_df.to_string(index=False)}")
        logger.info(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
