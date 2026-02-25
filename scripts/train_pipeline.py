"""
Main training pipeline orchestration script.
Replaces the monolithic analysis.py with a modular, configuration-driven approach.
"""

import argparse
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import json
import joblib
import pandas as pd
from typing import Dict, Any

# Data pipeline imports
from data_pipeline.loaders import DataLoader
from data_pipeline.preprocessors import DataPreprocessor, TargetEncoder
from data_pipeline.feature_engineer import ChurnFeatureEngineer, FeatureEncoder
from data_pipeline.splitters import DataSplitter, FeatureScaler, ImbalanceHandler
from data_pipeline.validators import DataValidator

# Training imports
from training.models import ModelFactory
from training.trainers import ModelTrainer
from training.evaluators import ModelEvaluator
from training.hyperparameter_tuner import HyperparameterTuner

# Utility imports
from utils.config_loader import ConfigLoader
from utils.logger import setup_logger
from utils.visualization import ModelVisualizer

# MLflow tracking
from monitoring.mlflow_tracker import MLflowTracker


def main():
    """Main pipeline execution function."""

    parser = argparse.ArgumentParser(description="RetainML training pipeline")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna hyperparameter optimization before final training",
    )
    args = parser.parse_args()

    # Load configuration
    config_loader = ConfigLoader(config_dir='config')
    config = config_loader.load_all_configs()

    # Setup logging
    pipeline_config = config.get('pipeline', {})
    logging_config = pipeline_config.get('logging', {})

    logger = setup_logger(
        name='retainml',
        level=logging_config.get('level', 'INFO'),
        log_file=logging_config.get('log_file') if logging_config.get('log_to_file', False) else None,
        log_format=logging_config.get('format')
    )

    logger.info("="*80)
    logger.info("RETAINML - CUSTOMER CHURN PREDICTION")
    logger.info("Feature Engineering & Baseline Model Development")
    logger.info(f"Pipeline: {pipeline_config.get('name', 'Unknown')}")
    logger.info(f"Version: {pipeline_config.get('version', 'Unknown')}")
    logger.info(f"HPO tuning: {'enabled' if args.tune else 'disabled'}")
    logger.info("="*80)

    # Initialize MLflow tracker
    tracker = MLflowTracker(config)

    # Wrap entire pipeline in parent MLflow run
    with tracker.start_pipeline_run():

        # 1. Load Data
        data_loader = DataLoader(config)
        df = data_loader.load_csv()

        # 1b. Validate raw data
        validator = DataValidator(config)
        raw_result = validator.validate_raw_data(df)
        if not raw_result.passed:
            logger.error("Raw data validation failed. Stopping pipeline.")
            return

        # 2. Preprocess Data
        preprocessor = DataPreprocessor(config)
        df = preprocessor.preprocess(df)

        # 3. Feature Engineering
        feature_engineer = ChurnFeatureEngineer(config)
        df = feature_engineer.create_all_features(df)

        # 4. Encode Target
        target_encoder = TargetEncoder(config)
        df = target_encoder.encode_target(df)

        # 4b. Validate processed data (before split — works on mixed-type df)
        proc_result = validator.validate_processed_data(df)
        if not proc_result.passed:
            logger.error("Processed data validation failed. Stopping pipeline.")
            return

        # 5. Split Data — done BEFORE feature encoding to prevent OHE leakage
        splitter = DataSplitter(config)
        X_train, X_test, y_train, y_test = splitter.split_data(df)

        # 5a. Fix high_monthly_charge: recompute threshold from training data only
        #     The feature_engineer computed it on the full df above; we override it
        #     here using only the train split so test data cannot influence the threshold.
        charge_threshold = X_train['MonthlyCharges'].median()
        X_train['high_monthly_charge'] = (X_train['MonthlyCharges'] > charge_threshold).astype(int)
        X_test['high_monthly_charge'] = (X_test['MonthlyCharges'] > charge_threshold).astype(int)
        logger.info(f"high_monthly_charge threshold (train median): {charge_threshold:.2f}")

        # 6. Encode Features — fit on X_train only, then transform X_test
        feature_encoder = FeatureEncoder(config)
        X_train = feature_encoder.fit_transform(X_train)
        X_test = feature_encoder.transform(X_test)

        # 7. Scale Features
        scaler = FeatureScaler(config)
        X_train_scaled, X_test_scaled = scaler.fit_transform(X_train, X_test)

        # 8. Handle Class Imbalance
        imbalance_handler = ImbalanceHandler(config)
        X_train_balanced, y_train_balanced = imbalance_handler.apply_smote(
            X_train_scaled, y_train
        )

        # Log dataset info and pipeline parameters to parent run
        tracker.log_dataset_info(
            config,
            n_rows=len(df),
            n_features=X_train.shape[1],
            class_distribution=y_train.value_counts().to_dict()
        )
        tracker.log_params({
            "pipeline.name": pipeline_config.get('name'),
            "pipeline.version": pipeline_config.get('version'),
            "data.test_size": config.get('preprocessing', {}).get('test_size'),
            "data.scaler_type": config.get('preprocessing', {}).get('scaler_type'),
            "data.smote_applied": config.get('preprocessing', {}).get('apply_smote'),
            "data.train_samples": X_train_balanced.shape[0],
            "data.test_samples": X_test_scaled.shape[0],
            "pipeline.hpo_enabled": args.tune,
        })

        # 9. Hyperparameter Optimization (optional, --tune flag)
        if args.tune:
            logger.info("\n" + "="*80)
            logger.info("HYPERPARAMETER OPTIMIZATION (Optuna)")
            logger.info("="*80)

            tuner = HyperparameterTuner(config)
            search_spaces = config.get('search_spaces', {})

            for model_key, model_cfg in config.get('models', {}).items():
                if not model_cfg.get('enabled', True):
                    continue
                if model_key not in search_spaces:
                    logger.info(f"No search space for {model_key}, skipping HPO")
                    continue

                display_name = model_cfg.get('display_name', model_key)
                logger.info(f"\nTuning {display_name}...")

                result = tuner.optimize(model_key, X_train_balanced, y_train_balanced)
                best_params = result['best_params']

                # Override config params with tuned values
                config['models'][model_key]['params'].update(best_params)

                logger.info(f"Best params for {display_name}: {best_params}")
                logger.info(f"Best CV score: {result['best_score']:.4f}")

                tracker.log_params({
                    f"hpo.{model_key}.best_score": result['best_score'],
                    **{f"hpo.{model_key}.{k}": v for k, v in best_params.items()},
                })

                # Save best params to results/hpo/
                hpo_output_dir = config.get('hpo', {}).get('output_dir', 'results/hpo')
                tuner.save_results(model_key, f"{hpo_output_dir}/{model_key}_best_params.json")

        # 10. Create Models (with HPO-tuned params if --tune was used)
        models = ModelFactory.create_models_from_config(config)

        # 11-12. Train and Evaluate Models (per-model for nested MLflow runs)
        trainer = ModelTrainer(config)
        evaluator = ModelEvaluator(config)
        trained_models = {}

        logger.info("\n" + "="*80)
        logger.info("TRAINING BASELINE MODELS")
        logger.info("="*80)

        for model_display_name, model in models.items():
            with tracker.start_model_run(model_display_name):

                # Train
                trained_model = trainer.train_single_model(
                    model_display_name, model, X_train_balanced, y_train_balanced
                )
                trained_models[model_display_name] = trained_model

                # Log model hyperparameters
                for key, mcfg in config.get('models', {}).items():
                    if mcfg.get('display_name') == model_display_name:
                        tracker.log_params({
                            f"model.{k}": v for k, v in mcfg.get('params', {}).items()
                        })
                        break

                # Evaluate
                metrics = evaluator.evaluate_single_model(
                    model_display_name, trained_model, X_test_scaled, y_test
                )

                # Log metrics to child run
                tracker.log_metrics({
                    "accuracy": metrics['accuracy'],
                    "precision": metrics['precision'],
                    "recall": metrics['recall'],
                    "f1_score": metrics['f1_score'],
                    "roc_auc": metrics.get('roc_auc', 0),
                })

                # Log the trained model artifact
                tracker.log_model(trained_model, model_display_name)

        logger.info("\nAll models trained successfully")

        # Build results DataFrame from evaluator's stored results
        results_list = []
        for name in trained_models:
            results_list.append(evaluator.results[name]['metrics'])
        results_df = pd.DataFrame(results_list)

        primary_metric = config['evaluation'].get('primary_metric', 'f1_score')
        if primary_metric in results_df.columns:
            results_df = results_df.sort_values(primary_metric, ascending=False)

        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON")
        logger.info("="*80)
        logger.info(f"\n{results_df.to_string(index=False)}")

        # 13. Detailed Evaluation of Best Model
        best_model_name = results_df.iloc[0]['Model']
        evaluator.get_detailed_evaluation(best_model_name, y_test)

        # 14. Feature Importance
        logger.info("\n" + "="*80)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*80)

        for model_name, model in trained_models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df = evaluator.get_feature_importance(
                    model_name, model, X_train.columns.tolist(),
                    top_n=config['evaluation']['visualizations'].get('feature_importance_top_n', 20)
                )

        # 15. Create Visualizations
        visualizer = ModelVisualizer(config)

        # Model comparison
        visualizer.plot_model_comparison(results_df)

        # Confusion matrices
        predictions = {name: evaluator.get_predictions(name) for name in trained_models.keys()}
        visualizer.plot_confusion_matrices(trained_models, predictions, y_test)

        # ROC curves
        visualizer.plot_roc_curves(predictions, y_test)

        # Feature importance for tree-based models
        for model_name, model in trained_models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df = evaluator.get_feature_importance(
                    model_name, model, X_train.columns.tolist()
                )
                visualizer.plot_feature_importance(importance_df, model_name)

        # 16. Save Results
        logger.info("\n" + "="*80)
        logger.info("SAVING RESULTS")
        logger.info("="*80)

        results_path = Path(config['pipeline']['paths']['results_dir'])
        results_path.mkdir(parents=True, exist_ok=True)

        results_file = results_path / 'baseline_model_results.csv'
        results_df.to_csv(results_file, index=False)
        logger.info(f"Results saved to {results_file}")

        # Log artifacts to parent MLflow run
        artifact_config = config.get('mlflow', {}).get('log_artifacts', {})

        if artifact_config.get('results_csv', True):
            tracker.log_artifact(str(results_file), "results")

        if artifact_config.get('plots', True):
            for png_file in results_path.glob('*.png'):
                tracker.log_artifact(str(png_file), "plots")

        if artifact_config.get('config_snapshot', True):
            config_dir = Path('config')
            for yaml_file in config_dir.glob('*.yaml'):
                tracker.log_artifact(str(yaml_file), "config")

        # Register best model in MLflow Model Registry
        best_model = trained_models[best_model_name]
        best_metrics = evaluator.results[best_model_name]['metrics']
        tracker.register_best_model(best_model, best_model_name, best_metrics)

        # 17. Save serving artifacts
        models_dir = Path('models')
        models_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(best_model, models_dir / 'model.pkl')
        joblib.dump(scaler.get_scaler(), models_dir / 'scaler.pkl')

        with open(models_dir / 'feature_columns.json', 'w') as f:
            json.dump(X_train.columns.tolist(), f)

        with open(models_dir / 'metadata.json', 'w') as f:
            json.dump({
                'best_model': best_model_name,
                'f1_score': float(results_df.iloc[0]['f1_score']),
                'roc_auc': float(results_df.iloc[0]['roc_auc']),
                'hpo_tuned': args.tune,
                'charge_threshold': float(charge_threshold),
            }, f, indent=2)

        logger.info(f"Serving artifacts saved to {models_dir}/")

        # Summary
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"F1-Score: {results_df.iloc[0]['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")

        if args.tune:
            logger.info("\nHPO best parameters have been logged to MLflow and results/hpo/")
        logger.info("\nNext steps:")
        logger.info("1. Review feature importance and select top features")
        logger.info("2. Try ensemble methods or run with --tune for HPO")
        logger.info("3. Implement cross-validation for robust evaluation")


if __name__ == "__main__":
    main()
