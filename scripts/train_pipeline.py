"""
Main training pipeline orchestration script.
Replaces the monolithic analysis.py with a modular, configuration-driven approach.
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
from typing import Dict, Any

# Data pipeline imports
from data_pipeline.loaders import DataLoader
from data_pipeline.preprocessors import DataPreprocessor, TargetEncoder
from data_pipeline.feature_engineer import ChurnFeatureEngineer, FeatureEncoder
from data_pipeline.splitters import DataSplitter, FeatureScaler, ImbalanceHandler

# Training imports
from training.models import ModelFactory
from training.trainers import ModelTrainer
from training.evaluators import ModelEvaluator

# Utility imports
from utils.config_loader import ConfigLoader
from utils.logger import setup_logger
from utils.visualization import ModelVisualizer

# MLflow tracking
from monitoring.mlflow_tracker import MLflowTracker


def main():
    """Main pipeline execution function."""

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
    logger.info("="*80)

    # Initialize MLflow tracker
    tracker = MLflowTracker(config)

    # Wrap entire pipeline in parent MLflow run
    with tracker.start_pipeline_run():

        # 1. Load Data
        data_loader = DataLoader(config)
        df = data_loader.load_csv()

        # 2. Preprocess Data
        preprocessor = DataPreprocessor(config)
        df = preprocessor.preprocess(df)

        # 3. Feature Engineering
        feature_engineer = ChurnFeatureEngineer(config)
        df = feature_engineer.create_all_features(df)

        # 4. Feature Encoding
        feature_encoder = FeatureEncoder(config)
        df = feature_encoder.encode_features(df)

        # 5. Encode Target
        target_encoder = TargetEncoder(config)
        df = target_encoder.encode_target(df)

        # 6. Split Data
        splitter = DataSplitter(config)
        X_train, X_test, y_train, y_test = splitter.split_data(df)

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
        })

        # 9. Create Models
        models = ModelFactory.create_models_from_config(config)

        # 10-11. Train and Evaluate Models (per-model for nested MLflow runs)
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

        # 12. Detailed Evaluation of Best Model
        best_model_name = results_df.iloc[0]['Model']
        evaluator.get_detailed_evaluation(best_model_name, y_test)

        # 13. Feature Importance
        logger.info("\n" + "="*80)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*80)

        for model_name, model in trained_models.items():
            if hasattr(model, 'feature_importances_'):
                importance_df = evaluator.get_feature_importance(
                    model_name, model, X_train.columns.tolist(),
                    top_n=config['evaluation']['visualizations'].get('feature_importance_top_n', 20)
                )

        # 14. Create Visualizations
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

        # 15. Save Results
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

        # Summary
        logger.info("\n" + "="*80)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("="*80)
        logger.info(f"\nBest Model: {best_model_name}")
        logger.info(f"F1-Score: {results_df.iloc[0]['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {results_df.iloc[0]['roc_auc']:.4f}")

        logger.info("\nNext steps:")
        logger.info("1. Review feature importance and select top features")
        logger.info("2. Perform hyperparameter tuning")
        logger.info("3. Try ensemble methods")
        logger.info("4. Implement cross-validation for robust evaluation")


if __name__ == "__main__":
    main()
