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

    # 9. Create Models
    models = ModelFactory.create_models_from_config(config)

    # 10. Train Models
    trainer = ModelTrainer(config)
    trained_models = trainer.train_all_models(models, X_train_balanced, y_train_balanced)

    # 11. Evaluate Models
    evaluator = ModelEvaluator(config)
    results_df = evaluator.evaluate_all_models(trained_models, X_test_scaled, y_test)

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
