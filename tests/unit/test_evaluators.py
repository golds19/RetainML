"""Tests for ModelEvaluator."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from training.evaluators import ModelEvaluator


class TestEvaluateSingleModel:

    def test_returns_metrics_dict(self, sample_config, fitted_lr_model, sample_X_y):
        _, X_test, _, y_test = sample_X_y
        evaluator = ModelEvaluator(sample_config)
        metrics = evaluator.evaluate_single_model("LR", fitted_lr_model, X_test, y_test)

        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "Model"]:
            assert key in metrics, f"Missing key: {key}"

    def test_metrics_in_valid_range(self, sample_config, fitted_lr_model, sample_X_y):
        _, X_test, _, y_test = sample_X_y
        evaluator = ModelEvaluator(sample_config)
        metrics = evaluator.evaluate_single_model("LR", fitted_lr_model, X_test, y_test)

        for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"

    def test_stores_results(self, sample_config, fitted_lr_model, sample_X_y):
        _, X_test, _, y_test = sample_X_y
        evaluator = ModelEvaluator(sample_config)
        evaluator.evaluate_single_model("LR", fitted_lr_model, X_test, y_test)

        assert "LR" in evaluator.results
        assert "metrics" in evaluator.results["LR"]
        assert "y_pred" in evaluator.results["LR"]
        assert "y_pred_proba" in evaluator.results["LR"]


class TestEvaluateAllModels:

    def test_returns_sorted_dataframe(self, sample_config, sample_X_y):
        X_train, X_test, y_train, y_test = sample_X_y

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)

        evaluator = ModelEvaluator(sample_config)
        results_df = evaluator.evaluate_all_models({"LR": lr, "RF": rf}, X_test, y_test)

        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 2
        # Should be sorted by f1_score descending
        assert results_df.iloc[0]["f1_score"] >= results_df.iloc[1]["f1_score"]

    def test_has_correct_columns(self, sample_config, sample_X_y):
        X_train, X_test, y_train, y_test = sample_X_y

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)

        evaluator = ModelEvaluator(sample_config)
        results_df = evaluator.evaluate_all_models({"LR": lr}, X_test, y_test)

        for col in ["Model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]:
            assert col in results_df.columns


class TestGetFeatureImportance:

    def test_tree_model_returns_dataframe(self, sample_config, sample_X_y):
        X_train, X_test, y_train, y_test = sample_X_y
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)

        evaluator = ModelEvaluator(sample_config)
        evaluator.evaluate_single_model("RF", rf, X_test, y_test)

        importance = evaluator.get_feature_importance("RF", rf, X_train.columns.tolist())
        assert isinstance(importance, pd.DataFrame)
        assert "Feature" in importance.columns
        assert "Importance" in importance.columns

    def test_linear_model_returns_none(self, sample_config, fitted_lr_model, sample_X_y):
        _, X_test, _, y_test = sample_X_y
        evaluator = ModelEvaluator(sample_config)
        evaluator.evaluate_single_model("LR", fitted_lr_model, X_test, y_test)

        result = evaluator.get_feature_importance("LR", fitted_lr_model, ["f0", "f1", "f2"])
        assert result is None


class TestGetPredictions:

    def test_after_evaluation(self, sample_config, fitted_lr_model, sample_X_y):
        _, X_test, _, y_test = sample_X_y
        evaluator = ModelEvaluator(sample_config)
        evaluator.evaluate_single_model("LR", fitted_lr_model, X_test, y_test)

        preds = evaluator.get_predictions("LR")
        assert "y_pred" in preds
        assert "y_pred_proba" in preds
        assert len(preds["y_pred"]) == len(y_test)

    def test_not_evaluated_raises(self, sample_config):
        evaluator = ModelEvaluator(sample_config)
        with pytest.raises(ValueError):
            evaluator.get_predictions("Unknown")


class TestGetDetailedEvaluation:

    def test_returns_confusion_matrix(self, sample_config, fitted_lr_model, sample_X_y):
        _, X_test, _, y_test = sample_X_y
        evaluator = ModelEvaluator(sample_config)
        evaluator.evaluate_single_model("LR", fitted_lr_model, X_test, y_test)

        detail = evaluator.get_detailed_evaluation("LR", y_test)
        assert "confusion_matrix" in detail
        assert detail["confusion_matrix"].shape == (2, 2)

    def test_not_evaluated_raises(self, sample_config):
        evaluator = ModelEvaluator(sample_config)
        with pytest.raises(ValueError):
            evaluator.get_detailed_evaluation("Unknown", pd.Series([0, 1]))
