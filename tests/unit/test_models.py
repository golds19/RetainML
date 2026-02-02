"""Tests for ModelFactory."""

import sys
from pathlib import Path
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from training.models import ModelFactory


class TestCreateModel:

    def test_logistic_regression(self):
        model = ModelFactory.create_model("logistic_regression", {"max_iter": 100})
        assert isinstance(model, LogisticRegression)
        assert model.max_iter == 100

    def test_random_forest(self):
        model = ModelFactory.create_model("random_forest", {"n_estimators": 50})
        assert isinstance(model, RandomForestClassifier)
        assert model.n_estimators == 50

    def test_gradient_boosting(self):
        model = ModelFactory.create_model("gradient_boosting", {"n_estimators": 50})
        assert isinstance(model, GradientBoostingClassifier)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create_model("xgboost", {})

    def test_empty_params_uses_defaults(self):
        model = ModelFactory.create_model("logistic_regression", {})
        assert isinstance(model, LogisticRegression)

    def test_registry_keys(self):
        expected = {"logistic_regression", "random_forest", "gradient_boosting"}
        assert set(ModelFactory.MODEL_REGISTRY.keys()) == expected


class TestCreateModelsFromConfig:

    def test_creates_all_enabled(self, sample_config):
        models = ModelFactory.create_models_from_config(sample_config)
        assert len(models) == 3
        assert "Logistic Regression" in models
        assert "Random Forest" in models
        assert "Gradient Boosting" in models

    def test_skips_disabled(self, sample_config):
        sample_config["models"]["random_forest"]["enabled"] = False
        models = ModelFactory.create_models_from_config(sample_config)
        assert len(models) == 2
        assert "Random Forest" not in models

    def test_all_have_fit_method(self, sample_config):
        models = ModelFactory.create_models_from_config(sample_config)
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit method"
