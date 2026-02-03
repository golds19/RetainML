"""Tests for serving.predict and serving.app (FastAPI)."""

import sys
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from serving.predict import ChurnPredictor
from tests.conftest import FEATURE_COLUMNS


# --- ChurnPredictor ---

class TestChurnPredictor:
    """Test ChurnPredictor with fully mocked file I/O."""

    def _build_predictor(self, mock_model, mock_scaler):
        """Construct a ChurnPredictor by patching all file loads."""
        metadata = {"best_model": "Random Forest", "f1_score": 0.63, "roc_auc": 0.84}
        feature_cols_json = json.dumps(FEATURE_COLUMNS)
        metadata_json = json.dumps(metadata)

        # Mock open() to return different content per file path
        def open_side_effect(path, *args, **kwargs):
            path_str = str(path)
            if "feature_columns" in path_str:
                return mock_open(read_data=feature_cols_json)()
            elif "metadata" in path_str:
                return mock_open(read_data=metadata_json)()
            return mock_open()()

        with patch("serving.predict.joblib.load") as mock_joblib, \
             patch("builtins.open", side_effect=open_side_effect), \
             patch("serving.predict.ConfigLoader") as mock_config_cls:

            mock_joblib.side_effect = [mock_model, mock_scaler]
            mock_config_cls.return_value.load_all_configs.return_value = {
                "preprocessing": {
                    "target_column": "Churn",
                    "id_column": "customerID",
                    "missing_value_strategy": {},
                    "numeric_conversions": ["TotalCharges"],
                },
                "feature_engineering": {
                    "tenure_groups": {"enabled": False},
                    "derived_features": {
                        "charge_ratio": {"enabled": False},
                        "avg_monthly_charge": {"enabled": False},
                        "high_monthly_charge": {"enabled": False},
                    },
                    "service_features": {"enabled": False},
                    "indicator_features": {
                        "has_internet": {"enabled": False},
                        "has_phone": {"enabled": False},
                        "has_family": {"enabled": False},
                        "contract_risk": {"enabled": False},
                        "payment_risk": {"enabled": False},
                        "senior_alone": {"enabled": False},
                    },
                    "encoding": {
                        "binary_encode_yes_no": True,
                        "one_hot_encode_categorical": True,
                        "drop_first": True,
                    },
                },
            }
            predictor = ChurnPredictor(models_dir="fake/models", config_dir="fake/config")

        return predictor

    def test_predict_returns_required_fields(
        self, mock_rf_model, mock_scaler, sample_customer_data
    ):
        predictor = self._build_predictor(mock_rf_model, mock_scaler)
        result = predictor.predict(sample_customer_data)

        assert "prediction" in result
        assert "churn_probability" in result
        assert "risk_level" in result
        assert "model" in result
        assert "concerns" in result
        assert "retention_plan" in result

    def test_predict_high_risk(self, mock_rf_model, mock_scaler, sample_customer_data):
        mock_rf_model.predict.return_value = np.array([1])
        mock_rf_model.predict_proba.return_value = np.array([[0.2, 0.8]])

        predictor = self._build_predictor(mock_rf_model, mock_scaler)
        result = predictor.predict(sample_customer_data)
        assert result["risk_level"] == "high"

    def test_predict_low_risk(self, mock_rf_model, mock_scaler, sample_customer_data):
        mock_rf_model.predict.return_value = np.array([0])
        mock_rf_model.predict_proba.return_value = np.array([[0.85, 0.15]])

        predictor = self._build_predictor(mock_rf_model, mock_scaler)
        result = predictor.predict(sample_customer_data)
        assert result["risk_level"] == "low"

    def test_predict_medium_risk(self, mock_rf_model, mock_scaler, sample_customer_data):
        mock_rf_model.predict.return_value = np.array([1])
        mock_rf_model.predict_proba.return_value = np.array([[0.45, 0.55]])

        predictor = self._build_predictor(mock_rf_model, mock_scaler)
        result = predictor.predict(sample_customer_data)
        assert result["risk_level"] == "medium"


# --- FastAPI endpoints ---

class TestFastAPIEndpoints:
    """Test FastAPI endpoints with mocked predictor."""

    def test_health_endpoint(self):
        import serving.app as app_module

        original = app_module.predictor
        app_module.predictor = MagicMock()
        try:
            result = app_module.health()
            assert result["status"] == "ok"
            assert result["model_loaded"] is True
        finally:
            app_module.predictor = original

    def test_predict_503_when_no_model(self):
        import serving.app as app_module

        original = app_module.predictor
        app_module.predictor = None
        try:
            with pytest.raises(Exception) as exc_info:
                app_module.predict(app_module.CustomerInput())
            assert "503" in str(exc_info.value.status_code)
        finally:
            app_module.predictor = original

    def test_predict_endpoint_returns_result(self):
        import serving.app as app_module

        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = {
            "prediction": 1,
            "churn_probability": 0.72,
            "risk_level": "high",
            "model": "Random Forest",
            "concerns": [],
            "retention_plan": [],
        }

        original = app_module.predictor
        app_module.predictor = mock_predictor
        try:
            result = app_module.predict(app_module.CustomerInput())
            assert result["prediction"] == 1
            assert result["churn_probability"] == 0.72
        finally:
            app_module.predictor = original
