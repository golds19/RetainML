"""
Inference logic for churn prediction.
Loads saved model artifacts and runs the full preprocessing pipeline on new data.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

from data_pipeline.preprocessors import DataPreprocessor
from data_pipeline.feature_engineer import ChurnFeatureEngineer, FeatureEncoder
from utils.config_loader import ConfigLoader
from serving.explainer import ChurnExplainer

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Loads model artifacts and predicts churn for new customer data."""

    def __init__(self, models_dir: str = "models", config_dir: str = "config"):
        models_path = Path(models_dir)

        self.model = joblib.load(models_path / "model.pkl")
        self.scaler = joblib.load(models_path / "scaler.pkl")

        with open(models_path / "feature_columns.json") as f:
            self.feature_columns = json.load(f)

        with open(models_path / "metadata.json") as f:
            self.metadata = json.load(f)

        config_loader = ConfigLoader(config_dir=config_dir)
        self.config = config_loader.load_all_configs()

        self.explainer = ChurnExplainer(self.model, self.feature_columns)
        logger.info(f"Loaded model: {self.metadata['best_model']}")

    def predict(self, customer_data: dict) -> dict:
        """
        Predict churn for a single customer.

        Args:
            customer_data: Dict with raw customer fields
                (tenure, MonthlyCharges, Contract, etc.)

        Returns:
            Dict with prediction, churn_probability, risk_level
        """
        # Build single-row DataFrame
        df = pd.DataFrame([customer_data])

        # Add a dummy target column so preprocessing doesn't fail
        if "Churn" not in df.columns:
            df["Churn"] = "No"

        # Add a dummy customerID if missing
        if "customerID" not in df.columns:
            df["customerID"] = "PREDICT-001"

        # Preprocess (type conversion, missing values)
        preprocessor = DataPreprocessor(self.config)
        df = preprocessor.preprocess(df)

        # Feature engineering
        feature_engineer = ChurnFeatureEngineer(self.config)
        df = feature_engineer.create_all_features(df)

        # Feature encoding (binary + one-hot)
        feature_encoder = FeatureEncoder(self.config)
        df = feature_encoder.encode_features(df)

        # Drop target column (we don't need it for prediction)
        target_col = self.config.get("preprocessing", {}).get("target_column", "Churn")
        if target_col in df.columns:
            df = df.drop(columns=[target_col])

        # Align columns to match training feature set
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        # Scale
        df_scaled = pd.DataFrame(
            self.scaler.transform(df),
            columns=self.feature_columns,
        )

        # Predict
        prediction = int(self.model.predict(df_scaled)[0])
        proba = float(self.model.predict_proba(df_scaled)[0][1])

        if proba >= 0.7:
            risk_level = "high"
        elif proba >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Explain prediction
        explanation = self.explainer.explain(df_scaled, customer_data)

        return {
            "prediction": prediction,
            "churn_probability": round(proba, 4),
            "risk_level": risk_level,
            "model": self.metadata["best_model"],
            "concerns": explanation["concerns"],
            "retention_plan": explanation["retention_plan"],
        }
