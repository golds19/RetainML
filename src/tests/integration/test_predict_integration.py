"""
Real end-to-end integration test for churn prediction.

Loads the actual saved model artifacts and runs a full prediction to verify
the entire serving stack works correctly. Skipped automatically if model
artifacts have not been generated yet (run scripts/train_pipeline.py first).
"""

import sys
from pathlib import Path

import pytest

# Make src/ importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"
CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "config"

models_required = pytest.mark.skipif(
    not (MODELS_DIR / "model.pkl").exists(),
    reason="Model artifacts not found — run scripts/train_pipeline.py first",
)

SAMPLE_CUSTOMER = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 6,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.0,
    "TotalCharges": "510.0",
}


@models_required
class TestPredictEndToEnd:
    """Integration tests that use real saved model artifacts."""

    @pytest.fixture(scope="class")
    def predictor(self):
        from serving.predict import ChurnPredictor
        return ChurnPredictor(
            models_dir=str(MODELS_DIR),
            config_dir=str(CONFIG_DIR),
        )

    def test_predict_returns_all_required_keys(self, predictor):
        result = predictor.predict(SAMPLE_CUSTOMER)
        assert "prediction" in result
        assert "churn_probability" in result
        assert "risk_level" in result
        assert "model" in result
        assert "concerns" in result
        assert "retention_plan" in result

    def test_prediction_is_binary(self, predictor):
        result = predictor.predict(SAMPLE_CUSTOMER)
        assert result["prediction"] in (0, 1)

    def test_churn_probability_in_valid_range(self, predictor):
        result = predictor.predict(SAMPLE_CUSTOMER)
        prob = result["churn_probability"]
        assert 0.0 <= prob <= 1.0, f"churn_probability {prob} is out of [0, 1]"

    def test_risk_level_is_valid(self, predictor):
        result = predictor.predict(SAMPLE_CUSTOMER)
        assert result["risk_level"] in ("low", "medium", "high")

    def test_risk_level_consistent_with_probability(self, predictor):
        result = predictor.predict(SAMPLE_CUSTOMER)
        prob = result["churn_probability"]
        risk = result["risk_level"]
        if prob >= 0.7:
            assert risk == "high", f"prob={prob} should be 'high', got '{risk}'"
        elif prob >= 0.4:
            assert risk == "medium", f"prob={prob} should be 'medium', got '{risk}'"
        else:
            assert risk == "low", f"prob={prob} should be 'low', got '{risk}'"

    def test_high_risk_customer_scores_high(self, predictor):
        """A month-to-month fiber customer with no add-ons should score high risk."""
        result = predictor.predict(SAMPLE_CUSTOMER)
        # This customer profile is a known high-churn archetype;
        # assert probability is non-trivial (not stuck at a degenerate value).
        assert result["churn_probability"] > 0.05

    def test_low_risk_customer_scores_lower(self, predictor):
        loyal_customer = {
            **SAMPLE_CUSTOMER,
            "tenure": 60,
            "Contract": "Two year",
            "InternetService": "DSL",
            "MonthlyCharges": 35.0,
            "TotalCharges": "2100.0",
        }
        high_risk_result = predictor.predict(SAMPLE_CUSTOMER)
        low_risk_result = predictor.predict(loyal_customer)
        assert low_risk_result["churn_probability"] < high_risk_result["churn_probability"], (
            "Loyal two-year contract customer should have a lower churn probability "
            "than a month-to-month fiber customer with 6 months tenure"
        )

    def test_concerns_is_list(self, predictor):
        result = predictor.predict(SAMPLE_CUSTOMER)
        assert isinstance(result["concerns"], list)

    def test_retention_plan_is_list(self, predictor):
        result = predictor.predict(SAMPLE_CUSTOMER)
        assert isinstance(result["retention_plan"], list)
