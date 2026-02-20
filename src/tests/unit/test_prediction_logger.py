"""
Unit tests for PredictionLogger.
Uses pytest's tmp_path fixture so no real data directory is touched.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from monitoring.prediction_logger import PredictionLogger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_INPUTS = {
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
    "gender": "Male",
}


def _log_one(logger: PredictionLogger, **overrides) -> None:
    kwargs = dict(
        inputs=SAMPLE_INPUTS,
        prediction=1,
        churn_probability=0.72,
        risk_level="high",
        model_name="Random Forest",
    )
    kwargs.update(overrides)
    logger.log(**kwargs)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInit:

    def test_creates_db_file(self, tmp_path):
        PredictionLogger(data_dir=tmp_path)
        assert (tmp_path / "predictions.db").exists()

    def test_creates_nested_data_dir(self, tmp_path):
        nested = tmp_path / "nested" / "subdir"
        PredictionLogger(data_dir=nested)
        assert nested.exists()
        assert (nested / "predictions.db").exists()

    def test_init_is_idempotent(self, tmp_path):
        PredictionLogger(data_dir=tmp_path)
        PredictionLogger(data_dir=tmp_path)
        assert (tmp_path / "predictions.db").exists()


# ---------------------------------------------------------------------------
# log()
# ---------------------------------------------------------------------------

class TestLog:

    def test_log_inserts_record(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        _log_one(pl)
        assert len(pl.get_recent()) == 1

    def test_log_stores_correct_fields(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        _log_one(pl, prediction=0, churn_probability=0.15, risk_level="low")
        record = pl.get_recent()[0]
        assert record["prediction"] == 0
        assert record["churn_probability"] == pytest.approx(0.15)
        assert record["risk_level"] == "low"
        assert record["model_name"] == "Random Forest"

    def test_log_stores_all_input_fields(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        _log_one(pl)
        assert pl.get_recent()[0]["inputs"] == SAMPLE_INPUTS

    def test_log_timestamp_is_iso_string(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        _log_one(pl)
        datetime.fromisoformat(pl.get_recent()[0]["timestamp"])

    def test_log_multiple_records(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        for _ in range(5):
            _log_one(pl)
        assert len(pl.get_recent()) == 5

    def test_log_does_not_raise_on_db_error(self, tmp_path, monkeypatch):
        pl = PredictionLogger(data_dir=tmp_path)
        def bad_connect():
            raise sqlite3.OperationalError("disk full")
        monkeypatch.setattr(pl, "_get_connection", bad_connect)
        _log_one(pl)  # must not raise


# ---------------------------------------------------------------------------
# get_recent()
# ---------------------------------------------------------------------------

class TestGetRecent:

    def test_empty_db_returns_empty_list(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        assert pl.get_recent() == []

    def test_default_limit_is_50(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        for _ in range(60):
            _log_one(pl)
        assert len(pl.get_recent()) == 50

    def test_custom_limit(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        for _ in range(10):
            _log_one(pl)
        assert len(pl.get_recent(n=3)) == 3

    def test_returns_newest_first(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        _log_one(pl, churn_probability=0.1)
        _log_one(pl, churn_probability=0.9)
        records = pl.get_recent()
        assert records[0]["churn_probability"] == pytest.approx(0.9)
        assert records[1]["churn_probability"] == pytest.approx(0.1)

    def test_record_has_expected_keys(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        _log_one(pl)
        assert set(pl.get_recent()[0].keys()) == {
            "id", "timestamp", "inputs", "prediction",
            "churn_probability", "risk_level", "model_name"
        }

    def test_inputs_is_dict_not_string(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        _log_one(pl)
        assert isinstance(pl.get_recent()[0]["inputs"], dict)

    def test_n_zero_returns_empty(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        _log_one(pl)
        assert pl.get_recent(n=0) == []

    def test_limit_larger_than_rows(self, tmp_path):
        pl = PredictionLogger(data_dir=tmp_path)
        for _ in range(3):
            _log_one(pl)
        assert len(pl.get_recent(n=100)) == 3
