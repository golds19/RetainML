"""Tests for data_pipeline.validators."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_pipeline.validators import DataValidator, ValidationResult


# --- ValidationResult ---

class TestValidationResult:
    def test_starts_passing(self):
        r = ValidationResult()
        assert r.passed is True
        assert r.errors == []
        assert r.warnings == []

    def test_add_error_fails(self):
        r = ValidationResult()
        r.add_error("bad")
        assert r.passed is False
        assert len(r.errors) == 1

    def test_add_warning_stays_passing(self):
        r = ValidationResult()
        r.add_warning("caution")
        assert r.passed is True
        assert len(r.warnings) == 1


# --- validate_raw_data ---

class TestValidateRawData:
    def test_valid_data_passes(self, sample_raw_df, validation_config):
        v = DataValidator(validation_config)
        result = v.validate_raw_data(sample_raw_df)
        assert result.passed is True
        assert result.errors == []

    def test_missing_columns_fails(self, sample_raw_df, validation_config):
        df = sample_raw_df.drop(columns=["tenure"])
        v = DataValidator(validation_config)
        result = v.validate_raw_data(df)
        assert result.passed is False
        assert any("tenure" in e for e in result.errors)

    def test_duplicate_ids_fails(self, sample_raw_df, validation_config):
        df = sample_raw_df.copy()
        df.loc[0, "customerID"] = df.loc[1, "customerID"]
        v = DataValidator(validation_config)
        result = v.validate_raw_data(df)
        assert result.passed is False
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_bad_target_values_fails(self, sample_raw_df, validation_config):
        df = sample_raw_df.copy()
        df.loc[0, "Churn"] = "Maybe"
        v = DataValidator(validation_config)
        result = v.validate_raw_data(df)
        assert result.passed is False
        assert any("unexpected" in e.lower() for e in result.errors)

    def test_low_row_count_fails(self, validation_config):
        df = pd.DataFrame({
            "customerID": ["A", "B"],
            "Churn": ["Yes", "No"],
        })
        validation_config["validation"]["min_rows"] = 100
        v = DataValidator(validation_config)
        result = v.validate_raw_data(df)
        assert result.passed is False
        assert any("row count" in e.lower() for e in result.errors)

    def test_high_missing_pct_warns(self, sample_raw_df, validation_config):
        df = sample_raw_df.copy()
        # Set >5% of a column to NaN (2 out of 20 = 10%)
        df.loc[:1, "tenure"] = np.nan
        v = DataValidator(validation_config)
        result = v.validate_raw_data(df)
        assert any("missing" in w.lower() for w in result.warnings)

    def test_unexpected_categorical_warns(self, sample_raw_df, validation_config):
        df = sample_raw_df.copy()
        df.loc[0, "Contract"] = "Weekly"
        v = DataValidator(validation_config)
        result = v.validate_raw_data(df)
        assert any("Contract" in w for w in result.warnings)


# --- validate_processed_data ---

class TestValidateProcessedData:
    def _make_processed_df(self):
        return pd.DataFrame({
            "tenure": [12, 24, 36],
            "MonthlyCharges": [50.0, 70.0, 90.0],
            "Churn": [0, 1, 0],
        })

    def test_valid_processed_passes(self, validation_config):
        v = DataValidator(validation_config)
        result = v.validate_processed_data(self._make_processed_df())
        assert result.passed is True

    def test_nulls_after_preprocessing_warns(self, validation_config):
        df = self._make_processed_df()
        df.loc[0, "tenure"] = np.nan
        v = DataValidator(validation_config)
        result = v.validate_processed_data(df)
        assert any("null" in w.lower() for w in result.warnings)

    def test_target_not_binary_fails(self, validation_config):
        df = self._make_processed_df()
        df.loc[0, "Churn"] = 5
        v = DataValidator(validation_config)
        result = v.validate_processed_data(df)
        assert result.passed is False
        assert any("0/1" in e for e in result.errors)

    def test_infinite_values_fails(self, validation_config):
        df = self._make_processed_df()
        df.loc[0, "MonthlyCharges"] = np.inf
        v = DataValidator(validation_config)
        result = v.validate_processed_data(df)
        assert result.passed is False
        assert any("infinite" in e.lower() for e in result.errors)

    def test_out_of_range_warns(self, validation_config):
        df = self._make_processed_df()
        df.loc[0, "tenure"] = -5
        v = DataValidator(validation_config)
        result = v.validate_processed_data(df)
        assert any("below" in w.lower() for w in result.warnings)
