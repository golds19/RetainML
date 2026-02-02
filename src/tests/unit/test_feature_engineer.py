"""Tests for ChurnFeatureEngineer and FeatureEncoder."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data_pipeline.feature_engineer import ChurnFeatureEngineer, FeatureEncoder


class TestChurnFeatureEngineer:

    def test_create_all_features_adds_columns(self, sample_config, sample_preprocessed_df):
        engineer = ChurnFeatureEngineer(sample_config)
        result = engineer.create_all_features(sample_preprocessed_df)

        expected_cols = [
            "tenure_group", "charge_ratio", "avg_monthly_charge",
            "high_monthly_charge", "services_count", "has_internet",
            "has_phone", "has_family", "contract_risk", "payment_risk",
            "senior_alone",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_does_not_modify_original(self, sample_config, sample_preprocessed_df):
        original_cols = sample_preprocessed_df.columns.tolist()
        engineer = ChurnFeatureEngineer(sample_config)
        engineer.create_all_features(sample_preprocessed_df)
        assert sample_preprocessed_df.columns.tolist() == original_cols

    def test_tenure_groups_correct_bins(self, sample_config, sample_preprocessed_df):
        engineer = ChurnFeatureEngineer(sample_config)
        result = engineer.create_all_features(sample_preprocessed_df)

        # tenure=6 should be "0-1 year" (bin 0-12)
        row_t6 = result[result["tenure"] == 6].iloc[0]
        assert row_t6["tenure_group"] == "0-1 year"

        # tenure=36 should be "2-4 years" (bin 24-48)
        row_t36 = result[result["tenure"] == 36].iloc[0]
        assert row_t36["tenure_group"] == "2-4 years"

    def test_charge_ratio_formula(self, sample_config, sample_preprocessed_df):
        engineer = ChurnFeatureEngineer(sample_config)
        result = engineer.create_all_features(sample_preprocessed_df)

        for _, row in result.iterrows():
            expected = row["MonthlyCharges"] / (row["TotalCharges"] + 1)
            assert abs(row["charge_ratio"] - expected) < 1e-6

    def test_avg_monthly_charge_formula(self, sample_config, sample_preprocessed_df):
        engineer = ChurnFeatureEngineer(sample_config)
        result = engineer.create_all_features(sample_preprocessed_df)

        for _, row in result.iterrows():
            expected = row["TotalCharges"] / (row["tenure"] + 1)
            assert abs(row["avg_monthly_charge"] - expected) < 1e-6

    def test_high_monthly_charge_is_binary(self, sample_config, sample_preprocessed_df):
        engineer = ChurnFeatureEngineer(sample_config)
        result = engineer.create_all_features(sample_preprocessed_df)
        assert set(result["high_monthly_charge"].unique()).issubset({0, 1})

    def test_services_count_range(self, sample_config, sample_preprocessed_df):
        engineer = ChurnFeatureEngineer(sample_config)
        result = engineer.create_all_features(sample_preprocessed_df)
        assert result["services_count"].min() >= 0
        assert result["services_count"].max() <= 9

    def test_has_family_logic(self, sample_config, sample_preprocessed_df):
        engineer = ChurnFeatureEngineer(sample_config)
        result = engineer.create_all_features(sample_preprocessed_df)

        for _, row in result.iterrows():
            has_partner = row.get("Partner") == "Yes"
            has_dependents = row.get("Dependents") == "Yes"
            expected = 1 if (has_partner or has_dependents) else 0
            assert row["has_family"] == expected

    def test_contract_risk_logic(self, sample_config, sample_preprocessed_df):
        engineer = ChurnFeatureEngineer(sample_config)
        result = engineer.create_all_features(sample_preprocessed_df)

        for _, row in result.iterrows():
            expected = 1 if row["Contract"] == "Month-to-month" else 0
            assert row["contract_risk"] == expected

    def test_senior_alone_logic(self, sample_config, sample_preprocessed_df):
        engineer = ChurnFeatureEngineer(sample_config)
        result = engineer.create_all_features(sample_preprocessed_df)

        for _, row in result.iterrows():
            expected = 1 if (row["SeniorCitizen"] == 1 and row["has_family"] == 0) else 0
            assert row["senior_alone"] == expected


class TestFeatureEncoder:

    def test_drops_customer_id(self, sample_config, sample_preprocessed_df):
        encoder = FeatureEncoder(sample_config)
        result = encoder.encode_features(sample_preprocessed_df)
        assert "customerID" not in result.columns

    def test_binary_encoding(self, sample_config, sample_preprocessed_df):
        encoder = FeatureEncoder(sample_config)
        result = encoder.encode_features(sample_preprocessed_df)

        # PaperlessBilling was Yes/No, should now be 0/1
        if "PaperlessBilling" in result.columns:
            assert set(result["PaperlessBilling"].unique()).issubset({0, 1})

    def test_no_object_columns_remain(self, sample_config, sample_preprocessed_df):
        encoder = FeatureEncoder(sample_config)
        result = encoder.encode_features(sample_preprocessed_df)

        # After encoding, no object/category columns should remain (except possibly target)
        obj_cols = result.select_dtypes(include=["object", "category"]).columns.tolist()
        assert len(obj_cols) == 0, f"Object columns remain: {obj_cols}"

    def test_does_not_modify_original(self, sample_config, sample_preprocessed_df):
        original_cols = sample_preprocessed_df.columns.tolist()
        encoder = FeatureEncoder(sample_config)
        encoder.encode_features(sample_preprocessed_df)
        assert sample_preprocessed_df.columns.tolist() == original_cols
