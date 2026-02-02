"""
Data validation for the churn prediction pipeline.
Lightweight checks using pandas — no external validation frameworks.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation run."""
    passed: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def log_summary(self):
        if self.passed:
            logger.info("Validation PASSED")
        else:
            logger.error("Validation FAILED")
        for e in self.errors:
            logger.error(f"  ERROR: {e}")
        for w in self.warnings:
            logger.warning(f"  WARNING: {w}")


class DataValidator:
    """Validates data at key pipeline stages."""

    def __init__(self, config: dict):
        self.config = config
        self.rules = config.get('validation', {})

    def validate_raw_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate raw data right after loading.

        Checks: required columns, row count, ID uniqueness,
        target values, missing percentages, categorical values.
        """
        result = ValidationResult()

        self._check_row_count(df, result)
        self._check_required_columns(df, result)
        self._check_id_unique(df, result)
        self._check_target_values(df, result)
        self._check_missing_values(df, result)
        self._check_categorical_values(df, result)

        logger.info("--- Raw Data Validation ---")
        result.log_summary()
        return result

    def validate_processed_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate data after preprocessing and target encoding.

        Checks: no nulls, target is 0/1, numeric ranges, no infinities.
        """
        result = ValidationResult()

        self._check_no_nulls(df, result)
        self._check_target_numeric(df, result)
        self._check_numeric_ranges(df, result)
        self._check_no_infinite(df, result)

        logger.info("--- Processed Data Validation ---")
        result.log_summary()
        return result

    # --- Raw data checks ---

    def _check_row_count(self, df: pd.DataFrame, result: ValidationResult):
        min_rows = self.rules.get('min_rows', 100)
        if len(df) < min_rows:
            result.add_error(f"Row count {len(df)} is below minimum {min_rows}")

    def _check_required_columns(self, df: pd.DataFrame, result: ValidationResult):
        required = self.rules.get('required_columns', [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            result.add_error(f"Missing required columns: {missing}")

    def _check_id_unique(self, df: pd.DataFrame, result: ValidationResult):
        id_col = self.rules.get('id_column', 'customerID')
        if id_col not in df.columns:
            return
        n_dupes = df[id_col].duplicated().sum()
        if n_dupes > 0:
            result.add_error(f"{id_col} has {n_dupes} duplicate values")

    def _check_target_values(self, df: pd.DataFrame, result: ValidationResult):
        target_col = self.rules.get('target_column', 'Churn')
        allowed = set(self.rules.get('target_values', ['Yes', 'No']))
        if target_col not in df.columns:
            return
        actual = set(df[target_col].dropna().unique())
        unexpected = actual - allowed
        if unexpected:
            result.add_error(
                f"Target column '{target_col}' has unexpected values: {unexpected}"
            )

    def _check_missing_values(self, df: pd.DataFrame, result: ValidationResult):
        max_pct = self.rules.get('max_missing_pct', 5.0)
        for col in df.columns:
            pct = df[col].isna().mean() * 100
            if pct > max_pct:
                result.add_warning(
                    f"Column '{col}' has {pct:.1f}% missing values (threshold: {max_pct}%)"
                )

    def _check_categorical_values(self, df: pd.DataFrame, result: ValidationResult):
        cat_rules = self.rules.get('categorical_values', {})
        for col, allowed in cat_rules.items():
            if col not in df.columns:
                continue
            actual = set(df[col].dropna().unique())
            unexpected = actual - set(allowed)
            if unexpected:
                result.add_warning(
                    f"Column '{col}' has unexpected values: {unexpected}"
                )

    # --- Processed data checks ---

    def _check_no_nulls(self, df: pd.DataFrame, result: ValidationResult):
        null_counts = df.isna().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            result.add_warning(
                f"Columns with null values after preprocessing: "
                f"{dict(cols_with_nulls)}"
            )

    def _check_target_numeric(self, df: pd.DataFrame, result: ValidationResult):
        target_col = self.rules.get('target_column', 'Churn')
        if target_col not in df.columns:
            return
        unique = set(df[target_col].dropna().unique())
        if not unique.issubset({0, 1}):
            result.add_error(
                f"Target column '{target_col}' should be 0/1 after encoding, "
                f"got: {unique}"
            )

    def _check_numeric_ranges(self, df: pd.DataFrame, result: ValidationResult):
        ranges = self.rules.get('numeric_ranges', {})
        for col, bounds in ranges.items():
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min < bounds.get('min', float('-inf')):
                result.add_warning(
                    f"Column '{col}' min={col_min} is below expected {bounds['min']}"
                )
            if col_max > bounds.get('max', float('inf')):
                result.add_warning(
                    f"Column '{col}' max={col_max} is above expected {bounds['max']}"
                )

    def _check_no_infinite(self, df: pd.DataFrame, result: ValidationResult):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            n_inf = np.isinf(df[col]).sum()
            if n_inf > 0:
                result.add_error(f"Column '{col}' has {n_inf} infinite values")
