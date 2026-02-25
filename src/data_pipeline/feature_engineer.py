"""
Feature engineering module.
Creates derived features and performs feature transformations.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ChurnFeatureEngineer:
    """Create engineered features for churn prediction."""

    def __init__(self, config: dict):
        """
        Initialize FeatureEngineer.

        Args:
            config: Feature engineering configuration dictionary
        """
        self.config = config['feature_engineering']

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.

        Args:
            df: Preprocessed DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("\n" + "="*80)
        logger.info("FEATURE ENGINEERING")
        logger.info("="*80)

        df = df.copy()

        # Create tenure groups
        df = self._create_tenure_groups(df)

        # Create derived features
        df = self._create_derived_features(df)

        # Create service features
        df = self._create_service_features(df)

        # Create indicator features
        df = self._create_indicator_features(df)

        logger.info(f"\nNew features created. Total features: {df.shape[1]}")
        return df

    def _create_tenure_groups(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tenure group categories."""
        config = self.config.get('tenure_groups', {})

        if not config.get('enabled', True):
            return df

        logger.info("\n1. Creating tenure groups...")

        df['tenure_group'] = pd.cut(
            df['tenure'],
            bins=config['bins'],
            labels=config['labels']
        )

        return df

    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create calculated/derived features."""
        derived_config = self.config.get('derived_features', {})

        # Charge ratio
        if derived_config.get('charge_ratio', {}).get('enabled', True):
            logger.info("2. Creating charge ratio...")
            df['charge_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

        # Average monthly charge
        if derived_config.get('avg_monthly_charge', {}).get('enabled', True):
            logger.info("3. Creating average monthly charge...")
            df['avg_monthly_charge'] = df['TotalCharges'] / (df['tenure'] + 1)

        # High monthly charge indicator
        if derived_config.get('high_monthly_charge', {}).get('enabled', True):
            threshold_type = derived_config['high_monthly_charge'].get('threshold_type', 'median')

            if threshold_type == 'median':
                threshold = df['MonthlyCharges'].median()
            else:
                threshold = df['MonthlyCharges'].mean()

            df['high_monthly_charge'] = (df['MonthlyCharges'] > threshold).astype(int)

        return df

    def _create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create service usage count features."""
        service_config = self.config.get('service_features', {})

        if not service_config.get('enabled', True):
            return df

        logger.info("4. Creating service usage indicators...")

        service_cols = service_config['service_columns']
        exclude_vals = service_config['exclude_values']

        # Count services used (excluding 'No' and 'No internet service' / 'No phone service')
        df['services_count'] = 0
        for col in service_cols:
            if col in df.columns:
                df['services_count'] += (~df[col].isin(exclude_vals)).astype(int)

        # Individual service indicators
        df['has_internet'] = (df['InternetService'] != 'No').astype(int)
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype(int)

        return df

    def _create_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary indicator features."""
        indicator_config = self.config.get('indicator_features', {})

        logger.info("5. Creating family indicators...")

        # Has family
        if indicator_config.get('has_family', {}).get('enabled', True):
            df['has_family'] = (
                (df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')
            ).astype(int)

        logger.info("6. Creating contract risk indicator...")

        # Contract risk
        if indicator_config.get('contract_risk', {}).get('enabled', True):
            df['contract_risk'] = (df['Contract'] == 'Month-to-month').astype(int)

        logger.info("7. Creating payment risk indicator...")

        # Payment risk
        if indicator_config.get('payment_risk', {}).get('enabled', True):
            df['payment_risk'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

        # Senior alone
        if indicator_config.get('senior_alone', {}).get('enabled', True):
            df['senior_alone'] = (
                (df['SeniorCitizen'] == 1) & (df['has_family'] == 0)
            ).astype(int)

        return df


class FeatureEncoder:
    """Encode categorical features."""

    def __init__(self, config: dict):
        """
        Initialize FeatureEncoder.

        Args:
            config: Encoding configuration dictionary
        """
        self.config = config['feature_engineering']['encoding']
        self.id_column = config['preprocessing']['id_column']
        self.target_column = config['preprocessing']['target_column']
        # Set by fit_transform(), used by transform()
        self._binary_cols: List[str] = []
        self._ohe_cols: List[str] = []
        self._fitted_columns: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit encoder on training data and transform it.

        Call this on X_train; then call transform() on X_test.

        Args:
            df: Training feature DataFrame (no target column)

        Returns:
            Encoded training DataFrame
        """
        df = df.copy()

        if self.id_column in df.columns:
            df = df.drop(self.id_column, axis=1)
            logger.info("Dropped customerID")

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)

        self._binary_cols = []
        if self.config.get('binary_encode_yes_no', True):
            self._binary_cols = self._get_binary_columns(df, categorical_cols)
            df = self._binary_encode(df, self._binary_cols)

        self._ohe_cols = [col for col in categorical_cols if col not in self._binary_cols]
        if self.config.get('one_hot_encode_categorical', True):
            df = self._one_hot_encode(df, self._ohe_cols)

        self._fitted_columns = df.columns.tolist()
        logger.info(f"Feature encoder fitted. Final feature count: {len(self._fitted_columns)}")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using the fitted encoder.

        Must call fit_transform() before calling this.

        Args:
            df: Feature DataFrame to transform (no target column)

        Returns:
            Encoded DataFrame aligned to training columns
        """
        df = df.copy()

        if self.id_column in df.columns:
            df = df.drop(self.id_column, axis=1)

        df = self._binary_encode(df, self._binary_cols)

        if self._ohe_cols:
            df = self._one_hot_encode(df, self._ohe_cols)

        # Align columns to training set (handles unseen categories gracefully)
        df = df.reindex(columns=self._fitted_columns, fill_value=0)
        return df

    def encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode all categorical features.

        Args:
            df: DataFrame with raw categorical features

        Returns:
            DataFrame with encoded features
        """
        logger.info("\n" + "="*80)
        logger.info("ENCODING CATEGORICAL FEATURES")
        logger.info("="*80)

        df = df.copy()

        # Drop ID column
        if self.id_column in df.columns:
            df = df.drop(self.id_column, axis=1)
            logger.info("Dropped customerID")

        # Get categorical columns (excluding target)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)

        logger.info(f"\nCategorical columns to encode: {len(categorical_cols)}")

        # Binary encoding for Yes/No columns
        binary_cols = []
        if self.config.get('binary_encode_yes_no', True):
            binary_cols = self._get_binary_columns(df, categorical_cols)
            df = self._binary_encode(df, binary_cols)

        # One-hot encoding for remaining categorical columns
        if self.config.get('one_hot_encode_categorical', True):
            remaining_categorical = [col for col in categorical_cols if col not in binary_cols]
            df = self._one_hot_encode(df, remaining_categorical)

        logger.info(f"\nFinal feature count: {df.shape[1]}")
        return df

    def _get_binary_columns(self, df: pd.DataFrame, categorical_cols: List[str]) -> List[str]:
        """Identify binary Yes/No columns."""
        binary_cols = []

        for col in categorical_cols:
            unique_vals = df[col].unique()
            if len(unique_vals) == 2 and set(unique_vals).issubset({'Yes', 'No'}):
                binary_cols.append(col)

        return binary_cols

    def _binary_encode(self, df: pd.DataFrame, binary_cols: List[str]) -> pd.DataFrame:
        """Binary encode Yes/No columns."""
        for col in binary_cols:
            df[col] = (df[col] == 'Yes').astype(int)

        logger.info(f"Binary encoded columns: {len(binary_cols)}")
        return df

    def _one_hot_encode(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """One-hot encode categorical columns."""
        if not categorical_cols:
            return df

        logger.info(f"One-hot encoding columns: {categorical_cols}")

        drop_first = self.config.get('drop_first', True)
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=drop_first)

        return df
