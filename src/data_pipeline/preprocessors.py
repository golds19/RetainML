"""
Data preprocessing module.
Handles data cleaning, missing value imputation, and type conversions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess and clean raw data."""

    def __init__(self, config: dict):
        """
        Initialize DataPreprocessor.

        Args:
            config: Preprocessing configuration dictionary
        """
        self.config = config
        self.preprocessing_config = config['preprocessing']

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all preprocessing steps.

        Args:
            df: Raw DataFrame

        Returns:
            Preprocessed DataFrame
        """
        logger.info("\n" + "="*80)
        logger.info("DATA PREPROCESSING")
        logger.info("="*80)

        df = df.copy()

        # Convert data types
        df = self._convert_types(df)

        # Handle missing values
        df = self._handle_missing_values(df)

        logger.info("Data preprocessing completed")
        logger.info(f"Final shape: {df.shape}")
        return df

    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert specified columns to numeric types."""
        logger.info("Converting data types")

        numeric_cols = self.preprocessing_config.get('numeric_conversions', [])

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                logger.debug(f"Converted {col} to numeric")

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values according to configuration."""
        logger.info("Handling missing values")

        missing_strategy = self.preprocessing_config.get('missing_value_strategy', {})

        for col, strategy in missing_strategy.items():
            if col not in df.columns:
                continue

            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue

            logger.info(f"Missing values in {col}: {missing_count}")

            if strategy['method'] == 'fill_with_column':
                fill_col = strategy['fill_column']
                df[col] = df[col].fillna(df[fill_col])
                logger.info(f"Filled {col} missing values with {fill_col}")

        return df


class TargetEncoder:
    """Encode target variable."""

    def __init__(self, config: dict):
        """
        Initialize TargetEncoder.

        Args:
            config: Configuration dictionary
        """
        self.target_col = config['preprocessing']['target_column']

    def encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode binary target variable (Yes/No to 1/0).

        Args:
            df: DataFrame with target column

        Returns:
            DataFrame with encoded target
        """
        df = df.copy()

        if self.target_col in df.columns:
            df[self.target_col] = (df[self.target_col] == 'Yes').astype(int)
            logger.info(f"Encoded target variable: {self.target_col}")

        return df
