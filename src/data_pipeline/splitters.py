"""
Data splitting and scaling module.
Handles train-test splits and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Split data into train and test sets."""

    def __init__(self, config: dict):
        """
        Initialize DataSplitter.

        Args:
            config: Data configuration dictionary
        """
        self.config = config['preprocessing']

    def split_data(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.

        Args:
            df: Complete dataset

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("\n" + "="*80)
        logger.info("PREPARING DATA FOR MODELING")
        logger.info("="*80)

        target_col = self.config['target_column']

        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        logger.info(f"\nFeature matrix shape: {X.shape}")
        logger.info("Target distribution:")
        logger.info(y.value_counts())
        logger.info(f"Churn rate: {y.mean():.2%}")

        # Train-test split
        stratify = y if self.config.get('stratify', True) else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=stratify
        )

        logger.info(f"\nTrain set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test


class FeatureScaler:
    """Scale features using StandardScaler or MinMaxScaler."""

    def __init__(self, config: dict):
        """
        Initialize FeatureScaler.

        Args:
            config: Data configuration dictionary
        """
        self.config = config['preprocessing']
        self.scaler = None
        self._initialize_scaler()

    def _initialize_scaler(self):
        """Initialize the appropriate scaler."""
        scaler_type = self.config.get('scaler_type', 'StandardScaler')

        if scaler_type == 'StandardScaler':
            self.scaler = StandardScaler()
        elif scaler_type == 'MinMaxScaler':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")

        logger.info(f"Initialized {scaler_type}")

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fit scaler on training data and transform both sets.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        logger.info("\nScaling features...")

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        logger.info("Feature scaling completed")
        return X_train_scaled, X_test_scaled

    def get_scaler(self):
        """Return fitted scaler for later use."""
        return self.scaler


class ImbalanceHandler:
    """Handle class imbalance using SMOTE."""

    def __init__(self, config: dict):
        """
        Initialize ImbalanceHandler.

        Args:
            config: Data configuration dictionary
        """
        self.config = config['preprocessing']

    def apply_smote(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE to balance classes.

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Tuple of (X_train_balanced, y_train_balanced)
        """
        if not self.config.get('apply_smote', False):
            logger.info("SMOTE not enabled, returning original data")
            return X_train, y_train

        logger.info("\n" + "="*80)
        logger.info("APPLYING SMOTE FOR CLASS BALANCING")
        logger.info("="*80)

        logger.info("Before SMOTE - Class distribution:")
        logger.info(y_train.value_counts())

        smote = SMOTE(random_state=self.config.get('smote_random_state', 42))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        logger.info("\nAfter SMOTE - Class distribution:")
        logger.info(pd.Series(y_train_balanced).value_counts())

        # Convert back to DataFrame/Series with proper columns
        X_train_balanced = pd.DataFrame(X_train_balanced, columns=X_train.columns)
        y_train_balanced = pd.Series(y_train_balanced, name=y_train.name)

        return X_train_balanced, y_train_balanced
