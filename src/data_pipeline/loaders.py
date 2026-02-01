"""
Data loading and validation module.
Handles reading data from various sources and basic validation.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and validate raw data."""

    def __init__(self, config: dict):
        """
        Initialize DataLoader.

        Args:
            config: Data configuration dictionary
        """
        self.config = config

    def load_csv(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file. If None, uses config default.

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data validation fails
        """
        if file_path is None:
            file_path = self.config['data']['raw_data_path']

        logger.info("="*80)
        logger.info("LOADING DATA")
        logger.info("="*80)
        logger.info(f"Loading data from {file_path}")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")

        df = pd.read_csv(file_path)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        self._validate_data(df)

        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate loaded data.

        Args:
            df: DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        # Check for required columns
        target_col = self.config['preprocessing']['target_column']
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")

        # Check for empty dataframe
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")

        logger.info("Data validation passed")


class DataSaver:
    """Save processed data."""

    @staticmethod
    def save_processed_data(
        df: pd.DataFrame,
        output_path: str,
        file_name: str = "processed_data.csv"
    ) -> None:
        """
        Save processed DataFrame to CSV.

        Args:
            df: DataFrame to save
            output_path: Output directory path
            file_name: Name of output file
        """
        Path(output_path).mkdir(parents=True, exist_ok=True)
        full_path = Path(output_path) / file_name

        df.to_csv(full_path, index=False)
        logger.info(f"Processed data saved to {full_path}")
