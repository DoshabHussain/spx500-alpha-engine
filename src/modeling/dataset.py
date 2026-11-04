"""
Dataset preparation and train/test splitting for time-series ML.

Critical for financial ML:
- Time-series aware splitting (no data leakage)
- Purging and embargo periods
- Walk-forward validation support
- Feature/label separation
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from datetime import datetime, timedelta

from src.core import get_logger

logger = get_logger(__name__)


class TimeSeriesDataset:
    """Time-series aware dataset preparation."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        timestamp_col: str = "timestamp",
    ):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with features and labels
            feature_cols: List of feature column names
            label_col: Target label column name
            timestamp_col: Timestamp column name
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.timestamp_col = timestamp_col
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[timestamp_col]):
            self.df[timestamp_col] = pd.to_datetime(self.df[timestamp_col])
        
        # Sort by timestamp
        self.df = self.df.sort_values(timestamp_col).reset_index(drop=True)
        
        logger.info(
            "Dataset initialized",
            total_rows=len(df),
            features=len(feature_cols),
            label=label_col,
        )
    
    def train_test_split(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        gap_days: int = 1,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Time-series aware train/validation/test split.
        
        CRITICAL: Always splits chronologically (no shuffling).
        Includes gap period to prevent leakage.
        
        Args:
            test_size: Fraction of data for test set (most recent)
            validation_size: Fraction of data for validation set
            gap_days: Gap in days between train and validation/test
            
        Returns:
            (train_df, val_df, test_df)
        """
        n = len(self.df)
        
        # Calculate split indices
        test_start_idx = int(n * (1 - test_size))
        val_start_idx = int(n * (1 - test_size - validation_size))
        
        # Get timestamps for gap calculation
        val_gap_date = self.df.iloc[val_start_idx][self.timestamp_col] - timedelta(days=gap_days)
        test_gap_date = self.df.iloc[test_start_idx][self.timestamp_col] - timedelta(days=gap_days)
        
        # Apply gaps
        train_df = self.df[self.df[self.timestamp_col] <= val_gap_date].copy()
        val_df = self.df[
            (self.df[self.timestamp_col] >= self.df.iloc[val_start_idx][self.timestamp_col]) &
            (self.df[self.timestamp_col] <= test_gap_date)
        ].copy()
        test_df = self.df[
            self.df[self.timestamp_col] >= self.df.iloc[test_start_idx][self.timestamp_col]
        ].copy()
        
        logger.info(
            "Train/val/test split complete",
            train_rows=len(train_df),
            val_rows=len(val_df),
            test_rows=len(test_df),
            train_period=f"{train_df[self.timestamp_col].min()} to {train_df[self.timestamp_col].max()}",
            test_period=f"{test_df[self.timestamp_col].min()} to {test_df[self.timestamp_col].max()}",
        )
        
        return train_df, val_df, test_df
    
    def get_X_y(
        self,
        df: pd.DataFrame,
        drop_na: bool = True,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features (X) and labels (y) from DataFrame.
        
        Args:
            df: Input DataFrame
            drop_na: Whether to drop rows with NaN in features or labels
            
        Returns:
            (X, y) where X is features and y is labels
        """
        X = df[self.feature_cols].copy()
        y = df[self.label_col].copy()
        
        if drop_na:
            # Drop rows where label is NaN
            valid_mask = y.notna()
            
            # Also drop rows with any NaN features
            valid_mask &= X.notna().all(axis=1)
            
            X = X[valid_mask]
            y = y[valid_mask]
            
            dropped = len(df) - len(X)
            if dropped > 0:
                logger.debug(f"Dropped {dropped} rows with NaN values")
        
        return X, y
    
    def get_splits(
        self,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        gap_days: int = 1,
    ) -> Tuple[Tuple[pd.DataFrame, pd.Series], 
               Tuple[pd.DataFrame, pd.Series], 
               Tuple[pd.DataFrame, pd.Series]]:
        """
        Get train/val/test splits as (X, y) tuples.
        
        Returns:
            ((X_train, y_train), (X_val, y_val), (X_test, y_test))
        """
        train_df, val_df, test_df = self.train_test_split(
            test_size=test_size,
            validation_size=validation_size,
            gap_days=gap_days,
        )
        
        X_train, y_train = self.get_X_y(train_df)
        X_val, y_val = self.get_X_y(val_df)
        X_test, y_test = self.get_X_y(test_df)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


class WalkForwardValidator:
    """
    Walk-forward validation for time-series models.
    
    Simulates realistic backtesting:
    - Train on historical window
    - Predict on next period
    - Roll window forward
    - Repeat
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        timestamp_col: str = "timestamp",
    ):
        """Initialize walk-forward validator."""
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.timestamp_col = timestamp_col
        
        # Ensure sorted by time
        self.df = self.df.sort_values(timestamp_col).reset_index(drop=True)
        
        logger.info("WalkForwardValidator initialized", total_rows=len(df))
    
    def generate_splits(
        self,
        train_window_days: int = 60,
        test_window_days: int = 5,
        step_days: int = 5,
        gap_days: int = 1,
    ):
        """
        Generate walk-forward splits.
        
        Args:
            train_window_days: Size of training window
            test_window_days: Size of test window
            step_days: How many days to step forward each iteration
            gap_days: Gap between train and test
            
        Yields:
            (train_df, test_df, fold_num)
        """
        min_date = self.df[self.timestamp_col].min()
        max_date = self.df[self.timestamp_col].max()
        
        current_date = min_date + timedelta(days=train_window_days)
        fold_num = 0
        
        while current_date + timedelta(days=test_window_days + gap_days) <= max_date:
            # Training window
            train_start = current_date - timedelta(days=train_window_days)
            train_end = current_date - timedelta(days=gap_days)
            
            # Test window
            test_start = current_date
            test_end = current_date + timedelta(days=test_window_days)
            
            # Filter data
            train_df = self.df[
                (self.df[self.timestamp_col] >= train_start) &
                (self.df[self.timestamp_col] <= train_end)
            ].copy()
            
            test_df = self.df[
                (self.df[self.timestamp_col] >= test_start) &
                (self.df[self.timestamp_col] <= test_end)
            ].copy()
            
            if len(train_df) > 0 and len(test_df) > 0:
                fold_num += 1
                logger.debug(
                    f"Fold {fold_num}",
                    train_rows=len(train_df),
                    test_rows=len(test_df),
                    train_period=f"{train_start.date()} to {train_end.date()}",
                    test_period=f"{test_start.date()} to {test_end.date()}",
                )
                yield train_df, test_df, fold_num
            
            # Step forward
            current_date += timedelta(days=step_days)
        
        logger.info(f"Walk-forward validation complete: {fold_num} folds")


def select_features(
    df: pd.DataFrame,
    label_col: str,
    exclude_patterns: List[str] = None,
    include_only: List[str] = None,
) -> List[str]:
    """
    Intelligently select feature columns.
    
    Args:
        df: DataFrame with potential features
        label_col: Label column to exclude
        exclude_patterns: Patterns to exclude (e.g., ['forward_', 'label_'])
        include_only: If provided, only include these columns
        
    Returns:
        List of feature column names
    """
    if exclude_patterns is None:
        exclude_patterns = ['forward_', 'label_', 'timestamp', 'symbol', 'date']
    
    # Start with all columns
    all_cols = df.columns.tolist()
    
    # If include_only specified, use that
    if include_only:
        feature_cols = [col for col in include_only if col in all_cols]
    else:
        # Exclude label column
        feature_cols = [col for col in all_cols if col != label_col]
        
        # Exclude based on patterns
        for pattern in exclude_patterns:
            feature_cols = [col for col in feature_cols if pattern not in col.lower()]
    
    logger.info(
        "Feature selection complete",
        total_features=len(feature_cols),
        excluded_patterns=exclude_patterns,
    )
    
    return feature_cols


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def prepare_dataset(
    df: pd.DataFrame,
    label_col: str = "label_30m",
    test_size: float = 0.2,
    validation_size: float = 0.1,
) -> Tuple[Tuple[pd.DataFrame, pd.Series], 
           Tuple[pd.DataFrame, pd.Series], 
           Tuple[pd.DataFrame, pd.Series]]:
    """
    Quick dataset preparation with automatic feature selection.
    
    Examples:
        >>> from src.modeling.dataset import prepare_dataset
        >>> (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataset(df)
    """
    # Auto-select features
    feature_cols = select_features(df, label_col)
    
    # Create dataset
    dataset = TimeSeriesDataset(df, feature_cols, label_col)
    
    # Get splits
    return dataset.get_splits(
        test_size=test_size,
        validation_size=validation_size,
    )
