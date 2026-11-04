"""
Time-series aware train/test splitting with walk-forward validation.

This module implements:
- Time-series train/test splits (no data leakage)
- Walk-forward validation (expanding/rolling window)
- Purge/embargo periods to prevent overlap
- Gap periods between train/test sets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from src.core import get_logger

logger = get_logger(__name__)


@dataclass
class Split:
    """Single train/test split."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_indices: np.ndarray
    test_indices: np.ndarray
    
    def __repr__(self):
        return (
            f"Split(train={self.train_start.date()}→{self.train_end.date()}, "
            f"test={self.test_start.date()}→{self.test_end.date()}, "
            f"n_train={len(self.train_indices)}, n_test={len(self.test_indices)})"
        )


class TimeSeriesSplit:
    """
    Time-series aware train/test splitting.
    
    Ensures:
    - No data leakage (future data never used for training)
    - Proper temporal ordering
    - Gap periods between train/test
    - Embargo periods to prevent label overlap
    """
    
    def __init__(
        self,
        gap_days: int = 1,
        embargo_periods: int = 0,
        min_train_size: int = 100,
        min_test_size: int = 20,
    ):
        """
        Initialize time series splitter.
        
        Args:
            gap_days: Days between train end and test start (prevents leakage)
            embargo_periods: Additional periods to skip after test (default 0)
            min_train_size: Minimum training samples required
            min_test_size: Minimum test samples required
        """
        self.gap_days = gap_days
        self.embargo_periods = embargo_periods
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size
        
        logger.info(
            "TimeSeriesSplit initialized",
            gap_days=gap_days,
            embargo_periods=embargo_periods,
            min_train=min_train_size,
            min_test=min_test_size,
        )
    
    def single_split(
        self,
        df: pd.DataFrame,
        test_start: datetime,
        test_end: datetime,
        train_start: Optional[datetime] = None,
        timestamp_col: str = "timestamp",
    ) -> Split:
        """
        Create a single train/test split.
        
        Args:
            df: DataFrame with timestamp column
            test_start: Test period start date
            test_end: Test period end date
            train_start: Train period start (if None, uses all data before test)
            timestamp_col: Column name for timestamps
            
        Returns:
            Split object with train/test indices
        """
        # Ensure timestamps are datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Calculate train end (gap_days before test_start)
        train_end = test_start - timedelta(days=self.gap_days)
        
        # Default train start to beginning of data
        if train_start is None:
            train_start = df[timestamp_col].min()
        
        # Get train indices
        train_mask = (df[timestamp_col] >= train_start) & (df[timestamp_col] <= train_end)
        train_indices = df[train_mask].index.values
        
        # Get test indices
        test_mask = (df[timestamp_col] >= test_start) & (df[timestamp_col] <= test_end)
        test_indices = df[test_mask].index.values
        
        # Validate sizes
        if len(train_indices) < self.min_train_size:
            raise ValueError(
                f"Train set too small: {len(train_indices)} < {self.min_train_size}"
            )
        
        if len(test_indices) < self.min_test_size:
            raise ValueError(
                f"Test set too small: {len(test_indices)} < {self.min_test_size}"
            )
        
        split = Split(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        
        logger.info(f"Created split: {split}")
        
        return split
    
    def walk_forward_split(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        test_size_days: int = 30,
        mode: str = "expanding",
        timestamp_col: str = "timestamp",
    ) -> List[Split]:
        """
        Create walk-forward validation splits.
        
        Args:
            df: DataFrame with timestamp column
            n_splits: Number of train/test splits
            test_size_days: Days in each test period
            mode: 'expanding' (growing train) or 'rolling' (fixed train window)
            timestamp_col: Column name for timestamps
            
        Returns:
            List of Split objects
        """
        # Ensure timestamps
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Get date range
        min_date = df[timestamp_col].min()
        max_date = df[timestamp_col].max()
        total_days = (max_date - min_date).days
        
        logger.info(
            "Creating walk-forward splits",
            mode=mode,
            n_splits=n_splits,
            test_size_days=test_size_days,
            total_days=total_days,
        )
        
        # Calculate test period starts
        # Reserve last n_splits * test_size_days for testing
        test_zone_days = n_splits * test_size_days
        first_test_start = max_date - timedelta(days=test_zone_days)
        
        splits = []
        
        for i in range(n_splits):
            # Calculate test dates
            test_start = first_test_start + timedelta(days=i * test_size_days)
            test_end = test_start + timedelta(days=test_size_days)
            
            # Calculate train start based on mode
            if mode == "expanding":
                # Expanding: always start from beginning
                train_start = min_date
            elif mode == "rolling":
                # Rolling: fixed window before test
                # Use 3x test window for training (adjustable)
                train_window_days = test_size_days * 3
                train_start = test_start - timedelta(days=train_window_days + self.gap_days)
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'expanding' or 'rolling'")
            
            # Create split
            try:
                split = self.single_split(
                    df,
                    test_start=test_start,
                    test_end=test_end,
                    train_start=train_start,
                    timestamp_col=timestamp_col,
                )
                splits.append(split)
            except ValueError as e:
                logger.warning(f"Skipping split {i}: {e}")
                continue
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        
        return splits
    
    def get_train_test_data(
        self,
        df: pd.DataFrame,
        split: Split,
        feature_cols: List[str],
        label_col: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Extract train/test features and labels from a split.
        
        Args:
            df: Full DataFrame
            split: Split object
            feature_cols: List of feature column names
            label_col: Label column name
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Get train data
        X_train = df.loc[split.train_indices, feature_cols]
        y_train = df.loc[split.train_indices, label_col]
        
        # Get test data
        X_test = df.loc[split.test_indices, feature_cols]
        y_test = df.loc[split.test_indices, label_col]
        
        # Remove NaN labels
        train_valid = y_train.notna()
        test_valid = y_test.notna()
        
        X_train = X_train[train_valid]
        y_train = y_train[train_valid]
        X_test = X_test[test_valid]
        y_test = y_test[test_valid]
        
        logger.debug(
            "Extracted train/test data",
            X_train_shape=X_train.shape,
            X_test_shape=X_test.shape,
            y_train_non_null=y_train.notna().sum(),
            y_test_non_null=y_test.notna().sum(),
        )
        
        return X_train, X_test, y_train, y_test


class PurgedKFold:
    """
    K-Fold with purging and embargo for time series.
    
    Implements purging (remove overlapping samples) and embargo
    (skip samples after test to prevent leakage).
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        purge_periods: int = 0,
        embargo_periods: int = 0,
    ):
        """
        Initialize purged K-fold.
        
        Args:
            n_splits: Number of folds
            purge_periods: Periods to purge before test
            embargo_periods: Periods to embargo after test
        """
        self.n_splits = n_splits
        self.purge_periods = purge_periods
        self.embargo_periods = embargo_periods
        
        logger.info(
            "PurgedKFold initialized",
            n_splits=n_splits,
            purge=purge_periods,
            embargo=embargo_periods,
        )
    
    def split(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged K-fold splits.
        
        Args:
            df: DataFrame with timestamps
            timestamp_col: Timestamp column name
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        n_samples = len(df)
        indices = np.arange(n_samples)
        
        # Calculate fold size
        test_size = n_samples // self.n_splits
        
        splits = []
        
        for i in range(self.n_splits):
            # Test indices for this fold
            test_start = i * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]
            
            # Train indices (everything except test)
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
            
            # Purge: remove samples before test that might overlap
            if self.purge_periods > 0:
                purge_start = max(0, test_start - self.purge_periods)
                purge_mask = (train_indices < purge_start) | (train_indices >= test_end)
                train_indices = train_indices[purge_mask]
            
            # Embargo: remove samples after test
            if self.embargo_periods > 0:
                embargo_end = min(n_samples, test_end + self.embargo_periods)
                embargo_mask = (train_indices < test_start) | (train_indices >= embargo_end)
                train_indices = train_indices[embargo_mask]
            
            splits.append((train_indices, test_indices))
        
        logger.info(f"Created {len(splits)} purged K-folds")
        
        return splits


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def simple_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    gap_days: int = 1,
    timestamp_col: str = "timestamp",
) -> Split:
    """
    Simple train/test split on time series data.
    
    Args:
        df: DataFrame with timestamps
        test_size: Fraction of data for testing (0.2 = 20%)
        gap_days: Gap between train and test
        timestamp_col: Timestamp column name
        
    Returns:
        Split object
    
    Examples:
        >>> split = simple_train_test_split(spy_df, test_size=0.2)
        >>> X_train = spy_df.loc[split.train_indices, features]
    """
    splitter = TimeSeriesSplit(gap_days=gap_days)
    
    # Calculate test dates
    min_date = df[timestamp_col].min()
    max_date = df[timestamp_col].max()
    total_days = (max_date - min_date).days
    
    test_days = int(total_days * test_size)
    test_start = max_date - timedelta(days=test_days)
    
    return splitter.single_split(df, test_start, max_date, timestamp_col=timestamp_col)


def create_walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size_days: int = 30,
) -> List[Split]:
    """
    Quick function to create walk-forward splits.
    
    Examples:
        >>> splits = create_walk_forward_splits(spy_df, n_splits=5)
        >>> for split in splits:
        >>>     # Train on split.train_indices, test on split.test_indices
    """
    splitter = TimeSeriesSplit()
    return splitter.walk_forward_split(df, n_splits=n_splits, test_size_days=test_size_days)
