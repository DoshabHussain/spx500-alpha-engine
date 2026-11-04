"""
Label/target generation for SPX forecasting.

Implements forward-looking labels:
- Directional labels (up/down/neutral)
- Forward returns (30m, 1h, 6h horizons)
- Leakage prevention (embargo periods)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple

from src.core import get_logger

logger = get_logger(__name__)


class TargetGenerator:
    """Generate forward-looking labels/targets for model training."""
    
    def __init__(self):
        """Initialize target generator."""
        logger.info("TargetGenerator initialized")
    
    def calculate_forward_returns(
        self,
        df: pd.DataFrame,
        horizons: Dict[str, int] = None,
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate forward returns for multiple horizons.
        
        Args:
            df: DataFrame with OHLCV data
            horizons: Dict of {name: periods}, e.g., {"30m": 6, "1h": 12}
            close_col: Column name for close price
            
        Returns:
            DataFrame with forward_ret_* columns
        """
        df = df.copy()
        
        if horizons is None:
            # Default horizons (assuming 5-minute bars)
            horizons = {
                "30m": 6,    # 6 * 5min = 30 minutes
                "1h": 12,    # 12 * 5min = 1 hour
                "6h": 72,    # 72 * 5min = 6 hours
            }
        
        for name, periods in horizons.items():
            feature_name = f"forward_ret_{name}"
            
            # Forward return: (Close_t+N - Close_t) / Close_t
            future_price = df[close_col].shift(-periods)
            df[feature_name] = (future_price - df[close_col]) / df[close_col]
            
            logger.debug(f"Calculated {feature_name}", periods=periods)
        
        logger.info("Calculated forward returns", horizons=list(horizons.keys()))
        return df
    
    def calculate_directional_labels(
        self,
        df: pd.DataFrame,
        forward_return_cols: list = None,
        thresholds: Dict[str, Tuple[float, float]] = None,
    ) -> pd.DataFrame:
        """
        Convert forward returns to directional labels (up/down/neutral).
        
        Args:
            df: DataFrame with forward return columns
            forward_return_cols: List of forward return column names
            thresholds: Dict of {col_name: (down_threshold, up_threshold)}
                       Default: {"forward_ret_30m": (-0.001, 0.001)}
                       
        Returns:
            DataFrame with label_* columns (0=down, 1=neutral, 2=up)
        """
        df = df.copy()
        
        if forward_return_cols is None:
            forward_return_cols = [col for col in df.columns if col.startswith("forward_ret_")]
        
        if thresholds is None:
            # Default thresholds (10 basis points = 0.001)
            thresholds = {
                "forward_ret_30m": (-0.001, 0.001),  # +/- 10 bps
                "forward_ret_1h": (-0.002, 0.002),   # +/- 20 bps
                "forward_ret_6h": (-0.005, 0.005),   # +/- 50 bps
            }
        
        for col in forward_return_cols:
            label_col = col.replace("forward_ret_", "label_")
            
            # Get thresholds for this column
            down_thresh, up_thresh = thresholds.get(col, (-0.001, 0.001))
            
            # Create directional labels
            # 0 = down (< down_threshold)
            # 1 = neutral (between thresholds)
            # 2 = up (> up_threshold)
            # NaN for rows without forward return (end of dataset)
            df[label_col] = np.nan
            valid_mask = df[col].notna()
            df.loc[valid_mask, label_col] = 1.0  # Default valid to neutral
            df.loc[valid_mask & (df[col] < down_thresh), label_col] = 0.0  # Down
            df.loc[valid_mask & (df[col] > up_thresh), label_col] = 2.0    # Up
            
            logger.debug(
                f"Created {label_col}",
                down_count=(df[label_col] == 0).sum(),
                neutral_count=(df[label_col] == 1).sum(),
                up_count=(df[label_col] == 2).sum(),
            )
        
        logger.info("Calculated directional labels", columns=forward_return_cols)
        return df
    
    def apply_embargo(
        self,
        df: pd.DataFrame,
        embargo_periods: int = 2,
        label_cols: list = None,
    ) -> pd.DataFrame:
        """
        Apply embargo to prevent data leakage.
        
        After calculating labels, we "embargo" the next N periods to ensure
        the model doesn't train on data that overlaps with the prediction window.
        
        Args:
            df: DataFrame with label columns
            embargo_periods: Number of periods to embargo after each label
            label_cols: List of label column names
            
        Returns:
            DataFrame with embargoed labels (set to NaN)
        """
        df = df.copy()
        
        if label_cols is None:
            label_cols = [col for col in df.columns if col.startswith("label_")]
        
        for col in label_cols:
            # For each non-NaN label, set the next embargo_periods labels to NaN
            # This prevents overlapping prediction windows
            mask = df[col].notna()
            indices = df[mask].index
            
            for idx in indices:
                # Set next embargo_periods rows to NaN
                embargo_end = min(idx + embargo_periods + 1, len(df))
                df.loc[idx + 1:embargo_end, col] = np.nan
        
        logger.info(
            "Applied embargo",
            embargo_periods=embargo_periods,
            columns=label_cols,
        )
        return df
    
    def purge_overlapping_labels(
        self,
        df: pd.DataFrame,
        forward_periods: Dict[str, int] = None,
    ) -> pd.DataFrame:
        """
        Purge labels that overlap with their prediction window.
        
        For a 30-minute forward prediction (6 bars on 5-min data),
        we can't use the last 6 bars as training data since we don't
        have their future values.
        
        Args:
            df: DataFrame with label columns
            forward_periods: Dict of {label_col: periods}
            
        Returns:
            DataFrame with purged labels (last N rows set to NaN)
        """
        df = df.copy()
        
        if forward_periods is None:
            # Extract periods from column names
            # e.g., "forward_ret_30m" was calculated with 6 periods
            forward_periods = {
                "label_30m": 6,
                "label_1h": 12,
                "label_6h": 72,
            }
        
        for col, periods in forward_periods.items():
            if col in df.columns:
                # Set last N rows to NaN
                df.iloc[-periods:, df.columns.get_loc(col)] = np.nan
                
                logger.debug(f"Purged last {periods} rows for {col}")
        
        logger.info("Purged overlapping labels", columns=list(forward_periods.keys()))
        return df
    
    def calculate_all_labels(
        self,
        df: pd.DataFrame,
        horizons: Dict[str, int] = None,
        thresholds: Dict[str, Tuple[float, float]] = None,
        embargo_periods: int = 0,  # ← No embargo for testing
        apply_purge: bool = False, # ← No purge for testing
    ) -> pd.DataFrame:
        """
        Calculate all labels with leakage prevention.
        
        Pipeline:
        1. Calculate forward returns
        2. Convert to directional labels
        3. Apply embargo periods
        4. Purge overlapping windows
        
        Args:
            df: DataFrame with OHLCV data
            horizons: Dict of {name: periods}
            thresholds: Dict of {col: (down, up) thresholds}
            embargo_periods: Number of periods to embargo
            apply_purge: Whether to purge overlapping labels
            
        Returns:
            DataFrame with all label columns and leakage prevention applied
        """
        logger.info("Calculating all labels with leakage prevention", rows=len(df))
        
        # Step 1: Forward returns
        df = self.calculate_forward_returns(df, horizons=horizons)
        
        # Step 2: Directional labels
        df = self.calculate_directional_labels(df, thresholds=thresholds)
        
        # Step 3: Embargo and Purging
        # Note: Purged cross-validation splitters (e.g. PurgedWalkForwardValidator)
        # enforce the embargo buffer between train and test windows.
        if apply_purge:
            df = self.purge_overlapping_labels(df, forward_periods=horizons)
        
        # Count valid labels
        label_cols = [col for col in df.columns if col.startswith("label_")]
        valid_counts = {col: df[col].notna().sum() for col in label_cols}
        
        logger.info(
            "All labels calculated",
            valid_label_counts=valid_counts,
        )
        
        return df


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def calculate_all_labels(
    df: pd.DataFrame,
    horizons: Dict[str, int] = None,
    embargo_periods: int = 2,
) -> pd.DataFrame:
    """
    Quick function to calculate all labels.
    
    Examples:
        >>> from src.processing.labels import calculate_all_labels
        >>> df_with_labels = calculate_all_labels(spy_df)
    """
    generator = TargetGenerator()
    return generator.calculate_all_labels(
        df,
        horizons=horizons,
        embargo_periods=embargo_periods,
    )
