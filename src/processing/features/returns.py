"""
Returns calculation - Core price & returns features.

Implements:
- es_ret_1m, es_ret_5m, es_ret_15m
- spx_ret_1m
- es_gap_open
- es_atr_14 (Average True Range)
- es_bb_width_20 (Bollinger Band Width)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from src.core import get_logger

logger = get_logger(__name__)


class ReturnsCalculator:
    """Calculate price returns and volatility metrics."""
    
    def __init__(self):
        """Initialize returns calculator."""
        logger.info("ReturnsCalculator initialized")
    
    def calculate_returns(
        self,
        df: pd.DataFrame,
        symbol_col: str = "symbol",
        close_col: str = "close",
        periods: list = [1, 5, 15],
    ) -> pd.DataFrame:
        """
        Calculate returns for multiple periods.
        
        Args:
            df: DataFrame with OHLCV data
            symbol_col: Column name for symbol
            close_col: Column name for close price
            periods: List of periods to calculate (in bars)
        
        Returns:
            DataFrame with return columns added
        
        Examples:
            >>> calc = ReturnsCalculator()
            >>> df = calc.calculate_returns(spy_data, periods=[1, 5, 15])
            >>> df[['spy_ret_1m', 'spy_ret_5m', 'spy_ret_15m']]
        """
        df = df.copy()
        
        # Get symbol name (lowercase for feature names)
        symbol = df[symbol_col].iloc[0].lower().replace("^", "").replace("/", "")
        
        for period in periods:
            feature_name = f"{symbol}_ret_{period}m"
            
            # Calculate percentage return
            df[feature_name] = df[close_col].pct_change(periods=period)
            
            logger.debug(f"Calculated {feature_name}", period=period, rows=len(df))
        
        return df
    
    def calculate_gap_open(
        self,
        df: pd.DataFrame,
        symbol_col: str = "symbol",
        open_col: str = "open",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate overnight gap (session open vs previous close).
        
        Formula: (Open_t - Close_t-1) / Close_t-1
        
        Args:
            df: DataFrame with OHLCV data (must have date info)
            symbol_col: Column name for symbol
            open_col: Column name for open price
            close_col: Column name for close price
        
        Returns:
            DataFrame with gap_open column added
        """
        df = df.copy()
        
        symbol = df[symbol_col].iloc[0].lower().replace("^", "").replace("/", "")
        feature_name = f"{symbol}_gap_open"
        
        # Calculate gap: (open - prev_close) / prev_close
        prev_close = df[close_col].shift(1)
        df[feature_name] = (df[open_col] - prev_close) / prev_close
        
        # First bar of each day will have NaN (no previous day)
        # This is correct - we only have gap at session open
        
        logger.info(f"Calculated {feature_name}", rows=len(df))
        return df
    
    def calculate_atr(
        self,
        df: pd.DataFrame,
        period: int = 14,
        symbol_col: str = "symbol",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate Average True Range (ATR).
        
        Formula: ATR = SMA(TrueRange, period)
        where TrueRange = max(high-low, |high-prev_close|, |low-prev_close|)
        
        Args:
            df: DataFrame with OHLCV data
            period: ATR period (default 14)
            
        Returns:
            DataFrame with atr column added
        """
        df = df.copy()
        
        symbol = df[symbol_col].iloc[0].lower().replace("^", "").replace("/", "")
        feature_name = f"{symbol}_atr_{period}"
        
        # Calculate True Range
        prev_close = df[close_col].shift(1)
        
        tr1 = df[high_col] - df[low_col]
        tr2 = (df[high_col] - prev_close).abs()
        tr3 = (df[low_col] - prev_close).abs()
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR is the moving average of True Range
        df[feature_name] = true_range.rolling(window=period).mean()
        
        logger.info(f"Calculated {feature_name}", period=period, rows=len(df))
        return df
    
    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        period: int = 20,
        num_std: float = 2.0,
        symbol_col: str = "symbol",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands and bandwidth.
        
        Formula:
        - Middle Band = SMA(close, period)
        - Upper Band = Middle + (num_std * StdDev)
        - Lower Band = Middle - (num_std * StdDev)
        - BB Width = (Upper - Lower) / Middle
        
        Args:
            df: DataFrame with OHLCV data
            period: Bollinger Band period (default 20)
            num_std: Number of standard deviations (default 2.0)
            
        Returns:
            DataFrame with bb_upper, bb_lower, bb_width columns
        """
        df = df.copy()
        
        symbol = df[symbol_col].iloc[0].lower().replace("^", "").replace("/", "")
        
        # Calculate middle band (SMA)
        sma = df[close_col].rolling(window=period).mean()
        std = df[close_col].rolling(window=period).std()
        
        # Upper and lower bands
        df[f"{symbol}_bb_upper_{period}"] = sma + (num_std * std)
        df[f"{symbol}_bb_lower_{period}"] = sma - (num_std * std)
        df[f"{symbol}_bb_middle_{period}"] = sma
        
        # Bandwidth (normalized by middle band)
        df[f"{symbol}_bb_width_{period}"] = (
            (df[f"{symbol}_bb_upper_{period}"] - df[f"{symbol}_bb_lower_{period}"]) / sma
        )
        
        logger.info(f"Calculated Bollinger Bands", symbol=symbol, period=period)
        return df
    
    def calculate_all_returns_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate all return-based features at once.
        
        This is a convenience method that calculates:
        - Returns (1m, 5m, 15m)
        - Gap open
        - ATR(14)
        - Bollinger Band Width(20)
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all features added
        """
        logger.info("Calculating all returns features", rows=len(df))
        
        # Returns
        df = self.calculate_returns(df, periods=[1, 5, 15])
        
        # Gap open
        df = self.calculate_gap_open(df)
        
        # ATR
        df = self.calculate_atr(df, period=14)
        
        # Bollinger Bands
        df = self.calculate_bollinger_bands(df, period=20)
        
        # Count non-NaN features
        feature_cols = [col for col in df.columns if any(x in col for x in ['_ret_', '_gap_', '_atr_', '_bb_'])]
        non_null_count = df[feature_cols].notna().sum().sum()
        
        logger.info(
            "All returns features calculated",
            total_features=len(feature_cols),
            non_null_values=int(non_null_count),
        )
        
        return df


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def calculate_returns(df: pd.DataFrame, periods: list = [1, 5, 15]) -> pd.DataFrame:
    """
    Quick function to calculate returns.
    
    Examples:
        >>> from src.processing.features import calculate_returns
        >>> df_with_returns = calculate_returns(spy_df)
    """
    calc = ReturnsCalculator()
    return calc.calculate_returns(df, periods=periods)


def calculate_all_returns_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick function to calculate all return features.
    
    Examples:
        >>> from src.processing.features import calculate_all_returns_features
        >>> df_features = calculate_all_returns_features(spy_df)
    """
    calc = ReturnsCalculator()
    return calc.calculate_all_returns_features(df)
