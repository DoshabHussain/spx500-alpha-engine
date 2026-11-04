"""
Technical indicators - Trend & momentum features.

Implements:
- es_ema_5, es_ema_15, es_ema_30 (Exponential Moving Averages)
- es_macd, es_macd_signal (MACD indicator)
- es_rsi_5, es_rsi_14 (Relative Strength Index)
- es_obv (On-Balance Volume)
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.core import get_logger

logger = get_logger(__name__)


class TechnicalIndicators:
    """Calculate technical analysis indicators."""
    
    def __init__(self):
        """Initialize technical indicators calculator."""
        logger.info("TechnicalIndicators initialized")
    
    def calculate_ema(
        self,
        df: pd.DataFrame,
        periods: list = [5, 15, 30],
        symbol_col: str = "symbol",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages (EMA).
        
        Formula: EMA_t = (Close_t * α) + (EMA_t-1 * (1 - α))
        where α = 2 / (period + 1)
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of EMA periods to calculate
            symbol_col: Column name for symbol
            close_col: Column name for close price
            
        Returns:
            DataFrame with EMA columns added
        """
        df = df.copy()
        
        symbol = df[symbol_col].iloc[0].lower().replace("^", "").replace("/", "")
        
        for period in periods:
            feature_name = f"{symbol}_ema_{period}"
            
            # pandas ewm uses span parameter (span = period)
            df[feature_name] = df[close_col].ewm(span=period, adjust=False).mean()
            
            logger.debug(f"Calculated {feature_name}", period=period)
        
        logger.info(f"Calculated EMAs for {symbol}", periods=periods)
        return df
    
    def calculate_macd(
        self,
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        symbol_col: str = "symbol",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Formula:
        - MACD = EMA(12) - EMA(26)
        - Signal = EMA(9) of MACD
        - Histogram = MACD - Signal
        
        Args:
            df: DataFrame with OHLCV data
            fast_period: Fast EMA period (default 12)
            slow_period: Slow EMA period (default 26)
            signal_period: Signal line EMA period (default 9)
            
        Returns:
            DataFrame with macd, macd_signal, macd_histogram columns
        """
        df = df.copy()
        
        symbol = df[symbol_col].iloc[0].lower().replace("^", "").replace("/", "")
        
        # Calculate EMAs
        ema_fast = df[close_col].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df[close_col].ewm(span=slow_period, adjust=False).mean()
        
        # MACD = Fast EMA - Slow EMA
        df[f"{symbol}_macd"] = ema_fast - ema_slow
        
        # Signal = EMA of MACD
        df[f"{symbol}_macd_signal"] = df[f"{symbol}_macd"].ewm(span=signal_period, adjust=False).mean()
        
        # Histogram = MACD - Signal
        df[f"{symbol}_macd_histogram"] = df[f"{symbol}_macd"] - df[f"{symbol}_macd_signal"]
        
        logger.info(
            f"Calculated MACD for {symbol}",
            fast=fast_period,
            slow=slow_period,
            signal=signal_period,
        )
        return df
    
    def calculate_rsi(
        self,
        df: pd.DataFrame,
        periods: list = [5, 14],
        symbol_col: str = "symbol",
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate Relative Strength Index (RSI).
        
        Formula:
        1. Gain = max(Close_t - Close_t-1, 0)
        2. Loss = max(Close_t-1 - Close_t, 0)
        3. Avg Gain = EMA(Gain, period)
        4. Avg Loss = EMA(Loss, period)
        5. RS = Avg Gain / Avg Loss
        6. RSI = 100 - (100 / (1 + RS))
        
        Args:
            df: DataFrame with OHLCV data
            periods: List of RSI periods (default [5, 14])
            
        Returns:
            DataFrame with RSI columns
        """
        df = df.copy()
        
        symbol = df[symbol_col].iloc[0].lower().replace("^", "").replace("/", "")
        
        # Calculate price changes
        delta = df[close_col].diff()
        
        for period in periods:
            feature_name = f"{symbol}_rsi_{period}"
            
            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Calculate average gain and loss using EMA
            avg_gain = gain.ewm(span=period, adjust=False).mean()
            avg_loss = loss.ewm(span=period, adjust=False).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            df[feature_name] = 100 - (100 / (1 + rs))
            
            # Handle division by zero (when avg_loss = 0, RSI = 100)
            df[feature_name] = df[feature_name].fillna(100)
            
            logger.debug(f"Calculated {feature_name}", period=period)
        
        logger.info(f"Calculated RSI for {symbol}", periods=periods)
        return df
    
    def calculate_obv(
        self,
        df: pd.DataFrame,
        symbol_col: str = "symbol",
        close_col: str = "close",
        volume_col: str = "volume",
        reset_daily: bool = False,
    ) -> pd.DataFrame:
        """
        Calculate On-Balance Volume (OBV).
        
        Formula:
        - If Close_t > Close_t-1: OBV_t = OBV_t-1 + Volume_t
        - If Close_t < Close_t-1: OBV_t = OBV_t-1 - Volume_t
        - If Close_t = Close_t-1: OBV_t = OBV_t-1
        
        Args:
            df: DataFrame with OHLCV data
            symbol_col: Column name for symbol
            close_col: Column name for close price
            volume_col: Column name for volume
            reset_daily: If True, reset OBV at start of each day
            
        Returns:
            DataFrame with obv column
        """
        df = df.copy()
        
        symbol = df[symbol_col].iloc[0].lower().replace("^", "").replace("/", "")
        feature_name = f"{symbol}_obv"
        
        # Calculate price direction
        price_change = df[close_col].diff()
        
        # Signed volume: positive if price up, negative if price down
        signed_volume = df[volume_col].copy()
        signed_volume[price_change < 0] = -signed_volume[price_change < 0]
        signed_volume[price_change == 0] = 0
        
        # Cumulative sum
        if reset_daily and 'date' in df.columns:
            # Reset OBV at start of each day
            df[feature_name] = signed_volume.groupby(df['date']).cumsum()
        else:
            # Continuous OBV across all data
            df[feature_name] = signed_volume.cumsum()
        
        logger.info(f"Calculated OBV for {symbol}", reset_daily=reset_daily)
        return df
    
    def calculate_all_technical_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate all technical indicator features at once.
        
        This is a convenience method that calculates:
        - EMAs (5, 15, 30)
        - MACD (12, 26, 9)
        - RSI (5, 14)
        - OBV
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical features added
        """
        logger.info("Calculating all technical features", rows=len(df))
        
        # EMAs
        df = self.calculate_ema(df, periods=[5, 15, 30])
        
        # MACD
        df = self.calculate_macd(df)
        
        # RSI
        df = self.calculate_rsi(df, periods=[5, 14])
        
        # OBV
        df = self.calculate_obv(df)
        
        # Count features
        feature_cols = [col for col in df.columns if any(x in col for x in ['_ema_', '_macd', '_rsi_', '_obv'])]
        
        logger.info(
            "All technical features calculated",
            total_features=len(feature_cols),
        )
        
        return df


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def calculate_all_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick function to calculate all technical features.
    
    Examples:
        >>> from src.processing.features import calculate_all_technical_features
        >>> df_tech = calculate_all_technical_features(spy_df)
    """
    calc = TechnicalIndicators()
    return calc.calculate_all_technical_features(df)
