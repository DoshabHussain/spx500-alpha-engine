"""
Volatility features - VIX, VVIX, VIXW, term structure.

Implements:
- vix_level (current VIX)
- vix_5m_chg (5-minute VIX change)
- vix_term_slope (VIX - VIXW)
- vixw_level, vvix_level (when data available)
"""

import pandas as pd
import numpy as np
from typing import Optional

from src.core import get_logger

logger = get_logger(__name__)


class VolatilityFeatures:
    """Calculate volatility-based features from VIX family indices."""
    
    def __init__(self):
        """Initialize volatility features calculator."""
        logger.info("VolatilityFeatures initialized")
    
    def calculate_vix_level(
        self,
        df: pd.DataFrame,
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Extract VIX level (just rename for consistency).
        
        Args:
            df: DataFrame with VIX data
            close_col: Column name for close/last price
            
        Returns:
            DataFrame with vix_level column
        """
        df = df.copy()
        df['vix_level'] = df[close_col]
        
        logger.info("Extracted vix_level", rows=len(df))
        return df
    
    def calculate_vix_change(
        self,
        df: pd.DataFrame,
        periods: list = [5],
        close_col: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate VIX percentage change over various periods.
        
        Formula: (VIX_t - VIX_t-5m) / VIX_t-5m
        
        Args:
            df: DataFrame with VIX data
            periods: List of periods to calculate (in bars, e.g., [5] for 5-minute)
            close_col: Column name for close/last price
            
        Returns:
            DataFrame with vix_Nm_chg columns
        """
        df = df.copy()
        
        for period in periods:
            feature_name = f"vix_{period}m_chg"
            
            # Calculate percentage change
            df[feature_name] = df[close_col].pct_change(periods=period)
            
            logger.debug(f"Calculated {feature_name}", period=period)
        
        logger.info("Calculated VIX changes", periods=periods)
        return df
    
    def calculate_vix_term_slope(
        self,
        vix_df: pd.DataFrame,
        vixw_df: Optional[pd.DataFrame] = None,
        vix_col: str = "close",
        vixw_col: str = "close",
    ) -> pd.DataFrame:
        """
        Calculate VIX term structure slope (VIX - VIXW).
        
        This measures the premium of 30-day vol (VIX) over 9-day vol (VIXW).
        Positive slope = contango (normal), Negative = backwardation (stress).
        
        Args:
            vix_df: DataFrame with VIX data
            vixw_df: DataFrame with VIXW data (optional)
            vix_col: Column for VIX values
            vixw_col: Column for VIXW values
            
        Returns:
            DataFrame with vix_term_slope column
        """
        df = vix_df.copy()
        
        if vixw_df is not None:
            # Align VIXW data to VIX timestamps
            # Assume both have the same index/timestamps
            if 'timestamp' in df.columns and 'timestamp' in vixw_df.columns:
                df = df.set_index('timestamp')
                vixw_aligned = vixw_df.set_index('timestamp')[vixw_col]
                df['vix_term_slope'] = df[vix_col] - vixw_aligned
                df = df.reset_index()
            else:
                # Simple direct subtraction if already aligned
                df['vix_term_slope'] = df[vix_col] - vixw_df[vixw_col].values
            
            logger.info("Calculated vix_term_slope from VIX and VIXW")
        else:
            # VIXW not available - set to NaN
            df['vix_term_slope'] = np.nan
            logger.warning("VIXW data not provided - vix_term_slope set to NaN")
        
        return df
    
    def calculate_all_volatility_features(
        self,
        vix_df: pd.DataFrame,
        vixw_df: Optional[pd.DataFrame] = None,
        vvix_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Calculate all volatility features at once.
        
        Args:
            vix_df: DataFrame with VIX data
            vixw_df: DataFrame with VIXW data (optional)
            vvix_df: DataFrame with VVIX data (optional)
            
        Returns:
            DataFrame with all volatility features
        """
        logger.info("Calculating all volatility features", rows=len(vix_df))
        
        # VIX level
        df = self.calculate_vix_level(vix_df)
        
        # VIX changes
        df = self.calculate_vix_change(df, periods=[5])
        
        # Term slope (if VIXW available)
        df = self.calculate_vix_term_slope(df, vixw_df)
        
        # VVIX level (if available)
        if vvix_df is not None:
            if 'timestamp' in df.columns and 'timestamp' in vvix_df.columns:
                df = df.set_index('timestamp')
                vvix_aligned = vvix_df.set_index('timestamp')['close']
                df['vvix_level'] = vvix_aligned
                df = df.reset_index()
            else:
                df['vvix_level'] = vvix_df['close'].values
            logger.info("Added vvix_level")
        else:
            df['vvix_level'] = np.nan
            logger.warning("VVIX data not provided - vvix_level set to NaN")
        
        # VIXW level (if available)
        if vixw_df is not None:
            if 'timestamp' in df.columns and 'timestamp' in vixw_df.columns:
                df = df.set_index('timestamp')
                vixw_aligned = vixw_df.set_index('timestamp')['close']
                df['vixw_level'] = vixw_aligned
                df = df.reset_index()
            else:
                df['vixw_level'] = vixw_df['close'].values
            logger.info("Added vixw_level")
        else:
            df['vixw_level'] = np.nan
        
        # Count non-NaN features
        feature_cols = ['vix_level', 'vix_5m_chg', 'vix_term_slope', 'vvix_level', 'vixw_level']
        feature_cols = [col for col in feature_cols if col in df.columns]
        non_null = df[feature_cols].notna().sum()
        
        logger.info(
            "All volatility features calculated",
            total_features=len(feature_cols),
            non_null_counts=non_null.to_dict(),
        )
        
        return df


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def calculate_all_volatility_features(
    vix_df: pd.DataFrame,
    vixw_df: Optional[pd.DataFrame] = None,
    vvix_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Quick function to calculate all volatility features.
    
    Examples:
        >>> from src.processing.features import calculate_all_volatility_features
        >>> df_vol = calculate_all_volatility_features(vix_df)
    """
    calc = VolatilityFeatures()
    return calc.calculate_all_volatility_features(vix_df, vixw_df, vvix_df)
