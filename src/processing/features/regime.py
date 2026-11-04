"""
Event & regime detection features.

Implements:
- session_minute (minutes since market open)
- is_open_first_hour (first hour flag)
- is_power_hour (last hour flag)
- is_fomc_day (FOMC decision day flag)
- minutes_to_event (minutes until major economic event)
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Optional, List

from src.core import get_logger, get_calendar

logger = get_logger(__name__)


class RegimeFeatures:
    """Calculate event and regime detection features."""
    
    def __init__(self):
        """Initialize regime features calculator."""
        self.calendar = get_calendar()
        logger.info("RegimeFeatures initialized")
    
    def calculate_session_minute(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        market_open: time = time(9, 30),
    ) -> pd.DataFrame:
        """
        Calculate minutes since market open.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Column name for timestamps
            market_open: Market open time (default 9:30 AM)
            
        Returns:
            DataFrame with session_minute column
        """
        df = df.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract time component
        df['_time'] = df[timestamp_col].dt.time
        
        # Calculate minutes since market open
        # Convert times to minutes since midnight
        market_open_minutes = market_open.hour * 60 + market_open.minute
        current_minutes = df[timestamp_col].dt.hour * 60 + df[timestamp_col].dt.minute
        
        df['session_minute'] = current_minutes - market_open_minutes
        
        # Negative values = pre-market (set to 0 or NaN depending on use case)
        df.loc[df['session_minute'] < 0, 'session_minute'] = np.nan
        
        df = df.drop('_time', axis=1)
        
        logger.info("Calculated session_minute", rows=len(df))
        return df
    
    def calculate_session_flags(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        market_open: time = time(9, 30),
        market_close: time = time(16, 0),
    ) -> pd.DataFrame:
        """
        Calculate session period flags (first hour, power hour).
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Column name for timestamps
            market_open: Market open time
            market_close: Market close time
            
        Returns:
            DataFrame with is_open_first_hour and is_power_hour columns
        """
        df = df.copy()
        
        # Ensure session_minute exists
        if 'session_minute' not in df.columns:
            df = self.calculate_session_minute(df, timestamp_col, market_open)
        
        # First hour: 0 to 60 minutes after open
        df['is_open_first_hour'] = ((df['session_minute'] >= 0) & (df['session_minute'] < 60)).astype(int)
        
        # Power hour: last 60 minutes before close
        # Market close at 16:00 = 390 minutes after 9:30 open
        market_close_minutes = (market_close.hour * 60 + market_close.minute) - (market_open.hour * 60 + market_open.minute)
        df['is_power_hour'] = ((df['session_minute'] >= market_close_minutes - 60) & (df['session_minute'] <= market_close_minutes)).astype(int)
        
        logger.info("Calculated session flags", first_hour_count=df['is_open_first_hour'].sum(), power_hour_count=df['is_power_hour'].sum())
        return df
    
    def calculate_fomc_flags(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        fomc_dates: Optional[List[datetime]] = None,
    ) -> pd.DataFrame:
        """
        Calculate FOMC day flags.
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Column name for timestamps
            fomc_dates: List of FOMC decision dates (if None, uses a default list)
            
        Returns:
            DataFrame with is_fomc_day column
        """
        df = df.copy()
        
        # Default FOMC dates for 2025 (update as needed)
        if fomc_dates is None:
            fomc_dates = [
                datetime(2025, 1, 29),   # Jan FOMC
                datetime(2025, 3, 19),   # Mar FOMC
                datetime(2025, 5, 7),    # May FOMC
                datetime(2025, 6, 18),   # Jun FOMC
                datetime(2025, 7, 30),   # Jul FOMC
                datetime(2025, 9, 17),   # Sep FOMC
                datetime(2025, 11, 5),   # Nov FOMC
                datetime(2025, 12, 17),  # Dec FOMC
            ]
        
        # Extract date only
        df['_date'] = pd.to_datetime(df[timestamp_col]).dt.date
        fomc_dates_only = [d.date() for d in fomc_dates]
        
        # Flag FOMC days
        df['is_fomc_day'] = df['_date'].isin(fomc_dates_only).astype(int)
        
        df = df.drop('_date', axis=1)
        
        logger.info("Calculated FOMC flags", fomc_days=df['is_fomc_day'].sum())
        return df
    
    def calculate_minutes_to_event(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        event_times: Optional[List[datetime]] = None,
    ) -> pd.DataFrame:
        """
        Calculate minutes until next major event (CPI, FOMC, NFP, etc.).
        
        Args:
            df: DataFrame with timestamp column
            timestamp_col: Column name for timestamps
            event_times: List of event datetimes (if None, uses defaults)
            
        Returns:
            DataFrame with minutes_to_event column
        """
        df = df.copy()
        
        # Default major events for 2025 (update as needed)
        if event_times is None:
            event_times = [
                # FOMC decisions (2:00 PM ET)
                datetime(2025, 1, 29, 14, 0),
                datetime(2025, 3, 19, 14, 0),
                datetime(2025, 5, 7, 14, 0),
                # CPI releases (8:30 AM ET, usually mid-month)
                datetime(2025, 2, 13, 8, 30),
                datetime(2025, 3, 12, 8, 30),
                datetime(2025, 4, 10, 8, 30),
                # NFP (Non-Farm Payrolls - first Friday, 8:30 AM ET)
                datetime(2025, 2, 7, 8, 30),
                datetime(2025, 3, 7, 8, 30),
                datetime(2025, 4, 4, 8, 30),
            ]
        
        event_times = pd.to_datetime(event_times)
        
        # Match timezone with input timestamps
        ts_series = pd.to_datetime(df[timestamp_col])
        if len(ts_series) > 0 and getattr(ts_series.iloc[0], "tzinfo", None) is not None:
            tz = ts_series.iloc[0].tz
            event_times = event_times.tz_localize(tz)
        
        # For each timestamp, find the next event
        minutes_to_next = []
        for ts in ts_series:
            future_events = event_times[event_times > ts]
            if len(future_events) > 0:
                next_event = future_events[0]
                minutes = (next_event - ts).total_seconds() / 60
                minutes_to_next.append(minutes)
            else:
                minutes_to_next.append(np.nan)
        
        df['minutes_to_event'] = minutes_to_next
        
        logger.info("Calculated minutes_to_event", non_null=df['minutes_to_event'].notna().sum())
        return df
    
    def calculate_all_regime_features(
        self,
        df: pd.DataFrame,
        fomc_dates: Optional[List[datetime]] = None,
        event_times: Optional[List[datetime]] = None,
    ) -> pd.DataFrame:
        """
        Calculate all regime/event features at once.
        
        Args:
            df: DataFrame with timestamp column
            fomc_dates: List of FOMC dates
            event_times: List of major event times
            
        Returns:
            DataFrame with all regime features
        """
        logger.info("Calculating all regime features", rows=len(df))
        
        # Session minute
        df = self.calculate_session_minute(df)
        
        # Session flags
        df = self.calculate_session_flags(df)
        
        # FOMC flags
        df = self.calculate_fomc_flags(df, fomc_dates=fomc_dates)
        
        # Minutes to event
        df = self.calculate_minutes_to_event(df, event_times=event_times)
        
        feature_cols = ['session_minute', 'is_open_first_hour', 'is_power_hour', 'is_fomc_day', 'minutes_to_event']
        
        logger.info(
            "All regime features calculated",
            total_features=len(feature_cols),
        )
        
        return df


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def calculate_all_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick function to calculate all regime features.
    
    Examples:
        >>> from src.processing.features import calculate_all_regime_features
        >>> df_regime = calculate_all_regime_features(spy_df)
    """
    calc = RegimeFeatures()
    return calc.calculate_all_regime_features(df)
