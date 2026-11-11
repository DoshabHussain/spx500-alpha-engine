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
from pathlib import Path
from datetime import datetime, time
from typing import Optional, List

import pandas as pd
import numpy as np
import pytz

from src.core import get_logger, get_calendar

logger = get_logger(__name__)

_CALENDAR_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "reference" / "economic_calendar.csv"
_NY_TZ = pytz.timezone("America/New_York")


def _load_economic_calendar() -> pd.DataFrame:
    """Load scheduled FOMC/CPI/NFP events from CSV with NY timezone awareness."""
    if _CALENDAR_PATH.exists():
        cal_df = pd.read_csv(_CALENDAR_PATH)
        cal_df["timestamp_et"] = pd.to_datetime(cal_df["timestamp_et"]).dt.tz_localize(_NY_TZ)
        return cal_df.sort_values("timestamp_et").reset_index(drop=True)
    return pd.DataFrame(columns=["event_type", "timestamp_et", "description"])


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
        Calculate FOMC decision day flags from reference calendar.
        """
        df = df.copy()

        if fomc_dates is None:
            cal_df = _load_economic_calendar()
            if not cal_df.empty:
                fomc_dates_only = cal_df.loc[cal_df["event_type"] == "FOMC", "timestamp_et"].dt.date.tolist()
            else:
                fomc_dates_only = []
        else:
            fomc_dates_only = [d.date() for d in fomc_dates]

        ts = pd.to_datetime(df[timestamp_col])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(_NY_TZ)
        df['is_fomc_day'] = ts.dt.date.isin(fomc_dates_only).astype(int)

        logger.info("Calculated FOMC flags", fomc_days=int(df['is_fomc_day'].sum()))
        return df

    def calculate_minutes_to_event(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        event_times: Optional[List[datetime]] = None,
        max_minutes_cap: float = 10080.0,
    ) -> pd.DataFrame:
        """
        Calculate minutes until next scheduled macro event (CPI, FOMC, NFP).
        """
        df = df.copy()

        if event_times is None:
            cal_df = _load_economic_calendar()
            if not cal_df.empty:
                event_ns = cal_df["timestamp_et"].dt.tz_convert("UTC").astype("int64").values
            else:
                event_ns = np.array([], dtype="int64")
        else:
            ev = pd.to_datetime(event_times)
            if ev.tz is None:
                ev = ev.tz_localize(_NY_TZ)
            event_ns = ev.tz_convert("UTC").sort_values().astype("int64").values

        ts_series = pd.to_datetime(df[timestamp_col])
        if ts_series.dt.tz is None:
            ts_utc = ts_series.dt.tz_localize(_NY_TZ).dt.tz_convert("UTC")
        else:
            ts_utc = ts_series.dt.tz_convert("UTC")

        ts_ns = ts_utc.astype("int64").values
        if len(event_ns) == 0:
            df['minutes_to_event'] = max_minutes_cap
        else:
            idx = np.searchsorted(event_ns, ts_ns, side="right")
            minutes_list = []
            for pos, t_val in zip(idx, ts_ns):
                if pos < len(event_ns):
                    mins = (event_ns[pos] - t_val) / (1e9 * 60.0)
                    minutes_list.append(min(max_minutes_cap, float(mins)))
                else:
                    minutes_list.append(max_minutes_cap)
            df['minutes_to_event'] = minutes_list

        logger.info("Calculated minutes_to_event", non_null=int(df['minutes_to_event'].notna().sum()))
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
