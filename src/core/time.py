"""
Time and timezone utilities for market data.

Handles:
- UTC ↔ NY ↔ Chicago conversions
- DST-safe date arithmetic
- Trading day calculations
- Timestamp parsing and formatting

All timestamps stored internally as UTC, converted for display.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import pandas as pd
import pytz
from dateutil import parser as dateutil_parser


# ─── CONSTANTS ──────────────────────────────────────────────────────────────

UTC = timezone.utc
NY_TZ = pytz.timezone("America/New_York")
CHICAGO_TZ = pytz.timezone("America/Chicago")
LONDON_TZ = pytz.timezone("Europe/London")


# ─── TIMEZONE CONVERSIONS ───────────────────────────────────────────────────

def to_utc(dt: Union[datetime, pd.Timestamp, str], from_tz: str = "America/New_York") -> datetime:
    """
    Convert a datetime to UTC.
    
    Args:
        dt: Input datetime (naive or aware)
        from_tz: Source timezone if dt is naive (default: NY)
    
    Returns:
        UTC datetime (aware)
    
    Examples:
        >>> # Convert NY market open to UTC
        >>> ny_open = datetime(2025, 2, 6, 9, 30)
        >>> utc_open = to_utc(ny_open)
        >>> print(utc_open)
        2025-02-06 14:30:00+00:00
        
        >>> # Convert string timestamp
        >>> to_utc("2025-02-06 09:30:00", "America/New_York")
        datetime.datetime(2025, 2, 6, 14, 30, tzinfo=datetime.timezone.utc)
    """
    # Parse string
    if isinstance(dt, str):
        dt = dateutil_parser.parse(dt)
    
    # Convert pandas Timestamp to datetime
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    
    # Handle naive datetime (assume from_tz)
    if dt.tzinfo is None:
        tz = pytz.timezone(from_tz)
        dt = tz.localize(dt)
    
    # Convert to UTC
    return dt.astimezone(UTC)


def from_utc(dt: datetime, to_tz: str = "America/New_York") -> datetime:
    """
    Convert UTC datetime to target timezone.
    
    Args:
        dt: UTC datetime
        to_tz: Target timezone
    
    Returns:
        Datetime in target timezone
    
    Examples:
        >>> utc_time = datetime(2025, 2, 6, 14, 30, tzinfo=timezone.utc)
        >>> ny_time = from_utc(utc_time, "America/New_York")
        >>> print(ny_time.strftime("%H:%M"))
        09:30
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    
    target_tz = pytz.timezone(to_tz)
    return dt.astimezone(target_tz)


def now_utc() -> datetime:
    """Get current time in UTC."""
    return datetime.now(UTC)


def now_ny() -> datetime:
    """Get current time in New York."""
    return now_utc().astimezone(NY_TZ)


def now_chicago() -> datetime:
    """Get current time in Chicago."""
    return now_utc().astimezone(CHICAGO_TZ)


# ─── MARKET HOURS ───────────────────────────────────────────────────────────

def is_market_hours(
    dt: Optional[datetime] = None,
    extended: bool = False,
    tz: str = "America/New_York",
) -> bool:
    """
    Check if datetime falls within market hours.
    
    Args:
        dt: Datetime to check (default: now)
        extended: Include extended hours (pre-market + after-hours)
        tz: Timezone of dt if naive
    
    Returns:
        True if within market hours
    
    Examples:
        >>> # Check if market is open now
        >>> is_market_hours()
        True  # (if called during RTH)
        
        >>> # Check specific time
        >>> dt = datetime(2025, 2, 6, 9, 30)  # Market open
        >>> is_market_hours(dt)
        True
    """
    if dt is None:
        dt = now_ny()
    
    # Convert to NY time
    if dt.tzinfo is None:
        dt = pytz.timezone(tz).localize(dt)
    dt_ny = dt.astimezone(NY_TZ)
    
    # Regular trading hours: 9:30 AM - 4:00 PM ET
    rth_open = dt_ny.replace(hour=9, minute=30, second=0, microsecond=0)
    rth_close = dt_ny.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if extended:
        # Extended hours: 4:00 AM - 8:00 PM ET
        eth_open = dt_ny.replace(hour=4, minute=0, second=0, microsecond=0)
        eth_close = dt_ny.replace(hour=20, minute=0, second=0, microsecond=0)
        return eth_open <= dt_ny <= eth_close
    else:
        return rth_open <= dt_ny <= rth_close


def market_open_time(date: Union[datetime, str], tz: str = "America/New_York") -> datetime:
    """
    Get market open time for a given date.
    
    Args:
        date: Target date
        tz: Timezone (default: NY)
    
    Returns:
        Market open datetime (9:30 AM ET)
    
    Examples:
        >>> open_time = market_open_time("2025-02-06")
        >>> print(open_time.strftime("%Y-%m-%d %H:%M %Z"))
        2025-02-06 09:30 EST
    """
    if isinstance(date, str):
        date = dateutil_parser.parse(date)
    
    target_tz = pytz.timezone(tz)
    return target_tz.localize(date.replace(hour=9, minute=30, second=0, microsecond=0))


def market_close_time(date: Union[datetime, str], tz: str = "America/New_York") -> datetime:
    """
    Get market close time for a given date.
    
    Args:
        date: Target date
        tz: Timezone (default: NY)
    
    Returns:
        Market close datetime (4:00 PM ET)
    
    Examples:
        >>> close_time = market_close_time("2025-02-06")
        >>> print(close_time.strftime("%Y-%m-%d %H:%M %Z"))
        2025-02-06 16:00 EST
    """
    if isinstance(date, str):
        date = dateutil_parser.parse(date)
    
    target_tz = pytz.timezone(tz)
    return target_tz.localize(date.replace(hour=16, minute=0, second=0, microsecond=0))


# ─── DATE ARITHMETIC ────────────────────────────────────────────────────────

def add_trading_days(date: datetime, days: int) -> datetime:
    """
    Add trading days to a date (skips weekends).
    
    Note: Does NOT skip holidays. Use calendar.py for holiday-aware logic.
    
    Args:
        date: Starting date
        days: Number of trading days to add (can be negative)
    
    Returns:
        New date
    
    Examples:
        >>> # Friday + 1 trading day = Monday
        >>> friday = datetime(2025, 2, 7)  # Friday
        >>> monday = add_trading_days(friday, 1)
        >>> print(monday.strftime("%A"))
        Monday
    """
    current = date
    remaining = abs(days)
    direction = 1 if days > 0 else -1
    
    while remaining > 0:
        current += timedelta(days=direction)
        # Skip weekends (Saturday=5, Sunday=6)
        if current.weekday() < 5:
            remaining -= 1
    
    return current


def is_weekend(dt: datetime) -> bool:
    """Check if datetime is on a weekend."""
    return dt.weekday() >= 5  # Saturday=5, Sunday=6


# ─── FORMATTING ─────────────────────────────────────────────────────────────

def format_timestamp(
    dt: datetime,
    fmt: str = "iso",
    tz: Optional[str] = None,
) -> str:
    """
    Format datetime as string.
    
    Args:
        dt: Datetime to format
        fmt: Format type (iso, us, compact, etc.)
        tz: Convert to this timezone before formatting
    
    Returns:
        Formatted string
    
    Examples:
        >>> dt = datetime(2025, 2, 6, 14, 30, tzinfo=timezone.utc)
        >>> format_timestamp(dt, "iso")
        '2025-02-06T14:30:00+00:00'
        
        >>> format_timestamp(dt, "us", "America/New_York")
        '02/06/2025 09:30 AM'
    """
    if tz:
        dt = from_utc(dt, tz)
    
    formats = {
        "iso": lambda d: d.isoformat(),
        "us": lambda d: d.strftime("%m/%d/%Y %I:%M %p"),
        "compact": lambda d: d.strftime("%Y%m%d_%H%M%S"),
        "date_only": lambda d: d.strftime("%Y-%m-%d"),
        "time_only": lambda d: d.strftime("%H:%M:%S"),
    }
    
    formatter = formats.get(fmt, formats["iso"])
    return formatter(dt)


def parse_timestamp(s: str, tz: Optional[str] = None) -> datetime:
    """
    Parse string to datetime (always returns UTC).
    
    Args:
        s: Timestamp string
        tz: Assume this timezone if string is naive
    
    Returns:
        UTC datetime
    
    Examples:
        >>> parse_timestamp("2025-02-06 09:30")
        datetime.datetime(2025, 2, 6, 9, 30, tzinfo=datetime.timezone.utc)
        
        >>> parse_timestamp("2025-02-06 09:30", "America/New_York")
        datetime.datetime(2025, 2, 6, 14, 30, tzinfo=datetime.timezone.utc)
    """
    dt = dateutil_parser.parse(s)
    return to_utc(dt, tz) if tz else dt.replace(tzinfo=UTC)


# ─── PANDAS INTEGRATION ─────────────────────────────────────────────────────

def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DataFrame index is UTC DatetimeIndex.
    
    Args:
        df: Input DataFrame with datetime index
    
    Returns:
        DataFrame with UTC index
    
    Examples:
        >>> df = pd.DataFrame({"close": [100, 101]}, 
        ...                   index=pd.date_range("2025-02-06", periods=2))
        >>> df_utc = ensure_utc_index(df)
        >>> print(df_utc.index.tz)
        UTC
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    elif df.index.tz != pytz.UTC:
        df.index = df.index.tz_convert("UTC")
    
    return df


def series_to_ny(series: pd.Series) -> pd.Series:
    """Convert datetime series to NY timezone."""
    if series.dt.tz is None:
        series = series.dt.tz_localize("UTC")
    return series.dt.tz_convert(NY_TZ)


# ─── UTILITIES ──────────────────────────────────────────────────────────────

def get_current_date() -> str:
    """Get current date as YYYY-MM-DD string (NY timezone)."""
    return now_ny().strftime("%Y-%m-%d")


def get_current_timestamp() -> str:
    """Get current timestamp as ISO string (UTC)."""
    return now_utc().isoformat()


def days_between(start: datetime, end: datetime) -> int:
    """Calculate calendar days between two dates."""
    return (end.date() - start.date()).days


def seconds_between(start: datetime, end: datetime) -> float:
    """Calculate seconds between two datetimes."""
    return (end - start).total_seconds()
