"""
Trading calendar and session management.

Handles:
- RTH (Regular Trading Hours) vs ETH (Extended Trading Hours)
- Holiday detection
- Early close detection
- Trading day validation

Uses pandas_market_calendars for official exchange calendars.
"""

from datetime import datetime, time, timedelta
from typing import List, Optional, Tuple

import pandas as pd
try:
    import pandas_market_calendars as mcal
except ImportError:
    mcal = None
import pytz

try:
    import holidays
except ImportError:
    holidays = None

from src.core.time import to_utc, from_utc, now_ny


# ─── CONSTANTS ──────────────────────────────────────────────────────────────

NY_TZ = pytz.timezone("America/New_York")

# Session times (Eastern Time)
RTH_OPEN = time(9, 30)   # Regular hours open: 9:30 AM
RTH_CLOSE = time(16, 0)  # Regular hours close: 4:00 PM
PRE_OPEN = time(4, 0)    # Pre-market open: 4:00 AM
POST_CLOSE = time(20, 0) # After-hours close: 8:00 PM

# Early close time (e.g., day before Thanksgiving)
EARLY_CLOSE = time(13, 0)  # 1:00 PM


# ─── CALENDAR MANAGER ───────────────────────────────────────────────────────

class FallbackCalendar:
    """Fallback exchange calendar using holidays package when pandas_market_calendars is absent."""
    def __init__(self, exchange: str = "NYSE") -> None:
        self.exchange = exchange
        self.holidays = set()
        if holidays is not None:
            try:
                self.holidays = holidays.financial_holidays(exchange)
            except Exception:
                try:
                    self.holidays = holidays.US()
                except Exception:
                    self.holidays = set()

    def schedule(self, start_date, end_date) -> pd.DataFrame:
        if hasattr(start_date, 'date'):
            start_d = start_date.date()
        else:
            start_d = pd.to_datetime(start_date).date()
        if hasattr(end_date, 'date'):
            end_d = end_date.date()
        else:
            end_d = pd.to_datetime(end_date).date()

        dates = pd.bdate_range(start_d, end_d)
        trading_dates = [d for d in dates if d.date() not in self.holidays]
        if not trading_dates:
            return pd.DataFrame(columns=["market_open", "market_close"])

        opens = [NY_TZ.localize(datetime.combine(d.date(), RTH_OPEN)) for d in trading_dates]
        closes = [NY_TZ.localize(datetime.combine(d.date(), RTH_CLOSE)) for d in trading_dates]
        return pd.DataFrame({"market_open": opens, "market_close": closes}, index=pd.DatetimeIndex([d.date() for d in trading_dates]))


class TradingCalendar:
    """
    Trading calendar manager using pandas_market_calendars (with holidays fallback).
    
    Supports: NYSE, NASDAQ, CBOE, CME
    """
    
    def __init__(self, exchange: str = "NYSE") -> None:
        """
        Initialize calendar.
        
        Args:
            exchange: Exchange name (NYSE, NASDAQ, CBOE, CME)
        
        Examples:
            >>> cal = TradingCalendar("NYSE")
            >>> cal.is_trading_day(datetime(2025, 2, 6))
            True
        """
        self.exchange = exchange
        if mcal is not None:
            try:
                self.calendar = mcal.get_calendar(exchange)
            except Exception:
                self.calendar = FallbackCalendar(exchange)
        else:
            self.calendar = FallbackCalendar(exchange)
        
        # Cache for performance
        self._trading_days_cache: Optional[pd.DatetimeIndex] = None
        self._cache_start: Optional[datetime] = None
        self._cache_end: Optional[datetime] = None
    
    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if date is a valid trading day.
        
        Args:
            date: Date to check
        
        Returns:
            True if valid trading day (not weekend, not holiday)
        
        Examples:
            >>> cal = TradingCalendar()
            >>> cal.is_trading_day(datetime(2025, 1, 1))  # New Year's Day
            False
            >>> cal.is_trading_day(datetime(2025, 2, 6))  # Normal Thursday
            True
        """
        # Quick weekend check
        if date.weekday() >= 5:
            return False
        
        # Check against official calendar
        schedule = self.calendar.schedule(
            start_date=date.date(),
            end_date=date.date()
        )
        return len(schedule) > 0
    
    def is_holiday(self, date: datetime) -> bool:
        """Check if date is a market holiday."""
        return date.weekday() < 5 and not self.is_trading_day(date)
    
    def is_early_close(self, date: datetime) -> bool:
        """
        Check if market closes early on this date.
        
        Args:
            date: Date to check
        
        Returns:
            True if early close day
        
        Examples:
            >>> cal = TradingCalendar()
            >>> # Day before Thanksgiving (usually early close)
            >>> cal.is_early_close(datetime(2025, 11, 28))
            True
        """
        if not self.is_trading_day(date):
            return False
        
        schedule = self.calendar.schedule(
            start_date=date.date(),
            end_date=date.date()
        )
        
        if schedule.empty:
            return False
        
        # Check if market close is before normal 4:00 PM
        market_close = schedule.iloc[0]["market_close"]
        normal_close = NY_TZ.localize(datetime.combine(date.date(), RTH_CLOSE))
        
        return market_close < normal_close
    
    def get_session_times(
        self,
        date: datetime,
        extended: bool = False,
    ) -> Optional[Tuple[datetime, datetime]]:
        """
        Get session open and close times for a date.
        
        Args:
            date: Target date
            extended: Return extended hours (pre + post market)
        
        Returns:
            Tuple of (open_time, close_time) or None if not a trading day
        
        Examples:
            >>> cal = TradingCalendar()
            >>> open_time, close_time = cal.get_session_times(datetime(2025, 2, 6))
            >>> print(open_time.strftime("%H:%M"))
            09:30
            >>> print(close_time.strftime("%H:%M"))
            16:00
        """
        if not self.is_trading_day(date):
            return None
        
        if extended:
            # Extended hours: 4:00 AM - 8:00 PM
            open_time = NY_TZ.localize(datetime.combine(date.date(), PRE_OPEN))
            close_time = NY_TZ.localize(datetime.combine(date.date(), POST_CLOSE))
        else:
            # Regular hours from official calendar
            schedule = self.calendar.schedule(
                start_date=date.date(),
                end_date=date.date()
            )
            
            if schedule.empty:
                return None
            
            open_time = schedule.iloc[0]["market_open"]
            close_time = schedule.iloc[0]["market_close"]
        
        return (open_time, close_time)
    
    def get_trading_days(
        self,
        start: datetime,
        end: datetime,
    ) -> pd.DatetimeIndex:
        """
        Get all trading days in a date range.
        
        Args:
            start: Start date
            end: End date
        
        Returns:
            DatetimeIndex of trading days
        
        Examples:
            >>> cal = TradingCalendar()
            >>> days = cal.get_trading_days(datetime(2025, 2, 1), datetime(2025, 2, 28))
            >>> len(days)  # ~20 trading days in February
            20
        """
        schedule = self.calendar.schedule(start_date=start.date(), end_date=end.date())
        return schedule.index
    
    def next_trading_day(self, date: datetime) -> datetime:
        """
        Get next trading day after given date.
        
        Args:
            date: Starting date
        
        Returns:
            Next valid trading day
        
        Examples:
            >>> cal = TradingCalendar()
            >>> friday = datetime(2025, 2, 7)  # Friday
            >>> next_day = cal.next_trading_day(friday)
            >>> print(next_day.strftime("%A"))  # Monday
            Monday
        """
        current = date + timedelta(days=1)
        while not self.is_trading_day(current):
            current += timedelta(days=1)
        return current
    
    def previous_trading_day(self, date: datetime) -> datetime:
        """Get previous trading day before given date."""
        current = date - timedelta(days=1)
        while not self.is_trading_day(current):
            current -= timedelta(days=1)
        return current
    
    def trading_days_between(self, start: datetime, end: datetime) -> int:
        """Count trading days between two dates (inclusive)."""
        days = self.get_trading_days(start, end)
        return len(days)


# ─── SESSION HELPERS ────────────────────────────────────────────────────────

def is_rth(dt: datetime, calendar: Optional[TradingCalendar] = None) -> bool:
    """
    Check if datetime falls within Regular Trading Hours.
    
    Args:
        dt: Datetime to check
        calendar: Optional trading calendar (default: NYSE)
    
    Returns:
        True if within RTH (9:30 AM - 4:00 PM ET)
    
    Examples:
        >>> is_rth(datetime(2025, 2, 6, 10, 0, tzinfo=NY_TZ))
        True
        >>> is_rth(datetime(2025, 2, 6, 8, 0, tzinfo=NY_TZ))
        False
    """
    if calendar is None:
        calendar = TradingCalendar()
    
    # Convert to NY time
    if dt.tzinfo is None:
        dt = NY_TZ.localize(dt)
    dt_ny = dt.astimezone(NY_TZ)
    
    # Check if trading day
    if not calendar.is_trading_day(dt_ny):
        return False
    
    # Check time range
    session = calendar.get_session_times(dt_ny, extended=False)
    if session is None:
        return False
    
    open_time, close_time = session
    return open_time <= dt_ny <= close_time


def is_eth(dt: datetime, calendar: Optional[TradingCalendar] = None) -> bool:
    """
    Check if datetime falls within Extended Trading Hours.
    
    Args:
        dt: Datetime to check
        calendar: Optional trading calendar
    
    Returns:
        True if within ETH (4:00 AM - 8:00 PM ET)
    """
    if calendar is None:
        calendar = TradingCalendar()
    
    # Convert to NY time
    if dt.tzinfo is None:
        dt = NY_TZ.localize(dt)
    dt_ny = dt.astimezone(NY_TZ)
    
    # Check if trading day
    if not calendar.is_trading_day(dt_ny):
        return False
    
    # Check extended time range
    session = calendar.get_session_times(dt_ny, extended=True)
    if session is None:
        return False
    
    open_time, close_time = session
    return open_time <= dt_ny <= close_time


def get_current_session() -> str:
    """
    Get current market session.
    
    Returns:
        "pre-market", "rth", "post-market", or "closed"
    
    Examples:
        >>> get_current_session()  # Called at 10:00 AM ET
        'rth'
    """
    now = now_ny()
    calendar = TradingCalendar()
    
    if not calendar.is_trading_day(now):
        return "closed"
    
    current_time = now.time()
    
    if current_time < RTH_OPEN:
        return "pre-market"
    elif current_time < RTH_CLOSE:
        return "rth"
    elif current_time < POST_CLOSE:
        return "post-market"
    else:
        return "closed"


def get_next_market_open(calendar: Optional[TradingCalendar] = None) -> datetime:
    """
    Get next market open time.
    
    Args:
        calendar: Optional trading calendar
    
    Returns:
        Next market open datetime (9:30 AM ET)
    
    Examples:
        >>> next_open = get_next_market_open()
        >>> print(next_open.strftime("%Y-%m-%d %H:%M"))
        2025-02-07 09:30  # Next trading day
    """
    if calendar is None:
        calendar = TradingCalendar()
    
    now = now_ny()
    
    # If currently within RTH, return today's open
    if is_rth(now, calendar):
        session = calendar.get_session_times(now)
        if session:
            return session[0]
    
    # Otherwise, get next trading day's open
    next_day = calendar.next_trading_day(now)
    session = calendar.get_session_times(next_day)
    return session[0] if session else next_day


def get_next_market_close(calendar: Optional[TradingCalendar] = None) -> datetime:
    """Get next market close time (4:00 PM ET)."""
    if calendar is None:
        calendar = TradingCalendar()
    
    now = now_ny()
    
    # If currently within RTH, return today's close
    if is_rth(now, calendar):
        session = calendar.get_session_times(now)
        if session:
            return session[1]
    
    # Otherwise, get next trading day's close
    next_day = calendar.next_trading_day(now)
    session = calendar.get_session_times(next_day)
    return session[1] if session else next_day


# ─── SINGLETON CALENDAR ─────────────────────────────────────────────────────

_default_calendar: Optional[TradingCalendar] = None


def get_calendar(exchange: str = "NYSE") -> TradingCalendar:
    """
    Get trading calendar (singleton pattern).
    
    Args:
        exchange: Exchange name
    
    Returns:
        TradingCalendar instance
    """
    global _default_calendar
    
    if _default_calendar is None or _default_calendar.exchange != exchange:
        _default_calendar = TradingCalendar(exchange)
    
    return _default_calendar


# ─── UTILITIES ──────────────────────────────────────────────────────────────

def align_to_market_open(date: datetime) -> datetime:
    """
    Align datetime to market open (9:30 AM ET).
    
    Args:
        date: Input datetime
    
    Returns:
        Datetime aligned to market open
    """
    calendar = get_calendar()
    session = calendar.get_session_times(date)
    return session[0] if session else date


def align_to_market_close(date: datetime) -> datetime:
    """Align datetime to market close (4:00 PM ET)."""
    calendar = get_calendar()
    session = calendar.get_session_times(date)
    return session[1] if session else date


def get_trading_minutes_in_day(date: datetime) -> int:
    """
    Get number of trading minutes in a day.
    
    Args:
        date: Target date
    
    Returns:
        Number of trading minutes (usually 390 for RTH)
    """
    calendar = get_calendar()
    session = calendar.get_session_times(date)
    
    if session is None:
        return 0
    
    open_time, close_time = session
    delta = close_time - open_time
    return int(delta.total_seconds() / 60)
