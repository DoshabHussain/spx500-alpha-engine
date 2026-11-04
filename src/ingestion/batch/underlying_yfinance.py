"""
yfinance data ingestion - FREE baseline for SPX/SPY/ES data.

Phase 1 Start: This is the first data source we implement.

Provides:
- SPY (S&P 500 ETF) - most liquid, good for daily baseline
- QQQ (Nasdaq ETF) - tech sector proxy
- ^VIX (VIX Index) - volatility measure

Note: yfinance is free but has limitations:
- Daily data only (for free tier)
- May have missing/delayed data
- Not real-time
- No options chains

For production, upgrade to Polygon or ORATS in Phase 2.
"""

import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
import yfinance as yf

from src.core import get_config, get_logger, to_utc
from src.core.schema import OHLCVBar, BarInterval
from src.storage.raw_store import get_raw_store


logger = get_logger(__name__)


# ─── YFINANCE INGESTION ─────────────────────────────────────────────────────

class YFinanceIngestion:
    """
    Ingest data from yfinance (free tier).
    
    This is our Phase 1 baseline - free and easy to set up.
    """
    
    def __init__(self) -> None:
        """Initialize yfinance ingestion."""
        config = get_config()
        self.config = config.data_sources.yfinance
        self.store = get_raw_store()
        
        logger.info(
            "YFinanceIngestion initialized",
            symbols=self.config.symbols,
            enabled=self.config.enabled,
        )
    
    def ingest_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Ingest data for a single symbol.
        
        Args:
            symbol: Ticker symbol (SPY, QQQ, ^VIX, etc.)
            start_date: Start date
            end_date: End date
            interval: Time interval (1d, 1h, etc.)
        
        Returns:
            DataFrame with OHLCV data
        
        Examples:
            >>> ingestion = YFinanceIngestion()
            >>> df = ingestion.ingest_symbol(
            ...     "SPY",
            ...     datetime(2025, 2, 1),
            ...     datetime(2025, 2, 28),
            ... )
            >>> print(len(df))
            20  # ~20 trading days
        """
        if not self.config.enabled:
            logger.warning("yfinance ingestion is disabled in config")
            return pd.DataFrame()
        
        logger.info(
            "Ingesting symbol",
            symbol=symbol,
            start=start_date.date(),
            end=end_date.date(),
            interval=interval,
        )
        
        try:
            # Download data with retry logic
            df = self._download_with_retry(symbol, start_date, end_date, interval)
            
            if df.empty:
                logger.warning("No data returned", symbol=symbol)
                return df
            
            # Clean and validate
            df = self._clean_data(df, symbol, interval)
            
            # Write to raw store
            for date, group in df.groupby(df.index.date):
                date_dt = datetime.combine(date, datetime.min.time())
                self.store.write_underlying(
                    group.reset_index(),
                    symbol=symbol,
                    date=date_dt,
                    data_type="bars",
                )
            
            logger.info(
                "Ingested symbol successfully",
                symbol=symbol,
                rows=len(df),
                start=df.index.min(),
                end=df.index.max(),
            )
            
            return df
        
        except Exception as e:
            logger.error(
                "Failed to ingest symbol",
                symbol=symbol,
                error=str(e),
                exc_info=True,
            )
            return pd.DataFrame()
    
    def ingest_all(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Ingest all configured symbols.
        
        Args:
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: today)
        
        Returns:
            Dict of symbol -> DataFrame
        
        Examples:
            >>> ingestion = YFinanceIngestion()
            >>> data = ingestion.ingest_all()
            >>> print(list(data.keys()))
            ['SPY', 'QQQ', '^VIX']
        """
        # Default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365)
        
        results = {}
        
        for symbol in self.config.symbols:
            df = self.ingest_symbol(symbol, start_date, end_date, self.config.interval)
            results[symbol] = df
            
            # Respect rate limits
            time.sleep(self.config.rate_limit["delay_seconds"])
        
        logger.info(
            "Ingested all symbols",
            symbols=len(results),
            total_rows=sum(len(df) for df in results.values()),
        )
        
        return results
    
    def ingest_latest(self) -> dict[str, pd.DataFrame]:
        """
        Ingest latest data (last 5 days).
        
        Useful for daily updates.
        
        Returns:
            Dict of symbol -> DataFrame
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        return self.ingest_all(start_date, end_date)
    
    # ─── INTERNAL HELPERS ───────────────────────────────────────────────────
    
    def _download_with_retry(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
    ) -> pd.DataFrame:
        """
        Download data with retry logic.
        
        yfinance can be flaky, so we retry on failure.
        """
        max_attempts = self.config.retry["max_attempts"]
        backoff_factor = self.config.retry["backoff_factor"]
        
        for attempt in range(max_attempts):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval,
                )
                
                if not df.empty:
                    return df
                
                logger.warning(
                    "Empty data on attempt",
                    symbol=symbol,
                    attempt=attempt + 1,
                )
            
            except Exception as e:
                logger.warning(
                    "Download failed on attempt",
                    symbol=symbol,
                    attempt=attempt + 1,
                    error=str(e),
                )
            
            # Backoff before retry
            if attempt < max_attempts - 1:
                sleep_time = backoff_factor ** attempt
                logger.debug(f"Retrying in {sleep_time}s")
                time.sleep(sleep_time)
        
        raise RuntimeError(f"Failed to download {symbol} after {max_attempts} attempts")
    
    def _clean_data(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
    ) -> pd.DataFrame:
        """
        Clean and standardize yfinance data.
        
        yfinance returns non-standard column names, we normalize them.
        """
        # Rename columns to lowercase
        df.columns = df.columns.str.lower()
        
        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert index to UTC datetime
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        
        # Add timestamp column
        df["timestamp"] = df.index
        
        # Add symbol column
        df["symbol"] = symbol
        
        # Add interval
        df["interval"] = interval
        
        # Drop NaN values
        df = df.dropna(subset=required)
        
        # Validate data integrity
        self._validate_ohlcv(df)
        
        return df
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> None:
        """
        Validate OHLCV data integrity.
        
        Checks:
        - High >= Low
        - Close between High and Low
        - Positive prices
        - Non-negative volume
        """
        # Check high >= low
        invalid_hl = df[df["high"] < df["low"]]
        if not invalid_hl.empty:
            logger.warning(
                "Found bars with high < low",
                count=len(invalid_hl),
                sample=invalid_hl.index[:3].tolist(),
            )
            # Fix by swapping
            df.loc[df["high"] < df["low"], ["high", "low"]] = df.loc[
                df["high"] < df["low"], ["low", "high"]
            ].values
        
        # Check close in range
        invalid_close = df[(df["close"] < df["low"]) | (df["close"] > df["high"])]
        if not invalid_close.empty:
            logger.warning(
                "Found bars with close out of range",
                count=len(invalid_close),
                sample=invalid_close.index[:3].tolist(),
            )
            # Clip to range
            df.loc[df["close"] < df["low"], "close"] = df.loc[df["close"] < df["low"], "low"]
            df.loc[df["close"] > df["high"], "close"] = df.loc[df["close"] > df["high"], "high"]
        
        # Check positive prices
        negative_prices = df[(df[["open", "high", "low", "close"]] <= 0).any(axis=1)]
        if not negative_prices.empty:
            logger.error(
                "Found negative or zero prices",
                count=len(negative_prices),
                sample=negative_prices.index[:3].tolist(),
            )
            raise ValueError("Data contains invalid prices (<= 0)")
        
        # Check non-negative volume
        negative_volume = df[df["volume"] < 0]
        if not negative_volume.empty:
            logger.warning(
                "Found negative volume",
                count=len(negative_volume),
            )
            df.loc[df["volume"] < 0, "volume"] = 0


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def ingest_yfinance_latest() -> dict[str, pd.DataFrame]:
    """
    Quick function to ingest latest data.
    
    Returns:
        Dict of symbol -> DataFrame
    
    Examples:
        >>> from src.ingestion.batch import ingest_yfinance_latest
        >>> data = ingest_yfinance_latest()
        >>> print(data["SPY"].tail())
    """
    ingestion = YFinanceIngestion()
    return ingestion.ingest_latest()


# Import Dict for type hints
from typing import Dict
