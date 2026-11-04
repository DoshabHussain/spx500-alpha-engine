"""
FRED (Federal Reserve Economic Data) ingestion.

FREE economic data from the St. Louis Fed:
- Treasury yields (DGS10, DGS2, T10Y2Y)
- VIX data
- Economic indicators (when needed in later phases)

Requires FREE API key from: https://fred.stlouisfed.org/docs/api/api_key.html
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from fredapi import Fred

from src.core import get_config, get_logger, get_api_key
from src.storage.raw_store import get_raw_store


logger = get_logger(__name__)


# ─── FRED INGESTION ─────────────────────────────────────────────────────────

class FREDIngestion:
    """
    Ingest macro/economic data from FRED API.
    
    FREE but requires API key.
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize FRED ingestion.
        
        Args:
            api_key: FRED API key (default: from environment)
        """
        self.api_key = api_key or get_api_key("fred")
        
        if not self.api_key:
            raise ValueError(
                "FRED API key not found. Get one at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        
        self.fred = Fred(api_key=self.api_key)
        
        config = get_config()
        self.config = config.data_sources.fred
        self.store = get_raw_store()
        
        logger.info(
            "FREDIngestion initialized",
            series_count=len(self.config.series),
            enabled=self.config.enabled,
        )
    
    def ingest_series(
        self,
        series_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Ingest a single FRED series.
        
        Args:
            series_id: FRED series ID (DGS10, VIXCLS, etc.)
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with series data
        
        Examples:
            >>> ingestion = FREDIngestion()
            >>> df = ingestion.ingest_series(
            ...     "DGS10",
            ...     datetime(2025, 1, 1),
            ...     datetime(2025, 2, 6),
            ... )
            >>> print(df.head())
        """
        if not self.config.enabled:
            logger.warning("FRED ingestion is disabled in config")
            return pd.DataFrame()
        
        logger.info(
            "Ingesting FRED series",
            series_id=series_id,
            start=start_date.date(),
            end=end_date.date(),
        )
        
        try:
            # Download series
            series = self.fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date,
            )
            
            if series.empty:
                logger.warning("No data returned", series_id=series_id)
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = series.to_frame(name="value")
            df.index.name = "date"
            df = df.reset_index()
            
            # Add timestamp (align to market open)
            df["timestamp"] = pd.to_datetime(df["date"]).dt.tz_localize("America/New_York")
            df["timestamp"] = df["timestamp"].apply(
                lambda x: x.replace(hour=9, minute=30)
            )
            df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")
            
            # Add metadata
            df["series_id"] = series_id
            
            # Drop NaN values (FRED often has gaps)
            df = df.dropna(subset=["value"])
            
            # Write to raw store (partition by date)
            for date, group in df.groupby(df["date"]):
                date_dt = pd.to_datetime(date).to_pydatetime()
                self.store.write_macro(
                    group,
                    series_name=series_id,
                    date=date_dt,
                )
            
            logger.info(
                "Ingested FRED series successfully",
                series_id=series_id,
                rows=len(df),
                start=df["date"].min(),
                end=df["date"].max(),
            )
            
            return df
        
        except Exception as e:
            logger.error(
                "Failed to ingest FRED series",
                series_id=series_id,
                error=str(e),
                exc_info=True,
            )
            return pd.DataFrame()
    
    def ingest_all(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Ingest all configured FRED series.
        
        Args:
            start_date: Start date (default: 1 year ago)
            end_date: End date (default: today)
        
        Returns:
            Dict of series_id -> DataFrame
        """
        # Default date range
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            lookback = self.config.lookback_days
            start_date = end_date - timedelta(days=lookback)
        
        results = {}
        
        for series_id in self.config.series:
            df = self.ingest_series(series_id, start_date, end_date)
            results[series_id] = df
        
        logger.info(
            "Ingested all FRED series",
            series_count=len(results),
            total_rows=sum(len(df) for df in results.values()),
        )
        
        return results
    
    def ingest_latest(self) -> Dict[str, pd.DataFrame]:
        """
        Ingest latest data (last 30 days).
        
        Returns:
            Dict of series_id -> DataFrame
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        return self.ingest_all(start_date, end_date)


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def ingest_fred_latest() -> Dict[str, pd.DataFrame]:
    """
    Quick function to ingest latest FRED data.
    
    Returns:
        Dict of series_id -> DataFrame
    
    Examples:
        >>> from src.ingestion.batch import ingest_fred_latest
        >>> data = ingest_fred_latest()
        >>> print(data["DGS10"].tail())
    """
    ingestion = FREDIngestion()
    return ingestion.ingest_latest()
