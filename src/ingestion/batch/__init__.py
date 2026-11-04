"""
Batch data ingestion modules.

Phase 2:
- yfinance: FREE SPY/QQQ/VIX data (baseline)
- FRED: FREE macro/economic data
"""

from src.ingestion.batch.macro_fred import FREDIngestion, ingest_fred_latest
from src.ingestion.batch.underlying_yfinance import (
    YFinanceIngestion,
    ingest_yfinance_latest,
)

__all__ = [
    "YFinanceIngestion",
    "ingest_yfinance_latest",
    "FREDIngestion",
    "ingest_fred_latest",
]
