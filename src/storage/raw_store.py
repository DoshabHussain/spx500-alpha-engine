"""
Raw data store - append-only immutable storage.

Handles:
- Append-only writes (no updates/deletes)
- Partitioning by date and instrument
- Parquet file format with compression
- Manifest tracking for reproducibility
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import hashlib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import xxhash
except ImportError:
    xxhash = None

from src.core import get_config, get_logger


logger = get_logger(__name__)


# ─── RAW STORE ──────────────────────────────────────────────────────────────

class RawStore:
    """
    Append-only storage for raw market data.
    
    Data is partitioned by:
    - dt (date): YYYY-MM-DD
    - instrument: Symbol (SPY, SPX, SPXW, etc.)
    
    Files are immutable once written (no updates).
    """
    
    def __init__(self, root_dir: Optional[Path] = None) -> None:
        """
        Initialize raw store.
        
        Args:
            root_dir: Root directory for raw data (default: from config)
        
        Examples:
            >>> store = RawStore()
            >>> store.root_dir
            PosixPath('data/raw')
        """
        config = get_config()
        self.root_dir = root_dir or Path(config.storage.root_dir) / "raw"
        self.compression = config.storage.compression
        
        # Ensure root exists
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("RawStore initialized", root_dir=str(self.root_dir))
    
    def write_underlying(
        self,
        df: pd.DataFrame,
        symbol: str,
        date: datetime,
        data_type: str = "bars",
    ) -> Path:
        """
        Write underlying price data (SPY, /ES, SPX).
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Instrument symbol
            date: Trading date
            data_type: Data type (bars, ticks, etc.)
        
        Returns:
            Path to written file
        
        Examples:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'timestamp': [datetime(2025, 2, 6, 9, 30)],
            ...     'open': [100.0],
            ...     'high': [101.0],
            ...     'low': [99.5],
            ...     'close': [100.5],
            ...     'volume': [1000000]
            ... })
            >>> store = RawStore()
            >>> path = store.write_underlying(df, "SPY", datetime(2025, 2, 6))
        """
        partition_path = self._get_partition_path(
            "underlying",
            date,
            symbol,
            data_type,
        )
        
        return self._write_parquet(df, partition_path)
    
    def write_options(
        self,
        df: pd.DataFrame,
        symbol: str,
        date: datetime,
    ) -> Path:
        """
        Write options chain data (SPXW, SPY options).
        
        Args:
            df: DataFrame with options quotes
            symbol: Underlying symbol
            date: Trading date
        
        Returns:
            Path to written file
        """
        partition_path = self._get_partition_path(
            "options",
            date,
            symbol,
        )
        
        return self._write_parquet(df, partition_path)
    
    def write_indices(
        self,
        df: pd.DataFrame,
        index_name: str,
        date: datetime,
    ) -> Path:
        """
        Write index data (VIX, VVIX, VIXW, $TICK, $ADD).
        
        Args:
            df: DataFrame with index values
            index_name: Index name (VIX, VVIX, etc.)
            date: Trading date
        
        Returns:
            Path to written file
        """
        partition_path = self._get_partition_path(
            "indices",
            date,
            index_name,
        )
        
        return self._write_parquet(df, partition_path)
    
    def write_macro(
        self,
        df: pd.DataFrame,
        series_name: str,
        date: datetime,
    ) -> Path:
        """
        Write macro/economic data (FRED series).
        
        Args:
            df: DataFrame with macro series
            series_name: Series name (DGS10, T10Y2Y, etc.)
            date: Trading date
        
        Returns:
            Path to written file
        """
        partition_path = self._get_partition_path(
            "macro",
            date,
            series_name,
        )
        
        return self._write_parquet(df, partition_path)
    
    def read_underlying(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Read underlying data for a date range.
        
        Args:
            symbol: Instrument symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
        
        Returns:
            Combined DataFrame
        
        Examples:
            >>> store = RawStore()
            >>> df = store.read_underlying("SPY", 
            ...     datetime(2025, 2, 1), datetime(2025, 2, 28))
        """
        files = self._list_files(
            "underlying",
            symbol,
            start_date,
            end_date,
        )
        
        if not files:
            logger.warning("No files found", symbol=symbol, start=start_date, end=end_date)
            return pd.DataFrame()
        
        # Read and concatenate all files
        dfs = [pd.read_parquet(f) for f in files]
        combined = pd.concat(dfs, ignore_index=True)
        
        # Ensure consistent timezone on timestamp column and drop duplicates
        if "timestamp" in combined.columns and not combined.empty:
            try:
                combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True).dt.tz_convert("America/New_York")
                combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
            except Exception:
                pass
        
        logger.info("Read underlying data", symbol=symbol, rows=len(combined), files=len(files))
        return combined
    
    def read_options(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Read options chain data for a date range."""
        files = self._list_files("options", symbol, start_date, end_date)
        
        if not files:
            return pd.DataFrame()
        
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
    
    def read_indices(
        self,
        index_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Read index data for a date range."""
        files = self._list_files("indices", index_name, start_date, end_date)
        
        if not files:
            return pd.DataFrame()
        
        dfs = [pd.read_parquet(f) for f in files]
        return pd.concat(dfs, ignore_index=True)
    
    # ─── INTERNAL HELPERS ───────────────────────────────────────────────────
    
    def _get_partition_path(
        self,
        data_type: str,
        date: datetime,
        symbol: str,
        sub_type: Optional[str] = None,
    ) -> Path:
        """
        Get partition path for data.
        
        Format: {root}/{ data_type}/dt={YYYY-MM-DD}/instrument={SYMBOL}/[subtype=X/]
        
        Examples:
            data/raw/underlying/dt=2025-02-06/instrument=SPY/bars/
            data/raw/options/dt=2025-02-06/instrument=SPXW/
        """
        date_str = date.strftime("%Y-%m-%d")
        
        # path = (
        #     self.root_dir
        #     / data_type
        #     / f"dt={date_str}"
        #     / f"instrument={symbol}"
        # )
        # NEW (CORRECT):
        date_str = date.strftime("%Y%m%d")  # Changed format
        path = self.root_dir / data_type / symbol / date_str
        
        if sub_type:
            path = path / sub_type
        
        path.mkdir(parents=True, exist_ok=True)
        return path


    # def _get_partition_path(self, data_type: str, symbol: str, date: datetime
    # ) -> Path:
    #     """
    #     Get partition path: {root}/{type}/{symbol}/{YYYYMMDD}/
    #     """
    #     date_str = date.strftime("%Y%m%d")
    #     path = self.root_dir / data_type / symbol / date_str
    #     path.mkdir(parents=True, exist_ok=True)
    #     return path
    
    def _write_parquet(
        self,
        df: pd.DataFrame,
        partition_path: Path,
    ) -> Path:
        """
        Write DataFrame to partitioned parquet file.
        
        Args:
            df: Data to write
            partition_path: Partition directory
        
        Returns:
            Path to written file
        """
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"part-{timestamp}.parquet"
        filepath = partition_path / filename
        
        # Write with compression
        df.to_parquet(
            filepath,
            compression=self.compression,
            index=False,
        )
        
        # Calculate file hash for manifest
        file_hash = self._calculate_hash(filepath)
        
        logger.info(
            "Wrote parquet file",
            filepath=str(filepath),
            rows=len(df),
            size_mb=round(filepath.stat().st_size / 1024 / 1024, 2),
            hash=file_hash[:16],
        )
        
        return filepath
        
    def _list_files(
        self,
        data_type: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Path]:
        """
        List all files in a date range.
        
        Args:
            data_type: Data type directory
            symbol: Instrument symbol
            start_date: Start date
            end_date: End date
        
        Returns:
            List of file paths
        """
        files = []
        
        # Iterate through each date in range
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y%m%d")  # ✅ NEW: YYYYMMDD format
            partition_dir = (
                self.root_dir
                / data_type
                / symbol           # ✅ NEW: Symbol first
                / date_str         # ✅ NEW: Then date
            )
            
            if partition_dir.exists():
                # Get all parquet files in partition (including subdirs like bars/)
                files.extend(partition_dir.glob("**/*.parquet"))
            
            current_date += pd.Timedelta(days=1)
        
        return sorted(files)
    def _calculate_hash(self, filepath: Path) -> str:
        """Calculate hash (xxhash or sha256 fallback) for a file."""
        if xxhash is not None:
            hasher = xxhash.xxh64()
        else:
            hasher = hashlib.sha256()
        
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        
        return hasher.hexdigest()


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

_default_store: Optional[RawStore] = None


def get_raw_store() -> RawStore:
    """Get singleton raw store instance."""
    global _default_store
    
    if _default_store is None:
        _default_store = RawStore()
    
    return _default_store
