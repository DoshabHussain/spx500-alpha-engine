"""
Pydantic schemas for data validation - FIXED for Pydantic 2.6+

Replace your current src/core/schema.py with this file.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict


# ─── ENUMS ──────────────────────────────────────────────────────────────────

class AssetType(str, Enum):
    """Asset type classification."""
    EQUITY = "equity"
    INDEX = "index"
    OPTION = "option"
    FUTURE = "future"
    ETF = "etf"


class OptionType(str, Enum):
    """Option type (call or put)."""
    CALL = "call"
    PUT = "put"


class OptionStyle(str, Enum):
    """Option exercise style."""
    AMERICAN = "american"
    EUROPEAN = "european"


class BarInterval(str, Enum):
    """Time bar intervals."""
    TICK = "tick"
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_30 = "30min"
    HOUR_1 = "1h"
    DAILY = "1d"


class TargetType(str, Enum):
    """Label/target type."""
    DIRECTIONAL = "directional"
    VOLATILITY = "volatility"
    RETURN = "return"
    RANKING = "ranking"


# ─── CORE DATA MODELS ───────────────────────────────────────────────────────

class OHLCVBar(BaseModel):
    """Single OHLCV (Open-High-Low-Close-Volume) bar."""
    model_config = ConfigDict(frozen=True)
    
    timestamp: datetime = Field(..., description="Bar timestamp (UTC)")
    symbol: str = Field(..., description="Instrument symbol")
    open: float = Field(..., ge=0, description="Opening price")
    high: float = Field(..., ge=0, description="Highest price")
    low: float = Field(..., ge=0, description="Lowest price")
    close: float = Field(..., ge=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    interval: BarInterval = Field(..., description="Bar interval")
    
    vwap: Optional[float] = Field(None, description="Volume-weighted average price")
    trades: Optional[int] = Field(None, ge=0, description="Number of trades")
    
    @field_validator("high")
    @classmethod
    def high_gte_low(cls, v: float, info) -> float:
        """Ensure high >= low."""
        if "low" in info.data and v < info.data["low"]:
            raise ValueError(f"high ({v}) must be >= low ({info.data['low']})")
        return v
    
    @field_validator("close")
    @classmethod
    def close_in_range(cls, v: float, info) -> float:
        """Ensure close is between low and high."""
        if "low" in info.data and "high" in info.data:
            low, high = info.data["low"], info.data["high"]
            if not (low <= v <= high):
                raise ValueError(f"close ({v}) must be between low ({low}) and high ({high})")
        return v


class OptionQuote(BaseModel):
    """Single option quote (bid/ask/last)."""
    model_config = ConfigDict(frozen=True)
    
    timestamp: datetime
    underlying_symbol: str
    expiration: datetime
    strike: float = Field(..., ge=0)
    option_type: OptionType
    
    bid: Optional[float] = Field(None, ge=0)
    ask: Optional[float] = Field(None, ge=0)
    last: Optional[float] = Field(None, ge=0)
    
    volume: Optional[int] = Field(None, ge=0)
    open_interest: Optional[int] = Field(None, ge=0)
    
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    
    iv: Optional[float] = Field(None, ge=0, le=10)
    
    @field_validator("ask")
    @classmethod
    def ask_gte_bid(cls, v: Optional[float], info) -> Optional[float]:
        """Ensure ask >= bid if both present."""
        if v is not None and "bid" in info.data and info.data["bid"] is not None:
            if v < info.data["bid"]:
                raise ValueError(f"ask ({v}) must be >= bid ({info.data['bid']})")
        return v


class OptionsChain(BaseModel):
    """Full options chain snapshot for an underlying."""
    timestamp: datetime
    underlying_symbol: str
    underlying_price: float = Field(..., ge=0)
    quotes: List[OptionQuote]
    
    days_to_expiration: Optional[int] = Field(None, ge=0)
    expiration_date: Optional[datetime] = None


# ─── FEATURES & LABELS ──────────────────────────────────────────────────────

class Feature(BaseModel):
    """Single engineered feature value."""
    name: str
    value: Union[float, int, bool, str]
    timestamp: datetime
    
    group: Optional[str] = None
    window: Optional[int] = None
    
    @field_validator("value")
    @classmethod
    def check_numeric_range(cls, v: Union[float, int, bool, str]) -> Union[float, int, bool, str]:
        """Check for NaN/Inf in numeric features."""
        if isinstance(v, float):
            if not (-1e10 < v < 1e10):
                raise ValueError(f"Feature value out of reasonable range: {v}")
        return v


class FeatureVector(BaseModel):
    """Collection of features for a single timestamp."""
    timestamp: datetime
    symbol: str
    features: Dict[str, Union[float, int, bool]]
    
    interval: Optional[BarInterval] = None
    feature_version: Optional[str] = None


class Label(BaseModel):
    """Single label/target value."""
    name: str
    value: Union[float, int, str]
    timestamp: datetime
    horizon: str
    target_type: TargetType
    
    future_value: Optional[float] = None


# ─── METADATA & MANIFESTS ───────────────────────────────────────────────────

class DataSource(BaseModel):
    """Data source metadata."""
    name: str
    version: Optional[str] = None
    url: Optional[str] = None
    api_key_required: bool = False
    rate_limit: Optional[Dict[str, int]] = None


class FileManifest(BaseModel):
    """File manifest for reproducibility."""
    filepath: str
    file_hash: str
    file_size: int
    created_at: datetime
    
    row_count: Optional[int] = None
    schema_version: Optional[str] = None
    data_source: Optional[str] = None


class PipelineRun(BaseModel):
    """Pipeline run metadata - FIXED for Pydantic 2.6+"""
    
    # Allow arbitrary types to avoid Pydantic errors with complex Dict types
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    run_id: str
    pipeline_name: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str
    
    # FIXED: Changed from Dict[str, Any] to avoid Pydantic schema generation issues
    config: Dict[str, str] = Field(default_factory=dict)
    
    input_files: List[FileManifest] = Field(default_factory=list)
    output_files: List[FileManifest] = Field(default_factory=list)
    
    rows_processed: Optional[int] = None
    errors: Optional[List[str]] = None


# ─── QUALITY METRICS ────────────────────────────────────────────────────────

class DataQualityMetrics(BaseModel):
    """Data quality metrics for a dataset."""
    timestamp: datetime
    dataset_name: str
    
    total_rows: int = Field(..., ge=0)
    missing_rows: int = Field(..., ge=0)
    missing_pct: float = Field(..., ge=0, le=100)
    
    latest_timestamp: datetime
    staleness_minutes: float = Field(..., ge=0)
    
    anomaly_count: int = Field(..., ge=0)
    anomalies: Optional[List[str]] = None
    
    status: str


# ─── VALIDATION HELPERS ─────────────────────────────────────────────────────

def validate_ohlcv_bar(data: Dict) -> OHLCVBar:
    """Validate and parse OHLCV bar data."""
    return OHLCVBar(**data)


def validate_feature_vector(data: Dict) -> FeatureVector:
    """Validate and parse feature vector."""
    return FeatureVector(**data)


def validate_label(data: Dict) -> Label:
    """Validate and parse label."""
    return Label(**data)