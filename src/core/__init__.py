"""
Core utilities for SPX500 forecasting system.

Provides:
- Configuration management (config.py)
- Time & timezone handling (time.py)
- Trading calendar logic (calendar.py)
- Data schemas & validation (schema.py)
- Structured logging (logging.py)
"""

from src.core.calendar import (
    TradingCalendar,
    get_calendar,
    is_rth,
    is_eth,
    get_current_session,
    get_next_market_open,
    get_next_market_close,
)
from src.core.config import (
    Config,
    ConfigLoader,
    get_config,
    get_api_key,
    is_dev_mode,
    is_debug_enabled,
)
from src.core.logging import (
    configure_logging,
    get_logger,
    LogContext,
    bind_context,
    clear_context,
    log_execution_time,
    DataQualityLogger,
    PipelineLogger,
)
from src.core.schema import (
    AssetType,
    OptionType,
    BarInterval,
    TargetType,
    OHLCVBar,
    OptionQuote,
    OptionsChain,
    Feature,
    FeatureVector,
    Label,
    DataQualityMetrics,
)
from src.core.time import (
    to_utc,
    from_utc,
    now_utc,
    now_ny,
    is_market_hours,
    market_open_time,
    market_close_time,
    format_timestamp,
    parse_timestamp,
)

__all__ = [
    # Config
    "Config",
    "ConfigLoader",
    "get_config",
    "get_api_key",
    "is_dev_mode",
    "is_debug_enabled",
    # Time
    "to_utc",
    "from_utc",
    "now_utc",
    "now_ny",
    "is_market_hours",
    "market_open_time",
    "market_close_time",
    "format_timestamp",
    "parse_timestamp",
    # Calendar
    "TradingCalendar",
    "get_calendar",
    "is_rth",
    "is_eth",
    "get_current_session",
    "get_next_market_open",
    "get_next_market_close",
    # Schema
    "AssetType",
    "OptionType",
    "BarInterval",
    "TargetType",
    "OHLCVBar",
    "OptionQuote",
    "OptionsChain",
    "Feature",
    "FeatureVector",
    "Label",
    "DataQualityMetrics",
    # Logging
    "configure_logging",
    "get_logger",
    "LogContext",
    "bind_context",
    "clear_context",
    "log_execution_time",
    "DataQualityLogger",
    "PipelineLogger",
]
