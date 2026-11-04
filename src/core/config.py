# """
# Configuration management with YAML + environment variable support.

# Loads configuration from:
# 1. YAML files (configs/*.yaml)
# 2. Environment variables (.env)
# 3. Runtime overrides

# All configs are validated using Pydantic models.
# """

# from pathlib import Path
# from typing import Any, Dict, List, Optional

# import yaml
# from pydantic import BaseModel, Field, field_validator
# from pydantic_settings import BaseSettings, SettingsConfigDict


# # ─── PYDANTIC MODELS FOR CONFIG VALIDATION ─────────────────────────────────

# class StorageConfig(BaseModel):
#     """Data storage configuration."""
    
#     root_dir: Path = Field(default=Path("./data"))
#     runs_dir: Path = Field(default=Path("./runs"))
#     partition_by: List[str] = Field(default=["dt", "instrument"])
#     format: str = Field(default="parquet")
#     compression: str = Field(default="snappy")
#     retention_days: Dict[str, int] = Field(
#         default={"raw": 90, "aligned": 60, "curated": 30, "models": 30}
#     )


# class YFinanceConfig(BaseModel):
#     """yfinance data source configuration."""
    
#     enabled: bool = True
#     symbols: List[str] = Field(default=["SPY", "QQQ", "^VIX"])
#     interval: str = "1d"
#     period: str = "1y"
#     rate_limit: Dict[str, Any] = Field(
#         default={"calls_per_hour": 2000, "delay_seconds": 0.5}
#     )
#     retry: Dict[str, Any] = Field(
#         default={"max_attempts": 3, "backoff_factor": 2}
#     )


# class FREDConfig(BaseModel):
#     """FRED economic data configuration."""
    
#     enabled: bool = True
#     series: List[str] = Field(default=["DGS10", "DGS2", "VIXCLS", "T10Y2Y"])
#     update_frequency: str = "daily"
#     lookback_days: int = 365
#     rate_limit: Dict[str, int] = Field(default={"calls_per_minute": 120})


# class DataSourcesConfig(BaseModel):
#     """All data source configurations."""
    
#     yfinance: YFinanceConfig = Field(default_factory=YFinanceConfig)
#     fred: FREDConfig = Field(default_factory=FREDConfig)


# class TemporalGridConfig(BaseModel):
#     """Single time grid configuration."""
    
#     name: str
#     interval: str
#     enabled: bool = True


# class SessionsConfig(BaseModel):
#     """Trading session configuration."""
    
#     market_open: str = "09:30"
#     market_close: str = "16:00"
#     timezone: str = "America/New_York"
#     extended_hours: Dict[str, Any] = Field(
#         default={"enabled": False, "premarket_open": "04:00", "aftermarket_close": "20:00"}
#     )


# class TemporalConfig(BaseModel):
#     """Temporal resolution and session configuration."""
    
#     grids: List[TemporalGridConfig]
#     sessions: SessionsConfig


# class FeaturesConfig(BaseModel):
#     """Feature engineering configuration."""
    
#     enabled_groups: List[str] = Field(default=["price_action", "technical", "volume"])
#     windows: Dict[str, int] = Field(default={"short": 20, "medium": 78, "long": 390})
#     max_features: int = 50
#     check_leakage: bool = True
#     check_stability: bool = True


# class LabelHorizonConfig(BaseModel):
#     """Single label horizon configuration."""
    
#     name: str
#     periods: int
#     target_type: str = "directional"
#     thresholds: Dict[str, float] = Field(default={"up": 0.001, "down": -0.001})


# class LabelsConfig(BaseModel):
#     """Label/target configuration."""
    
#     horizons: List[LabelHorizonConfig]
#     embargo_periods: int = 2
#     purge_overlap: bool = True


# class QualityConfig(BaseModel):
#     """Data quality configuration."""
    
#     max_staleness_minutes: int = 60
#     max_missing_pct: Dict[str, float] = Field(
#         default={"critical": 0.05, "warning": 0.02}
#     )
#     anomaly_detection: Dict[str, Any] = Field(
#         default={
#             "enabled": True,
#             "methods": ["zscore", "iqr"],
#             "thresholds": {"price_spike": 0.05, "volume_spike": 10.0},
#         }
#     )
#     check_frequency: str = "1h"


# class LoggingHandlerConfig(BaseModel):
#     """Single logging handler configuration."""
    
#     enabled: bool = True
#     level: str = "INFO"
#     path: Optional[Path] = None
#     rotation: Optional[str] = None
#     retention_days: Optional[int] = None


# class LoggingConfig(BaseModel):
#     """Logging configuration."""
    
#     level: str = "INFO"
#     format: str = "json"
#     handlers: Dict[str, LoggingHandlerConfig] = Field(
#         default={
#             "console": LoggingHandlerConfig(enabled=True, level="DEBUG"),
#             "file": LoggingHandlerConfig(
#                 enabled=True,
#                 level="INFO",
#                 path=Path("runs/logs"),
#                 rotation="daily",
#                 retention_days=7,
#             ),
#         }
#     )
#     loggers: Dict[str, str] = Field(
#         default={
#             "src.ingestion": "DEBUG",
#             "src.processing": "DEBUG",
#             "src.core": "INFO",
#             "src.storage": "INFO",
#         }
#     )


# class PerformanceConfig(BaseModel):
#     """Performance and resource configuration."""
    
#     num_workers: int = 2
#     chunk_size: int = 10000
#     max_memory_gb: int = 8
#     cache: Dict[str, Any] = Field(
#         default={"enabled": True, "backend": "local", "ttl_seconds": 3600}
#     )


# class Config(BaseModel):
#     """Main configuration model."""
    
#     environment: Dict[str, Any]
#     storage: StorageConfig
#     data_sources: DataSourcesConfig
#     temporal: TemporalConfig
#     features: FeaturesConfig
#     labels: LabelsConfig
#     quality: QualityConfig
#     logging: LoggingConfig
#     performance: PerformanceConfig
#     dev: Optional[Dict[str, Any]] = None
#     features_flags: Optional[Dict[str, bool]] = None
    
#     @field_validator("environment")
#     @classmethod
#     def validate_environment(cls, v: Dict[str, Any]) -> Dict[str, Any]:
#         """Ensure environment has required fields."""
#         required = ["name", "timezone", "log_level"]
#         for field in required:
#             if field not in v:
#                 raise ValueError(f"Environment config missing required field: {field}")
#         return v


# # ─── ENVIRONMENT VARIABLES ──────────────────────────────────────────────────

# class EnvSettings(BaseSettings):
#     """Environment variables from .env file."""
    
#     model_config = SettingsConfigDict(
#         env_file=".env",
#         env_file_encoding="utf-8",
#         case_sensitive=False,
#         extra="ignore",
#     )
    
#     # Environment
#     env: str = Field(default="development", alias="ENV")
#     debug: bool = Field(default=True, alias="DEBUG")
#     log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
#     # FRED
#     fred_api_key: Optional[str] = Field(default=None, alias="FRED_API_KEY")
    
#     # ORATS (Phase 2)
#     orats_api_key: Optional[str] = Field(default=None, alias="ORATS_API_KEY")
    
#     # Polygon (Phase 2)
#     polygon_api_key: Optional[str] = Field(default=None, alias="POLYGON_API_KEY")
    
#     # Paths
#     data_root: Path = Field(default=Path("./data"), alias="DATA_ROOT")
#     runs_root: Path = Field(default=Path("./runs"), alias="RUNS_ROOT")
    
#     # Performance
#     num_workers: int = Field(default=4, alias="NUM_WORKERS")
#     max_memory_gb: int = Field(default=16, alias="MAX_MEMORY_GB")


# # ─── CONFIG LOADER ──────────────────────────────────────────────────────────

# class ConfigLoader:
#     """
#     Load and merge configuration from YAML files and environment variables.
    
#     Priority (highest to lowest):
#     1. Runtime overrides (passed to load())
#     2. Environment variables
#     3. YAML config file
#     4. Defaults in Pydantic models
#     """
    
#     def __init__(self, config_dir: Path = Path("configs")) -> None:
#         self.config_dir = config_dir
#         self.env_settings = EnvSettings()
    
#     def load(
#         self,
#         config_name: str = "dev",
#         overrides: Optional[Dict[str, Any]] = None,
#     ) -> Config:
#         """
#         Load configuration.
        
#         Args:
#             config_name: Name of YAML config file (without .yaml extension)
#             overrides: Runtime overrides (highest priority)
        
#         Returns:
#             Validated Config object
        
#         Example:
#             >>> loader = ConfigLoader()
#             >>> config = loader.load("dev")
#             >>> print(config.environment["name"])
#             development
#         """
#         # Load YAML
#         yaml_path = self.config_dir / f"{config_name}.yaml"
#         if not yaml_path.exists():
#             raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
#         with open(yaml_path) as f:
#             yaml_config = yaml.safe_load(f)
        
#         # Merge environment variables
#         self._merge_env_vars(yaml_config)
        
#         # Apply runtime overrides
#         if overrides:
#             self._deep_merge(yaml_config, overrides)
        
#         # Validate and return
#         return Config(**yaml_config)
    
#     def _merge_env_vars(self, config: Dict[str, Any]) -> None:
#         """Merge environment variables into config (in-place)."""
#         env = self.env_settings
        
#         # Override paths
#         config["storage"]["root_dir"] = str(env.data_root)
#         config["storage"]["runs_dir"] = str(env.runs_root)
        
#         # Override performance
#         config["performance"]["num_workers"] = env.num_workers
#         config["performance"]["max_memory_gb"] = env.max_memory_gb
        
#         # Override log level
#         config["logging"]["level"] = env.log_level
#         config["environment"]["log_level"] = env.log_level
    
#     def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
#         """Deep merge updates into base (in-place)."""
#         for key, value in updates.items():
#             if key in base and isinstance(base[key], dict) and isinstance(value, dict):
#                 self._deep_merge(base[key], value)
#             else:
#                 base[key] = value


# # ─── SINGLETON INSTANCE ─────────────────────────────────────────────────────

# _config_loader = ConfigLoader()
# _cached_config: Optional[Config] = None


# def get_config(
#     config_name: Optional[str] = None,
#     overrides: Optional[Dict[str, Any]] = None,
#     reload: bool = False,
# ) -> Config:
#     """
#     Get configuration (singleton pattern with caching).
    
#     Args:
#         config_name: Config file to load (default: from ENV environment variable)
#         overrides: Runtime overrides
#         reload: Force reload from disk (ignore cache)
    
#     Returns:
#         Config object
    
#     Example:
#         >>> config = get_config()
#         >>> print(config.storage.root_dir)
#         ./data
#     """
#     global _cached_config
    
#     if _cached_config is None or reload:
#         if config_name is None:
#             env = EnvSettings()
#             config_name = "dev" if env.env == "development" else "prod"
        
#         _cached_config = _config_loader.load(config_name, overrides)
    
#     return _cached_config


# # ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

# def get_api_key(service: str) -> Optional[str]:
#     """
#     Get API key for a service.
    
#     Args:
#         service: Service name (fred, orats, polygon, etc.)
    
#     Returns:
#         API key or None if not set
    
#     Example:
#         >>> fred_key = get_api_key("fred")
#     """
#     env = EnvSettings()
#     key_map = {
#         "fred": env.fred_api_key,
#         "orats": env.orats_api_key,
#         "polygon": env.polygon_api_key,
#     }
#     return key_map.get(service.lower())


# def is_dev_mode() -> bool:
#     """Check if running in development mode."""
#     env = EnvSettings()
#     return env.env == "development"


# def is_debug_enabled() -> bool:
#     """Check if debug mode is enabled."""
#     env = EnvSettings()
#     return env.debug














































"""
Configuration management with YAML + environment variable support.

Loads configuration from:
1. YAML files (configs/*.yaml)
2. Environment variables (.env)
3. Runtime overrides

All configs are validated using Pydantic models.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    class BaseSettings(BaseModel):
        def __init__(self, **kwargs):
            init_data = {}
            for field_name, field_info in self.__class__.model_fields.items():
                alias = getattr(field_info, 'alias', None) or field_name
                env_val = os.getenv(alias) or os.getenv(field_name.upper()) or os.getenv(field_name)
                if env_val is not None:
                    init_data[field_name] = env_val
            init_data.update(kwargs)
            super().__init__(**init_data)

    def SettingsConfigDict(**kwargs):
        return kwargs


# ─── PYDANTIC MODELS FOR CONFIG VALIDATION ─────────────────────────────────

class StorageConfig(BaseModel):
    """Data storage configuration."""
    
    root_dir: Path = Field(default=Path("./data"))
    runs_dir: Path = Field(default=Path("./runs"))
    partition_by: List[str] = Field(default=["dt", "instrument"])
    format: str = Field(default="parquet")
    compression: str = Field(default="snappy")
    retention_days: Dict[str, int] = Field(
        default={"raw": 90, "aligned": 60, "curated": 30, "models": 30}
    )


class YFinanceConfig(BaseModel):
    """yfinance data source configuration."""
    
    enabled: bool = True
    symbols: List[str] = Field(default=["SPY", "QQQ", "^VIX"])
    interval: str = "1d"
    period: str = "1y"
    rate_limit: Dict[str, Any] = Field(
        default={"calls_per_hour": 2000, "delay_seconds": 0.5}
    )
    retry: Dict[str, Any] = Field(
        default={"max_attempts": 3, "backoff_factor": 2}
    )


class FREDConfig(BaseModel):
    """FRED economic data configuration."""
    
    enabled: bool = True
    series: List[str] = Field(default=["DGS10", "DGS2", "VIXCLS", "T10Y2Y"])
    update_frequency: str = "daily"
    lookback_days: int = 365
    rate_limit: Dict[str, int] = Field(default={"calls_per_minute": 120})


class DataSourcesConfig(BaseModel):
    """All data source configurations."""
    
    yfinance: YFinanceConfig = Field(default_factory=YFinanceConfig)
    fred: FREDConfig = Field(default_factory=FREDConfig)


class TemporalGridConfig(BaseModel):
    """Single time grid configuration."""
    
    name: str
    interval: str
    enabled: bool = True


class SessionsConfig(BaseModel):
    """Trading session configuration."""
    
    market_open: str = "09:30"
    market_close: str = "16:00"
    timezone: str = "America/New_York"
    extended_hours: Dict[str, Any] = Field(
        default={"enabled": False, "premarket_open": "04:00", "aftermarket_close": "20:00"}
    )


class TemporalConfig(BaseModel):
    """Temporal resolution and session configuration."""
    
    grids: List[TemporalGridConfig]
    sessions: SessionsConfig


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""
    
    enabled_groups: List[str] = Field(default=["price_action", "technical", "volume"])
    windows: Dict[str, int] = Field(default={"short": 20, "medium": 78, "long": 390})
    max_features: int = 50
    check_leakage: bool = True
    check_stability: bool = True


class LabelHorizonConfig(BaseModel):
    """Single label horizon configuration."""
    
    name: str
    periods: int
    target_type: str = "directional"
    thresholds: Dict[str, float] = Field(default={"up": 0.001, "down": -0.001})


class LabelsConfig(BaseModel):
    """Label/target configuration."""
    
    horizons: List[LabelHorizonConfig]
    embargo_periods: int = 2
    purge_overlap: bool = True


class QualityConfig(BaseModel):
    """Data quality configuration."""
    
    max_staleness_minutes: int = 60
    max_missing_pct: Dict[str, float] = Field(
        default={"critical": 0.05, "warning": 0.02}
    )
    anomaly_detection: Dict[str, Any] = Field(
        default={
            "enabled": True,
            "methods": ["zscore", "iqr"],
            "thresholds": {"price_spike": 0.05, "volume_spike": 10.0},
        }
    )
    check_frequency: str = "1h"


class LoggingHandlerConfig(BaseModel):
    """Single logging handler configuration."""
    
    enabled: bool = True
    level: str = "INFO"
    path: Optional[Path] = None
    rotation: Optional[str] = None
    retention_days: Optional[int] = None


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    level: str = "INFO"
    format: str = "json"
    handlers: Dict[str, LoggingHandlerConfig] = Field(
        default={
            "console": LoggingHandlerConfig(enabled=True, level="DEBUG"),
            "file": LoggingHandlerConfig(
                enabled=True,
                level="INFO",
                path=Path("runs/logs"),
                rotation="daily",
                retention_days=7,
            ),
        }
    )
    loggers: Dict[str, str] = Field(
        default={
            "src.ingestion": "DEBUG",
            "src.processing": "DEBUG",
            "src.core": "INFO",
            "src.storage": "INFO",
        }
    )


class PerformanceConfig(BaseModel):
    """Performance and resource configuration."""
    
    num_workers: int = 2
    chunk_size: int = 10000
    max_memory_gb: int = 8
    cache: Dict[str, Any] = Field(
        default={"enabled": True, "backend": "local", "ttl_seconds": 3600}
    )


class Config(BaseModel):
    """Main configuration model."""
    
    environment: Dict[str, Any]
    storage: StorageConfig
    data_sources: DataSourcesConfig
    temporal: TemporalConfig
    features: FeaturesConfig
    labels: LabelsConfig
    quality: QualityConfig
    logging: LoggingConfig
    performance: PerformanceConfig
    dev: Optional[Dict[str, Any]] = None
    features_flags: Optional[Dict[str, bool]] = None
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        required = ["name", "timezone", "log_level"]
        for field in required:
            if field not in v:
                raise ValueError(f"Environment config missing required field: {field}")
        return v


# ─── ENVIRONMENT VARIABLES ──────────────────────────────────────────────────

class EnvSettings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    env: str = Field(default="development", alias="ENV")
    debug: bool = Field(default=True, alias="DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    fred_api_key: Optional[str] = Field(default=None, alias="FRED_API_KEY")
    orats_api_key: Optional[str] = Field(default=None, alias="ORATS_API_KEY")
    polygon_api_key: Optional[str] = Field(default=None, alias="POLYGON_API_KEY")

    data_root: Path = Field(default=Path("./data"), alias="DATA_ROOT")
    runs_root: Path = Field(default=Path("./runs"), alias="RUNS_ROOT")

    num_workers: int = Field(default=4, alias="NUM_WORKERS")
    max_memory_gb: int = Field(default=16, alias="MAX_MEMORY_GB")


# ─── CONFIG LOADER ──────────────────────────────────────────────────────────

class ConfigLoader:

    def __init__(self, config_dir: Path = Path("configs")) -> None:
        self.config_dir = config_dir
        self.env_settings = EnvSettings()

    def load(
        self,
        config_name: str = "dev",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Config:

        yaml_path = self.config_dir / f"{config_name}.yaml"

        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        # ✅ FIXED: Windows encoding issue
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f) or {}

        self._merge_env_vars(yaml_config)

        if overrides:
            self._deep_merge(yaml_config, overrides)

        return Config(**yaml_config)

    def _merge_env_vars(self, config: Dict[str, Any]) -> None:
        env = self.env_settings

        config["storage"]["root_dir"] = str(env.data_root)
        config["storage"]["runs_dir"] = str(env.runs_root)

        config["performance"]["num_workers"] = env.num_workers
        config["performance"]["max_memory_gb"] = env.max_memory_gb

        config["logging"]["level"] = env.log_level
        config["environment"]["log_level"] = env.log_level

    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value


# ─── SINGLETON INSTANCE ─────────────────────────────────────────────────────

_config_loader = ConfigLoader()
_cached_config: Optional[Config] = None


def get_config(
    config_name: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    reload: bool = False,
) -> Config:

    global _cached_config

    if _cached_config is None or reload:

        if config_name is None:
            env = EnvSettings()
            config_name = "dev" if env.env == "development" else "prod"

        _cached_config = _config_loader.load(config_name, overrides)

    return _cached_config


# ─── CONVENIENCE FUNCTIONS ──────────────────────────────────────────────────

def get_api_key(service: str) -> Optional[str]:

    env = EnvSettings()

    key_map = {
        "fred": env.fred_api_key,
        "orats": env.orats_api_key,
        "polygon": env.polygon_api_key,
    }

    return key_map.get(service.lower())


def is_dev_mode() -> bool:

    env = EnvSettings()
    return env.env == "development"


def is_debug_enabled() -> bool:

    env = EnvSettings()
    return env.debug