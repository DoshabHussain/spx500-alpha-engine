"""
Structured logging with JSON output.

Features:
- Structured JSON logs for machine parsing
- Context-aware logging (attach request IDs, user context, etc.)
- Performance logging (timing decorators)
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Output to console + file with rotation

Uses structlog for structured logging.
"""

import json
import logging
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import structlog
    from structlog.stdlib import LoggerFactory
    HAS_STRUCTLOG = True
except ImportError:
    structlog = None
    HAS_STRUCTLOG = False

_global_context: Dict[str, Any] = {}

class FallbackBoundLogger:
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}

    def bind(self, **kwargs: Any) -> "FallbackBoundLogger":
        merged = {**self.context, **kwargs}
        return FallbackBoundLogger(self.logger.name, merged)

    def _log(self, level: int, event: Any, **kwargs: Any) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "logger": self.logger.name,
            "event": str(event),
            **_global_context,
            **self.context,
            **kwargs,
        }
        self.logger.log(level, json.dumps(payload, default=str))

    def debug(self, event: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, event, **kwargs)

    def info(self, event: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, event, **kwargs)

    def warning(self, event: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, event, **kwargs)

    def error(self, event: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, event, **kwargs)

    def critical(self, event: Any, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, event, **kwargs)


# ─── CONFIGURATION ──────────────────────────────────────────────────────────

def configure_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    json_format: bool = True,
) -> None:
    """
    Configure structured logging.
    
    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for log output
        json_format: Use JSON formatting (True) or human-readable (False)
    
    Examples:
        >>> configure_logging(log_level="DEBUG", log_file=Path("runs/logs/app.log"))
    """
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        stream=sys.stdout,
    )
    
    # Add file handler if specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        logging.root.addHandler(file_handler)
    
    # Configure structlog if available
    if HAS_STRUCTLOG and structlog is not None:
        if json_format:
            # JSON output for production
            processors = [
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ]
        else:
            # Human-readable output for development
            processors = [
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
                structlog.dev.ConsoleRenderer(),
            ]
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )


# ─── LOGGER FACTORY ─────────────────────────────────────────────────────────

def get_logger(name: str) -> Any:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name (usually __name__ of the module)
    
    Returns:
        Bound logger instance
    """
    if HAS_STRUCTLOG and structlog is not None:
        return structlog.get_logger(name)
    return FallbackBoundLogger(name)


# ─── CONTEXT MANAGEMENT ─────────────────────────────────────────────────────

class LogContext:
    """
    Context manager for adding context to all logs within a block.
    """
    
    def __init__(self, **context: Any) -> None:
        self.context = context
        self.token = None
    
    def __enter__(self) -> "LogContext":
        if HAS_STRUCTLOG and structlog is not None:
            self.token = structlog.contextvars.bind_contextvars(**self.context)
        else:
            _global_context.update(self.context)
        return self
    
    def __exit__(self, *args: Any) -> None:
        if HAS_STRUCTLOG and structlog is not None:
            structlog.contextvars.clear_contextvars()
        else:
            for k in self.context:
                _global_context.pop(k, None)


def bind_context(**context: Any) -> None:
    """
    Bind context variables that will be included in all subsequent logs.
    """
    if HAS_STRUCTLOG and structlog is not None:
        structlog.contextvars.bind_contextvars(**context)
    else:
        _global_context.update(context)


def clear_context() -> None:
    """Clear all bound context variables."""
    if HAS_STRUCTLOG and structlog is not None:
        structlog.contextvars.clear_contextvars()
    else:
        _global_context.clear()


# ─── PERFORMANCE LOGGING ────────────────────────────────────────────────────

def log_execution_time(logger: Optional[Any] = None) -> Callable:
    """
    Decorator to log function execution time.
    
    Args:
        logger: Optional logger (uses function's module logger if not provided)
    
    Examples:
        >>> @log_execution_time()
        ... def process_data(rows: int) -> None:
        ...     # Process data...
        ...     pass
        >>> 
        >>> process_data(1000)
        # Logs: {"event": "process_data completed", "duration_s": 2.5, ...}
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start_time = datetime.now()
            func_name = func.__name__
            
            logger.debug(f"{func_name} started")
            
            try:
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"{func_name} completed",
                    duration_s=round(duration, 3),
                )
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(
                    f"{func_name} failed",
                    duration_s=round(duration, 3),
                    error=str(e),
                    exc_info=True,
                )
                raise
        
        return wrapper
    return decorator


# ─── SPECIALIZED LOGGERS ────────────────────────────────────────────────────

class DataQualityLogger:
    """Logger specialized for data quality events."""
    
    def __init__(self, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("data_quality")
    
    def log_missing_data(
        self,
        dataset: str,
        missing_rows: int,
        total_rows: int,
        severity: str = "warning",
    ) -> None:
        """Log missing data event."""
        missing_pct = (missing_rows / total_rows * 100) if total_rows > 0 else 0
        
        log_func = getattr(self.logger, severity)
        log_func(
            "Missing data detected",
            dataset=dataset,
            missing_rows=missing_rows,
            total_rows=total_rows,
            missing_pct=round(missing_pct, 2),
        )
    
    def log_anomaly(
        self,
        dataset: str,
        anomaly_type: str,
        value: Any,
        threshold: Optional[float] = None,
    ) -> None:
        """Log anomaly detection."""
        self.logger.warning(
            "Anomaly detected",
            dataset=dataset,
            anomaly_type=anomaly_type,
            value=value,
            threshold=threshold,
        )
    
    def log_staleness(
        self,
        dataset: str,
        latest_timestamp: datetime,
        staleness_minutes: float,
    ) -> None:
        """Log stale data warning."""
        self.logger.warning(
            "Stale data detected",
            dataset=dataset,
            latest_timestamp=latest_timestamp.isoformat(),
            staleness_minutes=round(staleness_minutes, 1),
        )


class PipelineLogger:
    """Logger specialized for pipeline execution."""
    
    def __init__(self, pipeline_name: str, logger: Optional[Any] = None) -> None:
        self.logger = logger or get_logger("pipeline")
        self.pipeline_name = pipeline_name
        bind_context(pipeline=pipeline_name)
    
    def log_start(self, **metadata: Any) -> None:
        """Log pipeline start."""
        self.logger.info(
            f"Pipeline {self.pipeline_name} started",
            **metadata,
        )
    
    def log_stage(self, stage: str, **metadata: Any) -> None:
        """Log pipeline stage completion."""
        self.logger.info(
            f"Stage {stage} completed",
            stage=stage,
            **metadata,
        )
    
    def log_complete(self, rows_processed: int, duration_s: float, **metadata: Any) -> None:
        """Log pipeline completion."""
        self.logger.info(
            f"Pipeline {self.pipeline_name} completed",
            rows_processed=rows_processed,
            duration_s=round(duration_s, 3),
            **metadata,
        )
    
    def log_failure(self, error: str, **metadata: Any) -> None:
        """Log pipeline failure."""
        self.logger.error(
            f"Pipeline {self.pipeline_name} failed",
            error=error,
            **metadata,
        )


# ─── ERROR LOGGING ──────────────────────────────────────────────────────────

def log_exception(
    logger: Any,
    message: str,
    exc: Exception,
    **context: Any,
) -> None:
    """
    Log an exception with full context.
    
    Args:
        logger: Logger instance
        message: Human-readable message
        exc: Exception object
        **context: Additional context
    
    Examples:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_exception(logger, "Operation failed", e, user_id=123)
    """
    logger.error(
        message,
        error=str(exc),
        error_type=type(exc).__name__,
        exc_info=True,
        **context,
    )


# ─── INITIALIZATION ─────────────────────────────────────────────────────────

# Configure logging on module import with sensible defaults
# This can be reconfigured by calling configure_logging() explicitly
configure_logging(
    log_level="INFO",
    json_format=False,  # Human-readable for development
)
