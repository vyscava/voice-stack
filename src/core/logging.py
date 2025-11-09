"""
Logging configuration for Voice Stack services.

This module provides:
- Structured JSON logging support for better log parsing
- Service-specific loggers (ASR and TTS)
- Third-party library log level configuration
"""

import logging
import sys
from typing import Any

from .settings import get_settings

settings = get_settings()


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON objects for easy parsing and analysis.
    Includes standard fields plus any extra fields from the log record.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string."""
        import json
        from datetime import datetime, timezone

        # Base log structure
        log_dict: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add source location info for errors
        if record.levelno >= logging.WARNING:
            log_dict["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_dict["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None,
            }

        # Add any extra fields from the record
        # These come from logger.info("message", extra={"key": "value"})
        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key
            not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "taskName",
            }
        }

        if extra_fields:
            log_dict.update(extra_fields)

        return json.dumps(log_dict, default=str)


class HumanReadableFormatter(logging.Formatter):
    """
    Enhanced human-readable formatter with context support.

    Provides a clear, readable format for development while still
    including important context fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a human-readable string."""
        # Base format
        base_format = super().format(record)

        # Add extra fields if present
        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key
            not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "message",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "thread",
                "threadName",
                "exc_info",
                "exc_text",
                "stack_info",
                "taskName",
            }
        }

        if extra_fields:
            # Format extra fields as key=value pairs
            extras = " | ".join(f"{k}={v}" for k, v in extra_fields.items())
            return f"{base_format} | {extras}"

        return base_format


def setup_logging(level: str, use_json: bool = False) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        use_json: If True, use JSON structured logging; otherwise use human-readable format
    """
    # Choose formatter based on configuration
    if use_json:
        formatter = StructuredFormatter()
    else:
        formatter = HumanReadableFormatter(
            fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level.upper())

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler with our formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Configure uvicorn loggers
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(level.upper())

    # Configure third-party library log levels
    # Reduce noise from verbose libraries
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("TTS").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


# Determine if we should use JSON logging
# Use JSON logging in production (ENV=prod) or if explicitly enabled
use_json_logging = settings.ENV.lower() == "prod"

# Set up logging for the service
setup_logging(settings.LOG_LEVEL, use_json=use_json_logging)

# Create service-specific loggers
logger_asr = logging.getLogger(settings.ASR_LOG_NAME)
logger_tts = logging.getLogger(settings.TTS_LOG_NAME)

# Log the logging configuration on startup
if use_json_logging:
    logger_asr.info("JSON structured logging enabled")
    logger_tts.info("JSON structured logging enabled")
else:
    logger_asr.info("Human-readable logging enabled (set ENV=prod for JSON logging)")
    logger_tts.info("Human-readable logging enabled (set ENV=prod for JSON logging)")
