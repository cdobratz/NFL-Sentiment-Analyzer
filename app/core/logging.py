"""
Structured logging configuration with correlation IDs and monitoring integration.
"""

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any
from datetime import datetime
import json

from .config import settings

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records"""

    def filter(self, record):
        """
        Injects a `correlation_id` attribute into the given log record, using the current context's correlation ID or generating a new UUID when none is present.
        
        Parameters:
            record (logging.LogRecord): The log record to augment.
        
        Returns:
            bool: `True` to allow the record to be processed by subsequent handlers.
        """
        record.correlation_id = correlation_id.get() or str(uuid.uuid4())
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        """
        Format a logging.LogRecord into a JSON string containing structured log fields.
        
        The resulting JSON object includes core fields such as timestamp (UTC ISO 8601 with a trailing "Z"), level, logger name, message, correlation_id, module, function, and line number. If the record contains exception information, an `exception` field is included. Any additional attributes from the LogRecord that are not part of the formatter's excluded set are merged into the JSON object as extra fields.
        
        Parameters:
            record (logging.LogRecord): The log record to format.
        
        Returns:
            str: A JSON string representing the structured log entry.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", None),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "correlation_id",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry)


def setup_logging():
    """
    Configure the root logger for structured application logging.
    
    Sets root and console handler levels based on settings.debug, attaches a correlation ID filter, selects a human-readable formatter in debug mode or a JSON structured formatter otherwise, and silences noisy third-party loggers.
    
    Returns:
        logging.Logger: The configured root logger instance.
    """

    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO if not settings.debug else logging.DEBUG)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO if not settings.debug else logging.DEBUG)

    # Add correlation ID filter
    correlation_filter = CorrelationIdFilter()
    console_handler.addFilter(correlation_filter)

    # Set formatter
    if settings.debug:
        # Human-readable format for development
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s"
        )
    else:
        # JSON format for production
        formatter = StructuredFormatter()

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Configure specific loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return root_logger


def get_correlation_id() -> str:
    """
    Retrieve the current correlation identifier for the current execution context.
    
    If no correlation ID is present, a new UUID4 string is generated, stored in the context, and returned.
    
    Returns:
        str: The correlation ID for the current context.
    """
    current_id = correlation_id.get()
    if not current_id:
        current_id = str(uuid.uuid4())
        correlation_id.set(current_id)
    return current_id


def set_correlation_id(new_id: str):
    """
    Set the correlation ID for the current execution context.
    
    Parameters:
        new_id (str): Correlation identifier to store and propagate for the current context; used by logging to associate log records.
    """
    correlation_id.set(new_id)


def log_api_call(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None,
    **kwargs,
):
    """
    Record a structured log entry for an HTTP API call.
    
    Builds a log payload with event_type "api_call" containing method, path, status_code,
    duration_ms, optional user_id and any additional fields passed via kwargs. Logs with
    the "api" logger at level ERROR if status_code is 400 or greater, otherwise at INFO.
    
    Parameters:
        method (str): HTTP method (e.g., "GET", "POST").
        path (str): Request path or URL.
        status_code (int): HTTP response status code.
        duration_ms (float): Request duration in milliseconds.
        user_id (Optional[str]): Identifier of the user performing the call, if available.
        **kwargs: Additional structured fields to include in the log entry.
    """
    logger = logging.getLogger("api")

    log_data = {
        "event_type": "api_call",
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "user_id": user_id,
        **kwargs,
    }

    if status_code >= 400:
        logger.error("API call failed", extra=log_data)
    else:
        logger.info("API call completed", extra=log_data)


def log_business_event(
    event_type: str, event_data: Dict[str, Any], user_id: Optional[str] = None
):
    """
    Record a business analytics event with optional user context.
    
    Parameters:
    	event_type (str): A short name identifying the business event.
    	event_data (Dict[str, Any]): Additional attributes describing the event.
    	user_id (Optional[str]): Identifier of the user associated with the event, if available.
    """
    logger = logging.getLogger("business")

    log_data = {"event_type": event_type, "user_id": user_id, **event_data}

    logger.info(f"Business event: {event_type}", extra=log_data)


def log_error(
    error: Exception, context: Dict[str, Any] = None, user_id: Optional[str] = None
):
    """
    Log an application error with contextual metadata.
    
    Records the exception's type, message, and traceback along with any provided contextual fields and optional user identifier.
    
    Parameters:
        error (Exception): The exception to log.
        context (Dict[str, Any], optional): Additional key-value metadata to include in the log entry.
        user_id (Optional[str], optional): Identifier of the user related to the error, if available.
    """
    logger = logging.getLogger("error")

    log_data = {
        "event_type": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "user_id": user_id,
        **(context or {}),
    }

    logger.error("Application error occurred", extra=log_data, exc_info=error)


# Initialize logging
logger = setup_logging()
