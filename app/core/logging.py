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
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def filter(self, record):
        record.correlation_id = correlation_id.get() or str(uuid.uuid4())
        return True


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, 'correlation_id', None),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'correlation_id']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


def setup_logging():
    """Configure structured logging for the application"""
    
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
            '%(asctime)s - %(name)s - %(levelname)s - [%(correlation_id)s] - %(message)s'
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
    """Get current correlation ID or generate a new one"""
    current_id = correlation_id.get()
    if not current_id:
        current_id = str(uuid.uuid4())
        correlation_id.set(current_id)
    return current_id


def set_correlation_id(new_id: str):
    """Set correlation ID for current context"""
    correlation_id.set(new_id)


def log_api_call(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: Optional[str] = None,
    **kwargs
):
    """Log API call with structured data"""
    logger = logging.getLogger("api")
    
    log_data = {
        "event_type": "api_call",
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": duration_ms,
        "user_id": user_id,
        **kwargs
    }
    
    if status_code >= 400:
        logger.error("API call failed", extra=log_data)
    else:
        logger.info("API call completed", extra=log_data)


def log_business_event(
    event_type: str,
    event_data: Dict[str, Any],
    user_id: Optional[str] = None
):
    """Log business events for analytics"""
    logger = logging.getLogger("business")
    
    log_data = {
        "event_type": event_type,
        "user_id": user_id,
        **event_data
    }
    
    logger.info(f"Business event: {event_type}", extra=log_data)


def log_error(
    error: Exception,
    context: Dict[str, Any] = None,
    user_id: Optional[str] = None
):
    """Log errors with context"""
    logger = logging.getLogger("error")
    
    log_data = {
        "event_type": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "user_id": user_id,
        **(context or {})
    }
    
    logger.error("Application error occurred", extra=log_data, exc_info=error)


# Initialize logging
logger = setup_logging()