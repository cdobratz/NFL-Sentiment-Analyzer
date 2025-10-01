"""
Custom exception classes for the NFL Sentiment Analyzer API.
"""

from typing import Optional, Dict, Any


class APIError(Exception):
    """Base class for API errors"""
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "API_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIError):
    """Raised when input validation fails"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details
        )


class AuthenticationError(APIError):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication required"):
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(APIError):
    """Raised when authorization fails"""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR"
        )


class NotFoundError(APIError):
    """Raised when a resource is not found"""
    
    def __init__(self, resource: str, identifier: str = None):
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"
        
        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND"
        )


class ConflictError(APIError):
    """Raised when a resource conflict occurs"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=409,
            error_code="CONFLICT_ERROR",
            details=details
        )


class RateLimitError(APIError):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED"
        )


class ExternalServiceError(APIError):
    """Raised when external service fails"""
    
    def __init__(self, service: str, message: str = None):
        error_message = f"External service error: {service}"
        if message:
            error_message += f" - {message}"
        
        super().__init__(
            message=error_message,
            status_code=502,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service}
        )


class ServiceUnavailableError(APIError):
    """Raised when service is temporarily unavailable"""
    
    def __init__(self, message: str = "Service temporarily unavailable"):
        super().__init__(
            message=message,
            status_code=503,
            error_code="SERVICE_UNAVAILABLE"
        )


class DatabaseError(APIError):
    """Raised when database operations fail"""
    
    def __init__(self, message: str, operation: str = None):
        error_message = "Database error"
        if operation:
            error_message += f" during {operation}"
        if message:
            error_message += f": {message}"
        
        super().__init__(
            message=error_message,
            status_code=500,
            error_code="DATABASE_ERROR",
            details={"operation": operation} if operation else None
        )


class MLModelError(APIError):
    """Raised when ML model operations fail"""
    
    def __init__(self, message: str, model_name: str = None):
        error_message = "ML model error"
        if model_name:
            error_message += f" in {model_name}"
        if message:
            error_message += f": {message}"
        
        super().__init__(
            message=error_message,
            status_code=500,
            error_code="ML_MODEL_ERROR",
            details={"model_name": model_name} if model_name else None
        )


class DataIngestionError(APIError):
    """Raised when data ingestion fails"""
    
    def __init__(self, message: str, source: str = None):
        error_message = "Data ingestion error"
        if source:
            error_message += f" from {source}"
        if message:
            error_message += f": {message}"
        
        super().__init__(
            message=error_message,
            status_code=500,
            error_code="DATA_INGESTION_ERROR",
            details={"source": source} if source else None
        )


class ConfigurationError(APIError):
    """Raised when configuration is invalid"""
    
    def __init__(self, message: str, config_key: str = None):
        error_message = "Configuration error"
        if config_key:
            error_message += f" for {config_key}"
        if message:
            error_message += f": {message}"
        
        super().__init__(
            message=error_message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key} if config_key else None
        )