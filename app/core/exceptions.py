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
        details: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the APIError with a message, HTTP status code, error code, and optional details.

        Parameters:
            message (str): Human-readable error message shown to callers and stored as the exception message.
            status_code (int): HTTP status code associated with the error (defaults to 500).
            error_code (str): Machine-friendly error code for clients and logs (defaults to "API_ERROR").
            details (Optional[Dict[str, Any]]): Optional structured additional information about the error (defaults to empty dict).
        """
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(APIError):
    """Raised when input validation fails"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize a ValidationError used for input validation failures.

        Parameters:
            message (str): Human-readable error message describing the validation failure.
            details (Optional[Dict[str, Any]]): Optional dictionary with additional context about the validation error (e.g., field-level errors).
        """
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class AuthenticationError(APIError):
    """Raised when authentication fails"""

    def __init__(self, message: str = "Authentication required"):
        """
        Create an AuthenticationError indicating authentication is required.

        Parameters:
                message (str): Optional custom error message; defaults to "Authentication required". Sets the exception's HTTP status_code to 401 and error_code to "AUTHENTICATION_ERROR".
        """
        super().__init__(
            message=message, status_code=401, error_code="AUTHENTICATION_ERROR"
        )


class AuthorizationError(APIError):
    """Raised when authorization fails"""

    def __init__(self, message: str = "Insufficient permissions"):
        """
        Initialize an AuthorizationError representing insufficient permissions.

        This exception is populated with an HTTP status code of 403 and an error code of "AUTHORIZATION_ERROR".

        Parameters:
            message (str): Human-readable error message; defaults to "Insufficient permissions".
        """
        super().__init__(
            message=message, status_code=403, error_code="AUTHORIZATION_ERROR"
        )


class NotFoundError(APIError):
    """Raised when a resource is not found"""

    def __init__(self, resource: str, identifier: Optional[str] = None):
        """
        Initialize a NotFoundError for a missing resource.

        Parameters:
            resource (str): Name or type of the resource that was not found.
            identifier (str, optional): Optional identifier (e.g., ID or key) to append to the error message, producing "<resource> not found: <identifier>".

        Notes:
            The exception is initialized with HTTP status code 404 and error code "NOT_FOUND".
        """
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"

        super().__init__(message=message, status_code=404, error_code="NOT_FOUND")


class ConflictError(APIError):
    """Raised when a resource conflict occurs"""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Represents an HTTP 409 Conflict error raised when a resource conflict occurs.

        Parameters:
            message (str): Human-readable description of the conflict.
            details (Optional[Dict[str, Any]]): Optional structured metadata providing additional context about the conflict.
        """
        super().__init__(
            message=message,
            status_code=409,
            error_code="CONFLICT_ERROR",
            details=details,
        )


class RateLimitError(APIError):
    """Raised when rate limit is exceeded"""

    def __init__(self, message: str = "Rate limit exceeded"):
        """
        Initialize a RateLimitError representing an HTTP 429 rate limit condition.

        Parameters:
            message (str): Human-readable error message. Defaults to "Rate limit exceeded".
        """
        super().__init__(
            message=message, status_code=429, error_code="RATE_LIMIT_EXCEEDED"
        )


class ExternalServiceError(APIError):
    """Raised when external service fails"""

    def __init__(self, service: str, message: Optional[str] = None):
        """
        Initialize an APIError representing a failure in an external service.

        Parameters:
            service (str): Name or identifier of the external service that failed; included in the error details.
            message (str, optional): Additional context appended to the error message (appended after " - ").

        Description:
            Sets the HTTP status code to 502, the error code to "EXTERNAL_SERVICE_ERROR", and populates the error details with the `service`. The formatted error message starts with "External service error: <service>" and includes the optional `message` if provided.
        """
        error_message = f"External service error: {service}"
        if message:
            error_message += f" - {message}"

        super().__init__(
            message=error_message,
            status_code=502,
            error_code="EXTERNAL_SERVICE_ERROR",
            details={"service": service},
        )


class ServiceUnavailableError(APIError):
    """Raised when service is temporarily unavailable"""

    def __init__(self, message: str = "Service temporarily unavailable"):
        """
        Initialize the ServiceUnavailableError with an optional custom message.

        Parameters:
            message (str): Human-readable error message describing the service unavailability. Defaults to "Service temporarily unavailable".
        """
        super().__init__(
            message=message, status_code=503, error_code="SERVICE_UNAVAILABLE"
        )


class DatabaseError(APIError):
    """Raised when database operations fail"""

    def __init__(self, message: str, operation: str = None):
        """
        Create a DatabaseError that records an explanatory message and optional operation context.

        Parameters:
            message: Human-readable description of the database failure; appended to the base "Database error" prefix.
            operation: Optional name of the database operation (e.g., "insert", "update") to include in the constructed message and in the error's `details` mapping.
        """
        error_message = "Database error"
        if operation:
            error_message += f" during {operation}"
        if message:
            error_message += f": {message}"

        super().__init__(
            message=error_message,
            status_code=500,
            error_code="DATABASE_ERROR",
            details={"operation": operation} if operation else None,
        )


class MLModelError(APIError):
    """Raised when ML model operations fail"""

    def __init__(self, message: str, model_name: Optional[str] = None):
        """
        Constructs an MLModelError containing a composed error message and optional model context.

        Parameters:
            message (str): Human-readable description of the model error; appended to the composed message when provided.
            model_name (str, optional): Name of the ML model related to the error; included in the message and added to the error `details` when provided.
        """
        error_message = "ML model error"
        if model_name:
            error_message += f" in {model_name}"
        if message:
            error_message += f": {message}"

        super().__init__(
            message=error_message,
            status_code=500,
            error_code="ML_MODEL_ERROR",
            details={"model_name": model_name} if model_name else None,
        )


class DataIngestionError(APIError):
    """Raised when data ingestion fails"""

    def __init__(self, message: str, source: Optional[str] = None):
        """
        Initialize a DataIngestionError for failures that occur while ingesting data.

        This error sets the HTTP status code to 500 and the error code to "DATA_INGESTION_ERROR"; when provided, the `source` is included in the error details and incorporated into the error message.

        Parameters:
            message (str): Human-readable description of the ingestion failure.
            source (str, optional): Name or identifier of the data source where the failure occurred.
        """
        error_message = "Data ingestion error"
        if source:
            error_message += f" from {source}"
        if message:
            error_message += f": {message}"

        super().__init__(
            message=error_message,
            status_code=500,
            error_code="DATA_INGESTION_ERROR",
            details={"source": source} if source else None,
        )


class ConfigurationError(APIError):
    """Raised when configuration is invalid"""

    def __init__(self, message: str, config_key: Optional[str] = None):
        """
        Initialize the ConfigurationError with optional configuration key context.

        Constructs an error message beginning with "Configuration error", appends " for <config_key>" when a config_key is provided, and appends ": <message>" when a message is provided. The instance will carry an HTTP status code of 500, an error_code of "CONFIGURATION_ERROR", and include the `config_key` in `details` when present.

        Parameters:
            message (str): Optional additional context to append to the error message.
            config_key (str, optional): The configuration key related to the error.
        """
        error_message = "Configuration error"
        if config_key:
            error_message += f" for {config_key}"
        if message:
            error_message += f": {message}"

        super().__init__(
            message=error_message,
            status_code=500,
            error_code="CONFIGURATION_ERROR",
            details={"config_key": config_key} if config_key else None,
        )
