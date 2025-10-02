"""
Custom middleware for request handling, logging, and monitoring.
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging
from jose import jwt

from .logging import set_correlation_id, log_api_call, log_error
from .exceptions import APIError
from .config import settings

logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to handle correlation IDs for request tracing"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate correlation ID
        """
        Ensure each request has a correlation ID and propagate it through request state, logging context, and the response.

        If the incoming request includes an `X-Correlation-ID` header, that value is used; otherwise a new UUID is generated. The correlation ID is stored on `request.state.correlation_id` and set in the logging/context via `set_correlation_id`, and the same ID is injected into the response `X-Correlation-ID` header.

        @returns
        Response with the `X-Correlation-ID` header set to the correlation ID used for the request.
        """
        correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())

        # Set correlation ID in context
        set_correlation_id(correlation_id)

        # Add correlation ID to request state
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        return response


class AuthContextMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and store authentication context"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Extract authentication information from incoming request headers and attach it to request.state for downstream middleware and handlers.

        This middleware:
        - Initializes request.state.user to None and request.state.api_key to None.
        - If an X-API-Key header is present and valid, stores the validated API key object on request.state.api_key.
        - If an Authorization header contains a Bearer JWT and the token decodes successfully, stores the token's `sub` claim on request.state.user_id for downstream use (e.g., rate limiting).
        - Silently ignores invalid or malformed JWTs (does not raise).

        Returns:
            The HTTP response returned by the next handler in the middleware chain.
        """
        from .api_keys import api_key_manager
        from .dependencies import get_current_user, get_database, get_redis
        from fastapi.security import HTTPBearer
        from jose import jwt, JWTError

        # Initialize auth context
        request.state.user = None
        request.state.api_key = None

        # Check for API key
        api_key_header = request.headers.get("X-API-Key")
        if api_key_header:
            api_key = await api_key_manager.validate_api_key(api_key_header)
            if api_key:
                request.state.api_key = api_key

        # Check for JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            try:
                # Decode token to get user info (simplified)
                payload = jwt.decode(
                    token, settings.secret_key, algorithms=[settings.algorithm]
                )
                user_id = payload.get("sub")
                if user_id:
                    # Store user ID for rate limiting (full user object loaded by dependencies)
                    request.state.user_id = user_id
            except JWTError:
                pass

        return await call_next(request)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all API requests and responses"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log incoming requests and their outcomes, including duration and contextual metadata.

        Logs a successful response with method, path, status code, duration (ms), user_id (if present), query parameters, user agent, and client IP. If an exception occurs, logs the error with the same contextual fields and re-raises the exception.

        Returns:
            Response: The response returned by the downstream handler.

        Raises:
            Exception: Re-raises any exception raised by the downstream handler after logging it.
        """
        start_time = time.time()

        # Extract user info if available
        user_id = getattr(request.state, "user_id", None)

        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Log API call
            log_api_call(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
                user_id=user_id,
                query_params=dict(request.query_params),
                user_agent=request.headers.get("user-agent"),
                client_ip=request.client.host if request.client else None,
            )

            return response

        except Exception as exc:
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000

            # Determine status code
            if isinstance(exc, HTTPException):
                status_code = exc.status_code
            elif isinstance(exc, APIError):
                status_code = exc.status_code
            else:
                status_code = 500

            # Log error
            log_api_call(
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration_ms=duration_ms,
                user_id=user_id,
                error=str(exc),
            )

            # Re-raise the exception
            raise


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware to handle and format errors consistently"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Dispatch middleware that forwards the request to the next handler and converts raised exceptions into structured JSON error responses.

        On success, returns the response produced by `call_next`. If an APIError is raised, returns a JSON response using the error's status code and a body containing `error`, `message`, `details`, `timestamp`, and `correlation_id`. If a FastAPI `HTTPException` is raised, returns a JSON response with status code from the exception and a body containing `error: "HTTP_ERROR"`, `message`, `timestamp`, and `correlation_id`. For any other unexpected exception, logs the error (via `log_error`) and returns a 500 JSON response with `error: "INTERNAL_SERVER_ERROR"`, a generic message, `timestamp`, and `correlation_id`.

        Parameters:
            request (Request): The incoming ASGI request.
            call_next (Callable): The next request handler/callable in the middleware chain.

        Returns:
            Response: The downstream handler's response on success, or a JSONResponse with a structured error payload on failure.
        """
        try:
            return await call_next(request)
        except APIError as exc:
            # Handle custom API errors
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                    "timestamp": time.time(),
                    "correlation_id": getattr(request.state, "correlation_id", None),
                },
            )
        except HTTPException as exc:
            # Handle FastAPI HTTP exceptions
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": "HTTP_ERROR",
                    "message": exc.detail,
                    "timestamp": time.time(),
                    "correlation_id": getattr(request.state, "correlation_id", None),
                },
            )
        except Exception as exc:
            # Handle unexpected errors
            correlation_id = getattr(request.state, "correlation_id", None)

            # Log the error
            log_error(
                exc,
                context={
                    "method": request.method,
                    "path": request.url.path,
                    "correlation_id": correlation_id,
                },
            )

            # Return generic error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "timestamp": time.time(),
                    "correlation_id": correlation_id,
                },
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with Redis backend and user-specific quotas"""

    def __init__(self, app):
        """
        Initialize the RateLimitMiddleware and attach it to the given ASGI app.

        Parameters:
            app: The ASGI application instance to wrap with the rate limiting middleware.
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Enforces per-minute rate limits for incoming requests, attaches rate-limit headers to successful responses, and returns a structured 429 response when a limit is exceeded.

        Skips rate limiting for health and documentation endpoints. Determines the client identifier from request.client.host (or "unknown"), uses request.state.user and request.state.api_key if present, and consults the shared rate_limiter for allowance and rate metadata. If the request is not allowed, returns a JSON response with error details, a Retry-After header, and a correlation_id when available. On success, forwards the request and adds X-RateLimit-Limit, X-RateLimit-Remaining, and X-RateLimit-Reset headers to the response. If rate limiting raises an unexpected exception, logs the error and allows the request to proceed (fail-open).

        Parameters:
            request (Request): The incoming ASGI request.
            call_next (Callable): The next request handler to invoke when forwarding the request.

        Returns:
            Response: Either the forwarded handler response with rate-limit headers or a 429 JSONResponse when the limit is exceeded.
        """
        from .rate_limiting import rate_limiter, RateLimitType
        from .api_keys import APIKey

        # Skip rate limiting for health checks and docs
        if request.url.path in [
            "/health",
            "/health/",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]:
            return await call_next(request)

        # Get client identifier
        client_ip = request.client.host if request.client else "unknown"

        # Get user and API key from request state (set by auth middleware)
        user = getattr(request.state, "user", None)
        api_key = getattr(request.state, "api_key", None)

        # Extract endpoint for specific rate limiting
        endpoint = request.url.path

        try:
            # Check rate limits (per-minute for immediate feedback)
            allowed, rate_info = await rate_limiter.check_rate_limit(
                identifier=client_ip,
                limit_type=RateLimitType.PER_MINUTE,
                endpoint=endpoint,
                user=user,
                api_key=api_key,
            )

            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Maximum {rate_info['limit']} requests per minute.",
                        "timestamp": time.time(),
                        "correlation_id": getattr(
                            request.state, "correlation_id", None
                        ),
                        "retry_after": rate_info.get("retry_after", 60),
                    },
                    headers={"Retry-After": str(rate_info.get("retry_after", 60))},
                )

            # Process request
            response = await call_next(request)

            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
            response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
            response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])

            return response

        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Fail open - allow request if rate limiting fails
            return await call_next(request)
