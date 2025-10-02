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

from .logging import set_correlation_id, log_api_call, log_error
from .exceptions import APIError

logger = logging.getLogger(__name__)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Middleware to handle correlation IDs for request tracing"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get or generate correlation ID
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
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
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
