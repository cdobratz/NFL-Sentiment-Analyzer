"""
Common response models for the API.
"""

from pydantic import BaseModel
from typing import Optional, Dict, Any, List
from datetime import datetime


class ErrorResponse(BaseModel):
    """Standard error response model"""

    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: float
    correlation_id: Optional[str] = None


class SuccessResponse(BaseModel):
    """Standard success response model"""

    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: float
    correlation_id: Optional[str] = None


class PaginatedResponse(BaseModel):
    """Paginated response model"""

    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


class RateLimitInfo(BaseModel):
    """Rate limit information"""

    limit: int
    remaining: int
    reset: int
    retry_after: Optional[int] = None


class HealthStatus(BaseModel):
    """Health check status"""

    status: str  # healthy, degraded, unhealthy
    timestamp: str
    version: str
    check_duration_ms: Optional[float] = None


class ComponentHealth(BaseModel):
    """Individual component health status"""

    status: str
    response_time_ms: Optional[float] = None
    error: Optional[str] = None
    version: Optional[str] = None
    uptime_seconds: Optional[int] = None


class SystemMetrics(BaseModel):
    """System resource metrics"""

    cpu: Dict[str, Any]
    memory: Dict[str, Any]
    disk: Dict[str, Any]


class APIKeyInfo(BaseModel):
    """API key information (without the actual key)"""

    id: str
    name: str
    scopes: List[str]
    status: str
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int
    rate_limit: int
    created_by: str
    metadata: Dict[str, Any] = {}


class ValidationErrorDetail(BaseModel):
    """Validation error detail"""

    field: str
    message: str
    invalid_value: Optional[Any] = None


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field details"""

    validation_errors: List[ValidationErrorDetail] = []
