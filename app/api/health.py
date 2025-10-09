"""
Health check endpoints for monitoring and observability.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
import time
import psutil
import logging
from pymongo.errors import OperationFailure

from ..core.database import db_manager
from ..core.config import settings
from ..core.exceptions import ServiceUnavailableError
from ..core.dependencies import get_optional_user
from ..services.data_ingestion_service import data_ingestion_service
from ..services.sentiment_service import sentiment_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=["Health"])


class HealthChecker:
    """Health check service for monitoring system components"""

    async def check_mongodb(self) -> Dict[str, Any]:
        """
        Evaluate MongoDB connectivity and basic server health.

        Performs a connectivity check and retrieves basic server status metrics.

        Returns:
            dict: A dictionary describing MongoDB health. Keys:
                - "status": `"healthy"` when checks succeed, `"unhealthy"` otherwise.
                - "response_time_ms": Round-trip latency in milliseconds (float) when healthy.
                - "version": Server version string or `None` if unavailable.
                - "uptime_seconds": Server uptime in seconds or `None` if unavailable.
                - "connections": Current connection count (int, 0 if not reported).
                - "error": Error message string present when `"status"` is `"unhealthy"`.
        """
        try:
            start_time = time.time()

            # Test connection
            db = db_manager.get_database()
            await db.command("ping")

            # Get server status (optional - may fail due to privileges)
            server_status = {}
            try:
                server_status = await db.command("serverStatus")
            except OperationFailure as op_error:
                logger.warning(f"MongoDB serverStatus command failed (insufficient privileges): {op_error}")
                # Continue with empty server_status - ping is the primary connectivity check

            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "version": server_status.get("version"),
                "uptime_seconds": server_status.get("uptime"),
                "connections": server_status.get("connections", {}).get("current", 0),
            }
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def check_redis(self) -> Dict[str, Any]:
        """
        Check Redis connectivity and return a structured health payload.

        Returns:
            dict: Health information for Redis. On success contains:
                - `status`: "healthy"
                - `response_time_ms`: Response time in milliseconds (rounded to 2 decimals)
                - `version`: Redis server version string or `None`
                - `uptime_seconds`: Server uptime in seconds or `None`
                - `connected_clients`: Number of connected clients (int)
                - `used_memory`: Human-readable memory usage string or `None`
            On failure contains:
                - `status`: "unhealthy"
                - `error`: Error message string describing the failure
        """
        try:
            start_time = time.time()

            # Test connection
            redis_client = db_manager.get_redis()
            
            # Check if redis_client is available
            if not redis_client:
                return {"status": "unhealthy", "error": "Redis unavailable"}

            # Run blocking Redis methods in thread pool
            await asyncio.to_thread(redis_client.ping)
            await redis_client.ping()
            info = await redis_client.info()

            response_time = (time.time() - start_time) * 1000

            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "version": info.get("redis_version"),
                "uptime_seconds": info.get("uptime_in_seconds"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human"),
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def check_external_apis(self) -> Dict[str, Any]:
        """
        Check connectivity to external APIs used by the application.

        Returns:
            results (Dict[str, Any]): Mapping of external service identifiers to their health details.
                Each value contains a `status` field with one of: `"healthy"`, `"stopped"`, or `"unhealthy"`.
                When a service is `"unhealthy"`, an `error` field will contain the error message.
        """
        results = {}

        # Check data ingestion service
        try:
            # This would test actual API connectivity
            results["data_ingestion"] = {
                "status": "healthy" if data_ingestion_service.is_running else "stopped"
            }
        except Exception as e:
            results["data_ingestion"] = {"status": "unhealthy", "error": str(e)}

        return results

    async def check_ml_services(self) -> Dict[str, Any]:
        """
        Verifies availability of ML services used by the application.

        Performs a live test of the sentiment analysis service and reports its health and test outcome.

        Returns:
            result (Dict[str, Any]): A mapping with key "sentiment_service" containing:
                - "status": "healthy" or "unhealthy".
                - If healthy, "test_result" (bool) indicates whether the test call produced a truthy result.
                - If unhealthy, "error" (str) contains the exception message.
        """
        try:
            # Test sentiment analysis
            test_result = await sentiment_service.analyze_text(
                "Test sentiment analysis"
            )

            return {
                "sentiment_service": {
                    "status": "healthy",
                    "test_result": bool(test_result),
                }
            }
        except Exception as e:
            logger.error(f"ML services health check failed: {e}")
            return {"sentiment_service": {"status": "unhealthy", "error": str(e)}}

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Collect current system resource metrics for CPU, memory, and the root filesystem disk.

        Returns:
            dict: A mapping with keys:
                - "cpu": dict with "usage_percent" (float) and "count" (int).
                - "memory": dict with "total" (int bytes), "available" (int bytes), "percent" (float), and "used" (int bytes).
                - "disk": dict with "total" (int bytes), "used" (int bytes), "free" (int bytes), and "percent" (float).
                - On failure, returns {"error": "<error message>"} describing the collection error.
        """
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()

            # Disk usage
            disk = psutil.disk_usage("/")

            return {
                "cpu": {"usage_percent": cpu_percent, "count": psutil.cpu_count()},
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100,
                },
            }
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
            return {"error": str(e)}


health_checker = HealthChecker()


@router.get("/")
async def basic_health_check():
    """
    Return a minimal health status for the service.

    Returns:
        dict: Health payload containing:
            - status (str): "healthy" or other overall status string.
            - timestamp (str): ISO 8601 UTC timestamp (ends with "Z").
            - version (str): Application version from settings.
            - service (str): Application/service name from settings.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": settings.app_version,
        "service": settings.app_name,
    }


@router.get("/detailed")
async def detailed_health_check():
    """
    Perform a comprehensive health check across database, cache, external APIs, ML services, and system metrics.

    Runs MongoDB, Redis, external API, and ML-service checks concurrently, collects system metrics, and aggregates results into a structured payload describing each component and overall status. If any component check raises an exception or indicates an unhealthy state, the overall status is set accordingly.

    Returns:
        dict: Health report with keys:
            - status: overall health string, one of "healthy", "degraded", or "unhealthy".
            - timestamp: ISO 8601 UTC timestamp string.
            - version: application version string from settings.
            - check_duration_ms: total duration of checks in milliseconds (rounded).
            - components: mapping with keys "mongodb", "redis", "external_apis", and "ml_services" containing each component's result or an error object.
            - system_metrics: CPU, memory, and disk usage metrics collected from the host.

    Raises:
        HTTPException: with status code 503 and the health payload when overall status is "unhealthy".
    """
    start_time = time.time()

    # Run all health checks concurrently
    mongodb_check, redis_check, external_apis_check, ml_services_check = (
        await asyncio.gather(
            health_checker.check_mongodb(),
            health_checker.check_redis(),
            health_checker.check_external_apis(),
            health_checker.check_ml_services(),
            return_exceptions=True,
        )
    )

    # Get system metrics
    system_metrics = health_checker.get_system_metrics()

    total_time = (time.time() - start_time) * 1000

    # Determine overall status
    components = [mongodb_check, redis_check, external_apis_check, ml_services_check]
    overall_status = "healthy"

    for component in components:
        if isinstance(component, Exception):
            overall_status = "unhealthy"
            break
        elif isinstance(component, dict):
            if component.get("status") == "unhealthy":
                overall_status = "degraded"
            elif any(
                service.get("status") == "unhealthy"
                for service in component.values()
                if isinstance(service, dict)
            ):
                overall_status = "degraded"

    result = {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": settings.app_version,
        "check_duration_ms": round(total_time, 2),
        "components": {
            "mongodb": (
                mongodb_check
                if not isinstance(mongodb_check, Exception)
                else {"status": "error", "error": str(mongodb_check)}
            ),
            "redis": (
                redis_check
                if not isinstance(redis_check, Exception)
                else {"status": "error", "error": str(redis_check)}
            ),
            "external_apis": (
                external_apis_check
                if not isinstance(external_apis_check, Exception)
                else {"status": "error", "error": str(external_apis_check)}
            ),
            "ml_services": (
                ml_services_check
                if not isinstance(ml_services_check, Exception)
                else {"status": "error", "error": str(ml_services_check)}
            ),
        },
        "system_metrics": system_metrics,
    }

    # Return appropriate status code
    if overall_status == "unhealthy":
        raise HTTPException(status_code=503, detail=result)

    return result


@router.get("/ready")
async def readiness_check():
    """
    Perform a Kubernetes readiness probe by verifying critical dependencies.

    Checks MongoDB and Redis health; if both are healthy returns a payload with status "ready" and an ISO8601 UTC timestamp. If any check fails, raises HTTPException with status code 503 and a detail object containing status "not_ready", an error message, and a timestamp.

    Returns:
        dict: {"status": "ready", "timestamp": "<ISO8601 UTC time>Z"}
    """
    try:
        # Check critical dependencies
        mongodb_status = await health_checker.check_mongodb()
        redis_status = await health_checker.check_redis()

        if (
            mongodb_status.get("status") != "healthy"
            or redis_status.get("status") != "healthy"
        ):
            raise ServiceUnavailableError("Critical dependencies not ready")

        return {"status": "ready", "timestamp": datetime.utcnow().isoformat() + "Z"}
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        )


@router.get("/live")
async def liveness_check():
    """
    Return liveness information for the service (Kubernetes liveness probe).

    Returns:
        health (dict): Dictionary with keys:
            - status (str): "alive".
            - timestamp (str): UTC ISO 8601 timestamp ending with "Z".
            - uptime_seconds (float): Seconds elapsed since system boot.
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": time.time() - psutil.boot_time(),
    }


@router.get("/metrics")
async def metrics_endpoint():
    """
    Expose system and application metrics in Prometheus text format.

    Collects CPU, memory, and disk metrics from the health checker and formats them as Prometheus-compatible gauge lines, plus an application info metric with the current version.

    Returns:
        prometheus_text (str): Prometheus-formatted metrics payload as a single string ending with a newline.
    """
    system_metrics = health_checker.get_system_metrics()

    # Convert to Prometheus format
    metrics = []

    if "cpu" in system_metrics:
        metrics.append(
            f"nfl_analyzer_cpu_usage_percent {system_metrics['cpu']['usage_percent']}"
        )
        metrics.append(f"nfl_analyzer_cpu_count {system_metrics['cpu']['count']}")

    if "memory" in system_metrics:
        metrics.append(
            f"nfl_analyzer_memory_total_bytes {system_metrics['memory']['total']}"
        )
        metrics.append(
            f"nfl_analyzer_memory_used_bytes {system_metrics['memory']['used']}"
        )
        metrics.append(
            f"nfl_analyzer_memory_usage_percent {system_metrics['memory']['percent']}"
        )

    if "disk" in system_metrics:
        metrics.append(
            f"nfl_analyzer_disk_total_bytes {system_metrics['disk']['total']}"
        )
        metrics.append(f"nfl_analyzer_disk_used_bytes {system_metrics['disk']['used']}")
        metrics.append(
            f"nfl_analyzer_disk_usage_percent {system_metrics['disk']['percent']}"
        )

    # Add application metrics
    metrics.append(f'nfl_analyzer_info{{version="{settings.app_version}"}} 1')

    return "\n".join(metrics) + "\n"


@router.get("/rate-limits")
async def get_rate_limit_status(
    request: Request, current_user: Optional[dict] = Depends(get_optional_user)
):
    """
    Return the current rate limit status for the requesting client (by API key, authenticated user, or IP).

    Provides the rate limit details computed for the client, an ISO 8601 UTC timestamp, and the inferred client type.

    Returns:
        dict: A payload containing:
            - rate_limits: Mapping of rate-limit windows to status/usage details for the client.
            - timestamp: ISO 8601 UTC timestamp string when the status was generated (e.g., "2025-10-02T12:34:56.789Z").
            - client_type: One of `"api_key"`, `"user"`, or `"anonymous"` indicating how the client was identified.
    """
    from ..core.rate_limiting import rate_limiter
    from ..core.api_keys import APIKey

    # Get client identifier
    client_ip = request.client.host if request.client else "unknown"

    # Get auth context
    user = getattr(request.state, "user", current_user)
    api_key = getattr(request.state, "api_key", None)

    # Get rate limit status
    status = await rate_limiter.get_rate_limit_status(
        identifier=client_ip, user=user, api_key=api_key
    )

    return {
        "rate_limits": status,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "client_type": "api_key" if api_key else ("user" if user else "anonymous"),
    }
