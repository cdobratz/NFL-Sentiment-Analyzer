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
        """Check MongoDB connection and status"""
        try:
            start_time = time.time()

            # Test connection
            db = db_manager.get_database()
            await db.command("ping")

            # Get server status
            server_status = await db.command("serverStatus")

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
        """Check Redis connection and status"""
        try:
            start_time = time.time()

            # Test connection
            redis_client = db_manager.get_redis()
            await redis_client.ping()

            # Get info
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
        """Check external API connectivity"""
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
        """Check ML services status"""
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
        """Get system resource metrics"""
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
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": settings.app_version,
        "service": settings.app_name,
    }


@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with all system components"""
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
    """Kubernetes readiness probe endpoint"""
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
    """Kubernetes liveness probe endpoint"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": time.time() - psutil.boot_time(),
    }


@router.get("/metrics")
async def metrics_endpoint():
    """Prometheus-style metrics endpoint"""
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
    Get current rate limit status for the authenticated user or IP.

    Shows remaining requests for different time windows and when limits reset.
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
