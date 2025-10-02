from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import asyncio

from .core.config import settings
from .core.database import db_manager
from .core.logging import setup_logging, get_correlation_id
from .core.middleware import (
    CorrelationIdMiddleware,
    AuthContextMiddleware,
    RequestLoggingMiddleware,
    ErrorHandlingMiddleware,
    RateLimitMiddleware,
)
from .core.monitoring import performance_monitor, metrics_collector
from .core.openapi import setup_openapi_docs
from .api import (
    auth,
    sentiment,
    data,
    admin,
    websocket,
    mlops,
    analytics,
    health,
    api_keys,
    docs,
)
from .services.data_processing_pipeline import data_processing_pipeline
from .services.scheduled_tasks_service import scheduled_tasks_service

# Configure structured logging
logger = setup_logging()


async def start_monitoring():
    """Start performance monitoring background task"""
    while True:
        try:
            await performance_monitor.run_checks()
            await asyncio.sleep(60)  # Run checks every minute
        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")
            await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info(
        "Starting NFL Sentiment Analyzer...",
        extra={"correlation_id": get_correlation_id()},
    )
    await db_manager.connect_mongodb()
    await db_manager.connect_redis()

    # Start data processing pipeline
    await data_processing_pipeline.start()
    logger.info("Data processing pipeline started")

    # Start scheduled tasks service
    await scheduled_tasks_service.start()
    logger.info("Scheduled tasks service started")

    # Start performance monitoring
    monitoring_task = asyncio.create_task(start_monitoring())
    logger.info("Performance monitoring started")

    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down NFL Sentiment Analyzer...")

    # Cancel monitoring task
    monitoring_task.cancel()
    try:
        await monitoring_task
    except asyncio.CancelledError:
        pass
    logger.info("Performance monitoring stopped")

    # Stop data processing pipeline
    await data_processing_pipeline.stop()
    logger.info("Data processing pipeline stopped")

    # Stop scheduled tasks service
    await scheduled_tasks_service.stop()
    logger.info("Scheduled tasks service stopped")

    await db_manager.disconnect()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Advanced NFL Sentiment Analysis Platform with Real-time Data Processing",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Setup comprehensive OpenAPI documentation
setup_openapi_docs(app)


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

    # Only add HSTS in production
    if not settings.debug:
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )

    return response


# Add custom middleware (order matters - first added is outermost)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(AuthContextMiddleware)
app.add_middleware(CorrelationIdMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(health.router)  # Health checks first
app.include_router(docs.router)  # API documentation
app.include_router(auth.router)
app.include_router(sentiment.router)
app.include_router(data.router)
app.include_router(admin.router)
app.include_router(api_keys.router)  # API key management
app.include_router(websocket.router)
app.include_router(mlops.router)
app.include_router(analytics.router)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NFL Sentiment Analysis API",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs",
        "redoc": "/redoc",
    }


# Legacy health endpoint (redirect to new health router)
@app.get("/health")
async def legacy_health_check():
    """Legacy health check endpoint - redirects to /health/"""
    from datetime import datetime

    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": settings.app_version,
        "note": "Use /health/ for basic checks or /health/detailed for comprehensive status",
    }
