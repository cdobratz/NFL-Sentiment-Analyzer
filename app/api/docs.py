"""
API documentation and information endpoints.
"""

from fastapi import APIRouter, Depends, Request
from typing import Dict, Any, List
from datetime import datetime

from ..core.config import settings
from ..core.dependencies import get_optional_user
from ..core.api_keys import APIKeyScope

router = APIRouter(prefix="/api", tags=["Documentation"])


@router.get("/info")
async def api_info():
    """
    Get comprehensive API information including version, features, and capabilities.
    
    This endpoint provides an overview of the NFL Sentiment Analyzer API,
    including available features, authentication methods, and rate limits.
    """
    return {
        "name": "NFL Sentiment Analyzer API",
        "version": settings.app_version,
        "description": "Advanced NFL Sentiment Analysis Platform with Real-time Data Processing",
        "features": [
            "Real-time sentiment analysis with NFL-specific context",
            "Multi-source data integration (Twitter/X, ESPN, betting APIs)",
            "MLOps pipeline with automated model training and deployment",
            "Real-time WebSocket updates",
            "Comprehensive analytics and historical trends",
            "Admin panel with user and system management"
        ],
        "authentication": {
            "methods": [
                {
                    "type": "JWT Bearer Token",
                    "description": "User-based authentication for web applications",
                    "header": "Authorization: Bearer <token>",
                    "obtain_endpoint": "/auth/login"
                },
                {
                    "type": "API Key",
                    "description": "Programmatic access for third-party integrations",
                    "header": "X-API-Key: <api-key>",
                    "obtain_method": "Contact administrator"
                }
            ]
        },
        "rate_limits": {
            "default": {
                "per_minute": settings.rate_limit_requests,
                "per_hour": settings.rate_limit_requests * 60,
                "per_day": settings.rate_limit_requests * 60 * 24
            },
            "user_roles": {
                "user": {
                    "per_minute": 100,
                    "per_hour": 5000,
                    "per_day": 50000
                },
                "admin": {
                    "per_minute": 1000,
                    "per_hour": 50000,
                    "per_day": 500000
                }
            },
            "api_keys": "Custom limits based on key configuration"
        },
        "endpoints": {
            "documentation": "/docs (Swagger UI), /redoc (ReDoc)",
            "health_checks": "/health/",
            "api_info": "/api/info",
            "rate_limits": "/health/rate-limits"
        },
        "support": {
            "documentation": "https://docs.nflsentiment.com",
            "email": "api-support@nflsentiment.com",
            "status_page": "https://status.nflsentiment.com"
        },
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.get("/endpoints")
async def list_endpoints():
    """
    List all available API endpoints with their authentication requirements and rate limits.
    
    Provides a comprehensive overview of all endpoints, their HTTP methods,
    required authentication, and any special rate limiting rules.
    """
    endpoints = [
        {
            "group": "Authentication",
            "endpoints": [
                {
                    "path": "/auth/register",
                    "method": "POST",
                    "description": "Register a new user account",
                    "authentication": "None",
                    "rate_limit": "Standard"
                },
                {
                    "path": "/auth/login",
                    "method": "POST", 
                    "description": "Login and obtain JWT token",
                    "authentication": "None",
                    "rate_limit": "Standard"
                },
                {
                    "path": "/auth/refresh",
                    "method": "POST",
                    "description": "Refresh JWT token",
                    "authentication": "JWT Token",
                    "rate_limit": "Standard"
                },
                {
                    "path": "/auth/logout",
                    "method": "DELETE",
                    "description": "Logout and invalidate token",
                    "authentication": "JWT Token",
                    "rate_limit": "Standard"
                }
            ]
        },
        {
            "group": "Sentiment Analysis",
            "endpoints": [
                {
                    "path": "/sentiment/analyze",
                    "method": "POST",
                    "description": "Analyze sentiment of text with NFL context",
                    "authentication": "JWT Token or API Key (write:sentiment)",
                    "rate_limit": "Standard"
                },
                {
                    "path": "/sentiment/batch",
                    "method": "POST",
                    "description": "Batch sentiment analysis",
                    "authentication": "JWT Token or API Key (write:sentiment)",
                    "rate_limit": "Higher limits apply"
                },
                {
                    "path": "/sentiment/team/{team_id}",
                    "method": "GET",
                    "description": "Get team-specific sentiment data",
                    "authentication": "JWT Token or API Key (read:sentiment)",
                    "rate_limit": "Standard"
                },
                {
                    "path": "/sentiment/player/{player_id}",
                    "method": "GET",
                    "description": "Get player-specific sentiment data",
                    "authentication": "JWT Token or API Key (read:sentiment)",
                    "rate_limit": "Standard"
                }
            ]
        },
        {
            "group": "NFL Data",
            "endpoints": [
                {
                    "path": "/data/teams",
                    "method": "GET",
                    "description": "Get NFL teams data",
                    "authentication": "JWT Token or API Key (read:data)",
                    "rate_limit": "Standard"
                },
                {
                    "path": "/data/players",
                    "method": "GET",
                    "description": "Get NFL players data",
                    "authentication": "JWT Token or API Key (read:data)",
                    "rate_limit": "Standard"
                },
                {
                    "path": "/data/games",
                    "method": "GET",
                    "description": "Get NFL games and schedule",
                    "authentication": "JWT Token or API Key (read:data)",
                    "rate_limit": "Standard"
                },
                {
                    "path": "/data/betting-lines",
                    "method": "GET",
                    "description": "Get current betting lines",
                    "authentication": "JWT Token or API Key (read:data)",
                    "rate_limit": "Standard"
                }
            ]
        },
        {
            "group": "Analytics",
            "endpoints": [
                {
                    "path": "/analytics/trends",
                    "method": "GET",
                    "description": "Get sentiment trends and analytics",
                    "authentication": "JWT Token or API Key (read:analytics)",
                    "rate_limit": "Standard"
                },
                {
                    "path": "/analytics/insights",
                    "method": "GET",
                    "description": "Get sentiment insights and patterns",
                    "authentication": "JWT Token or API Key (read:analytics)",
                    "rate_limit": "Standard"
                }
            ]
        },
        {
            "group": "Admin",
            "endpoints": [
                {
                    "path": "/admin/users",
                    "method": "GET",
                    "description": "List and manage users",
                    "authentication": "JWT Token (Admin role)",
                    "rate_limit": "Admin limits"
                },
                {
                    "path": "/admin/api-keys",
                    "method": "GET, POST, DELETE",
                    "description": "Manage API keys",
                    "authentication": "JWT Token (Admin role)",
                    "rate_limit": "Admin limits"
                },
                {
                    "path": "/admin/system-health",
                    "method": "GET",
                    "description": "System health and monitoring",
                    "authentication": "JWT Token (Admin role)",
                    "rate_limit": "Admin limits"
                }
            ]
        },
        {
            "group": "Health & Monitoring",
            "endpoints": [
                {
                    "path": "/health/",
                    "method": "GET",
                    "description": "Basic health check",
                    "authentication": "None",
                    "rate_limit": "No limit"
                },
                {
                    "path": "/health/detailed",
                    "method": "GET",
                    "description": "Detailed system health",
                    "authentication": "None",
                    "rate_limit": "No limit"
                },
                {
                    "path": "/health/ready",
                    "method": "GET",
                    "description": "Kubernetes readiness probe",
                    "authentication": "None",
                    "rate_limit": "No limit"
                },
                {
                    "path": "/health/live",
                    "method": "GET",
                    "description": "Kubernetes liveness probe",
                    "authentication": "None",
                    "rate_limit": "No limit"
                },
                {
                    "path": "/health/metrics",
                    "method": "GET",
                    "description": "Prometheus metrics",
                    "authentication": "None",
                    "rate_limit": "No limit"
                }
            ]
        }
    ]
    
    return {
        "endpoints": endpoints,
        "total_endpoints": sum(len(group["endpoints"]) for group in endpoints),
        "authentication_note": "API keys require specific scopes. Contact admin for API key access.",
        "rate_limit_note": "Rate limits vary by authentication method and user role.",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.get("/scopes")
async def list_api_scopes():
    """
    List all available API key scopes and their permissions.
    
    API keys can be granted specific scopes that determine which endpoints
    they can access. This endpoint lists all available scopes.
    """
    scopes = [
        {
            "scope": APIKeyScope.READ_SENTIMENT.value,
            "description": "Read sentiment analysis data and results",
            "endpoints": [
                "GET /sentiment/team/{team_id}",
                "GET /sentiment/player/{player_id}",
                "GET /sentiment/game/{game_id}",
                "GET /sentiment/trends"
            ]
        },
        {
            "scope": APIKeyScope.WRITE_SENTIMENT.value,
            "description": "Submit text for sentiment analysis",
            "endpoints": [
                "POST /sentiment/analyze",
                "POST /sentiment/batch"
            ]
        },
        {
            "scope": APIKeyScope.READ_DATA.value,
            "description": "Read NFL data (teams, players, games, betting lines)",
            "endpoints": [
                "GET /data/teams",
                "GET /data/players", 
                "GET /data/games",
                "GET /data/betting-lines"
            ]
        },
        {
            "scope": APIKeyScope.WRITE_DATA.value,
            "description": "Submit data updates (limited use cases)",
            "endpoints": [
                "POST /data/teams",
                "POST /data/players"
            ]
        },
        {
            "scope": APIKeyScope.READ_ANALYTICS.value,
            "description": "Read analytics, trends, and insights",
            "endpoints": [
                "GET /analytics/trends",
                "GET /analytics/insights",
                "GET /analytics/reports"
            ]
        },
        {
            "scope": APIKeyScope.ADMIN.value,
            "description": "Full administrative access to all endpoints",
            "endpoints": [
                "All endpoints (equivalent to admin user role)"
            ]
        }
    ]
    
    return {
        "scopes": scopes,
        "total_scopes": len(scopes),
        "note": "API keys can have multiple scopes. Admin scope grants access to all endpoints.",
        "contact": "Contact administrator to obtain API keys with required scopes.",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }