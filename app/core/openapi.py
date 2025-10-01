"""
OpenAPI documentation configuration and customization.
"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any

from .config import settings


def custom_openapi(app: FastAPI) -> Dict[str, Any]:
    """Generate custom OpenAPI schema with comprehensive documentation"""
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="NFL Sentiment Analyzer API",
        version=settings.app_version,
        description="""
# NFL Sentiment Analyzer API

A comprehensive real-time NFL sentiment analysis platform that provides advanced sentiment analysis, 
data ingestion, and analytics capabilities for NFL-related content.

## Features

- **Real-time Sentiment Analysis**: Advanced ML-powered sentiment analysis with NFL-specific context
- **Multi-source Data Integration**: Collect data from Twitter/X, ESPN, and betting APIs
- **MLOps Pipeline**: Automated model training, deployment, and monitoring
- **Real-time Updates**: WebSocket connections for live data streaming
- **Comprehensive Analytics**: Historical trends and aggregated insights
- **Admin Panel**: User management and system monitoring

## Authentication

This API supports two authentication methods:

### 1. JWT Bearer Token Authentication
For user-based access, obtain a JWT token by logging in and include it in the Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

### 2. API Key Authentication
For programmatic access, use an API key in the X-API-Key header:
```
X-API-Key: <your-api-key>
```

API keys have specific scopes that determine which endpoints they can access.

## Rate Limiting

All endpoints are subject to rate limiting:
- **Default**: 100 requests per minute per IP/user
- **API Keys**: Custom rate limits based on key configuration

Rate limit headers are included in responses:
- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when rate limit resets

## Error Handling

All errors follow a consistent format:
```json
{
    "error": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {},
    "timestamp": 1234567890,
    "correlation_id": "uuid-for-tracing"
}
```

## Monitoring and Observability

- **Health Checks**: `/health/` endpoints for monitoring
- **Metrics**: Prometheus-compatible metrics at `/health/metrics`
- **Correlation IDs**: All requests include correlation IDs for tracing
- **Structured Logging**: JSON-formatted logs with correlation IDs

## WebSocket Connections

Real-time data is available via WebSocket connections:
- **Endpoint**: `/ws/sentiment`
- **Authentication**: Include JWT token as query parameter: `?token=<jwt-token>`
- **Data Format**: JSON messages with sentiment updates

## Data Models

The API uses consistent data models across all endpoints. Key models include:
- **SentimentAnalysis**: Individual sentiment analysis results
- **Team**: NFL team information and sentiment aggregates
- **Player**: Player information and sentiment data
- **Game**: Game information with sentiment context

## Support

For API support and questions:
- **Documentation**: This interactive documentation
- **Health Status**: Check `/health/detailed` for system status
- **Correlation IDs**: Include correlation ID from response headers when reporting issues
        """,
        routes=app.routes,
        servers=[
            {
                "url": "https://api.nflsentiment.com",
                "description": "Production server"
            },
            {
                "url": "https://staging-api.nflsentiment.com", 
                "description": "Staging server"
            },
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            }
        ]
    )
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT token obtained from /auth/login endpoint"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for programmatic access. Contact admin to obtain an API key."
        }
    }
    
    # Add global security requirement (can be overridden per endpoint)
    openapi_schema["security"] = [
        {"BearerAuth": []},
        {"ApiKeyAuth": []}
    ]
    
    # Add custom tags with descriptions
    openapi_schema["tags"] = [
        {
            "name": "Health",
            "description": "Health check and monitoring endpoints"
        },
        {
            "name": "Authentication", 
            "description": "User authentication and session management"
        },
        {
            "name": "Sentiment Analysis",
            "description": "NFL-specific sentiment analysis endpoints"
        },
        {
            "name": "Data",
            "description": "NFL data endpoints (teams, players, games, betting lines)"
        },
        {
            "name": "Analytics",
            "description": "Historical analytics and trend analysis"
        },
        {
            "name": "Admin",
            "description": "Administrative endpoints (requires admin role)"
        },
        {
            "name": "MLOps",
            "description": "Machine learning operations and model management"
        },
        {
            "name": "WebSocket",
            "description": "Real-time data streaming via WebSocket connections"
        }
    ]
    
    # Add contact information
    openapi_schema["info"]["contact"] = {
        "name": "NFL Sentiment Analyzer API Support",
        "email": "api-support@nflsentiment.com",
        "url": "https://nflsentiment.com/support"
    }
    
    # Add license information
    openapi_schema["info"]["license"] = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Full Documentation and Guides",
        "url": "https://docs.nflsentiment.com"
    }
    
    # Add custom extensions
    openapi_schema["info"]["x-logo"] = {
        "url": "https://nflsentiment.com/logo.png",
        "altText": "NFL Sentiment Analyzer"
    }
    
    # Add response examples for common error codes
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    
    openapi_schema["components"]["examples"] = {
        "ValidationError": {
            "summary": "Validation Error Example",
            "value": {
                "error": "VALIDATION_ERROR",
                "message": "Invalid input data",
                "details": {
                    "field": "team_id",
                    "issue": "Team ID must be a valid NFL team identifier"
                },
                "timestamp": 1234567890,
                "correlation_id": "abc123-def456-ghi789"
            }
        },
        "AuthenticationError": {
            "summary": "Authentication Error Example", 
            "value": {
                "error": "AUTHENTICATION_ERROR",
                "message": "Authentication required",
                "timestamp": 1234567890,
                "correlation_id": "abc123-def456-ghi789"
            }
        },
        "RateLimitError": {
            "summary": "Rate Limit Error Example",
            "value": {
                "error": "RATE_LIMIT_EXCEEDED", 
                "message": "Rate limit exceeded. Maximum 100 requests per minute.",
                "timestamp": 1234567890,
                "correlation_id": "abc123-def456-ghi789"
            }
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


def setup_openapi_docs(app: FastAPI):
    """Setup OpenAPI documentation for the FastAPI app"""
    
    # Custom OpenAPI schema
    app.openapi = lambda: custom_openapi(app)
    
    # Update app metadata
    app.title = "NFL Sentiment Analyzer API"
    app.description = "Advanced NFL Sentiment Analysis Platform with Real-time Data Processing"
    app.version = settings.app_version
    app.terms_of_service = "https://nflsentiment.com/terms"
    app.contact = {
        "name": "API Support",
        "url": "https://nflsentiment.com/support",
        "email": "api-support@nflsentiment.com"
    }
    app.license_info = {
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT"
    }