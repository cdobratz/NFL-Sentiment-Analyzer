"""
Tests for API documentation accuracy, completeness, rate limiting, 
authentication mechanisms, load testing, and monitoring system validation.

This test suite covers:
- API documentation accuracy and completeness
- Rate limiting mechanisms
- Authentication system validation
- Load testing for API performance
- Monitoring and alerting system validation

Requirements: 6.2, 6.3, 6.4
"""

import os
# Set test environment variables before importing app
os.environ.update({
    "MONGODB_URL": "mongodb://localhost:27017",
    "DATABASE_NAME": "nfl_analyzer_test", 
    "REDIS_URL": "redis://localhost:6379/1",
    "SECRET_KEY": "78183c734c4337b3b9ac71f816dfab85a8a3bebbc4f4dc6ecd5d1b9c0d4307f1",
    "DEBUG": "true",
    "TESTING": "true"
})

import pytest
import asyncio
import time
import json
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import FastAPI
import httpx

from app.main import app
from app.core.config import settings
from app.core.rate_limiting import rate_limiter, RateLimitType
from app.core.monitoring import metrics_collector, alert_manager, performance_monitor
from app.core.api_keys import APIKey
from app.api.health import health_checker


class TestAPIDocumentation:
    """Test API documentation accuracy and completeness"""
    
    def test_openapi_schema_generation(self):
        """Test that OpenAPI schema is generated correctly"""
        client = TestClient(app)
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        schema = response.json()
        
        # Verify basic schema structure
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema
        
        # Verify API info
        assert schema["info"]["title"] == "NFL Sentiment Analyzer API"
        assert "version" in schema["info"]
        assert "description" in schema["info"]
        
        # Verify security schemes are defined
        assert "securitySchemes" in schema["components"]
        assert "BearerAuth" in schema["components"]["securitySchemes"]
        assert "ApiKeyAuth" in schema["components"]["securitySchemes"]
    
    def test_swagger_ui_accessibility(self):
        """Test that Swagger UI is accessible"""
        client = TestClient(app)
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "swagger" in response.text.lower()
    
    def test_redoc_accessibility(self):
        """Test that ReDoc is accessible"""
        client = TestClient(app)
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "redoc" in response.text.lower()
    
    def test_api_endpoints_documented(self):
        """Test that all API endpoints are properly documented"""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()
        
        # Critical endpoints that must be documented
        required_endpoints = [
            "/health/",
            "/auth/login",
            "/auth/register",
            "/sentiment/analyze",
            "/data/teams",
            "/admin/users"
        ]
        
        documented_paths = list(schema["paths"].keys())
        
        for endpoint in required_endpoints:
            assert endpoint in documented_paths, f"Endpoint {endpoint} not documented"
    
    def test_endpoint_documentation_completeness(self):
        """Test that endpoints have complete documentation"""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()
        
        # Check a sample endpoint for completeness
        sentiment_analyze = schema["paths"].get("/sentiment/analyze", {}).get("post", {})
        
        assert "summary" in sentiment_analyze
        assert "description" in sentiment_analyze
        assert "requestBody" in sentiment_analyze
        assert "responses" in sentiment_analyze
        
        # Check response documentation
        responses = sentiment_analyze["responses"]
        assert "200" in responses
        assert "400" in responses
        assert "401" in responses
        assert "429" in responses
    
    def test_security_documentation(self):
        """Test that security requirements are properly documented"""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()
        
        # Check global security
        assert "security" in schema
        
        # Check that protected endpoints have security requirements
        admin_users = schema["paths"].get("/admin/users", {}).get("get", {})
        if "security" not in admin_users:
            # Should inherit from global security
            assert len(schema["security"]) > 0
    
    def test_error_response_examples(self):
        """Test that error response examples are documented"""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()
        
        # Check that error examples are defined
        assert "examples" in schema["components"]
        examples = schema["components"]["examples"]
        
        assert "ValidationError" in examples
        assert "AuthenticationError" in examples
        assert "RateLimitError" in examples


class TestRateLimiting:
    """Test rate limiting mechanisms"""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis for rate limiting tests"""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.incr.return_value = 1
        mock_redis.expire.return_value = True
        mock_redis.ttl.return_value = 60
        mock_redis.pipeline.return_value = mock_redis
        mock_redis.execute.return_value = [1, True]
        return mock_redis
    
    @pytest.mark.asyncio
    async def test_rate_limit_check_within_limits(self, mock_redis):
        """Test rate limiting when within limits"""
        with patch('app.core.database.db_manager.get_redis', return_value=mock_redis):
            allowed, info = await rate_limiter.check_rate_limit(
                identifier="test_ip",
                limit_type=RateLimitType.PER_MINUTE
            )
            
            assert allowed is True
            assert "limit" in info
            assert "remaining" in info
            assert "reset" in info
            assert info["remaining"] > 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_check_exceeded(self, mock_redis):
        """Test rate limiting when limits are exceeded"""
        # Mock Redis to return count at limit
        mock_redis.get.return_value = "100"  # At limit
        
        with patch('app.core.database.db_manager.get_redis', return_value=mock_redis):
            allowed, info = await rate_limiter.check_rate_limit(
                identifier="test_ip",
                limit_type=RateLimitType.PER_MINUTE
            )
            
            assert allowed is False
            assert info["remaining"] == 0
            assert "retry_after" in info
    
    @pytest.mark.asyncio
    async def test_user_based_rate_limits(self, mock_redis):
        """Test that user-based rate limits work correctly"""
        user = {"_id": "user123", "role": "user"}
        
        with patch('app.core.database.db_manager.get_redis', return_value=mock_redis):
            allowed, info = await rate_limiter.check_rate_limit(
                identifier="test_ip",
                user=user,
                limit_type=RateLimitType.PER_MINUTE
            )
            
            assert allowed is True
            # User limits should be higher than default
            assert info["limit"] >= 100
    
    @pytest.mark.asyncio
    async def test_api_key_rate_limits(self, mock_redis):
        """Test that API key rate limits work correctly"""
        api_key = APIKey(
            id="key123",
            name="Test Key",
            key_hash="hash123",
            rate_limit=1000,  # 1000 requests per hour
            scopes=["read"],
            is_active=True
        )
        
        with patch('app.core.database.db_manager.get_redis', return_value=mock_redis):
            allowed, info = await rate_limiter.check_rate_limit(
                identifier="test_ip",
                api_key=api_key,
                limit_type=RateLimitType.PER_HOUR
            )
            
            assert allowed is True
            assert info["limit"] == 1000
    
    @pytest.mark.asyncio
    async def test_multiple_rate_limit_types(self, mock_redis):
        """Test checking multiple rate limit types simultaneously"""
        with patch('app.core.database.db_manager.get_redis', return_value=mock_redis):
            allowed, results = await rate_limiter.check_multiple_limits(
                identifier="test_ip"
            )
            
            assert allowed is True
            assert "per_minute" in results
            assert "per_hour" in results
            assert "per_day" in results
    
    def test_rate_limit_headers_in_response(self):
        """Test that rate limit headers are included in API responses"""
        client = TestClient(app)
        
        # Make a request to any endpoint
        response = client.get("/health/")
        
        # Check for rate limit headers
        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
    
    def test_rate_limit_status_endpoint(self):
        """Test the rate limit status endpoint"""
        client = TestClient(app)
        response = client.get("/health/rate-limits")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "rate_limits" in data
        assert "timestamp" in data
        assert "client_type" in data


class TestAuthentication:
    """Test authentication mechanisms"""
    
    def test_jwt_authentication_required(self):
        """Test that protected endpoints require JWT authentication"""
        client = TestClient(app)
        
        # Try to access protected endpoint without auth
        response = client.get("/admin/users")
        assert response.status_code == 401
        
        # Check error response format
        error = response.json()
        assert "error" in error
        assert "message" in error
    
    def test_api_key_authentication(self):
        """Test API key authentication"""
        client = TestClient(app)
        
        # Try with invalid API key
        headers = {"X-API-Key": "invalid-key"}
        response = client.get("/data/teams", headers=headers)
        assert response.status_code == 401
    
    def test_authentication_error_format(self):
        """Test that authentication errors follow consistent format"""
        client = TestClient(app)
        response = client.get("/admin/users")
        
        assert response.status_code == 401
        error = response.json()
        
        # Check error format
        assert "error" in error
        assert "message" in error
        assert "timestamp" in error
        
        # Should not expose sensitive information
        assert "password" not in str(error).lower()
        assert "secret" not in str(error).lower()
    
    def test_cors_headers(self):
        """Test that CORS headers are properly set"""
        client = TestClient(app)
        
        # Make an OPTIONS request
        response = client.options("/health/")
        
        assert "Access-Control-Allow-Origin" in response.headers
        assert "Access-Control-Allow-Methods" in response.headers
        assert "Access-Control-Allow-Headers" in response.headers
    
    def test_security_headers(self):
        """Test that security headers are present"""
        client = TestClient(app)
        response = client.get("/health/")
        
        # Check security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-XSS-Protection" in response.headers
        assert "Referrer-Policy" in response.headers
        
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"


class TestLoadTesting:
    """Test API performance under high traffic"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self):
        """Test API performance with concurrent requests"""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            # Create multiple concurrent requests
            tasks = []
            for _ in range(50):  # 50 concurrent requests
                task = client.get("/health/")
                tasks.append(task)
            
            start_time = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Check that most requests succeeded
            successful_responses = [r for r in responses if not isinstance(r, Exception)]
            assert len(successful_responses) >= 45  # Allow some failures
            
            # Check response times
            total_time = end_time - start_time
            avg_response_time = total_time / len(successful_responses)
            
            # Should handle 50 requests in reasonable time
            assert avg_response_time < 1.0  # Less than 1 second average
    
    @pytest.mark.asyncio
    async def test_rate_limiting_under_load(self):
        """Test that rate limiting works correctly under high load"""
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            # Make requests rapidly to trigger rate limiting
            responses = []
            for _ in range(150):  # Exceed typical rate limits
                try:
                    response = await client.get("/health/")
                    responses.append(response)
                except Exception as e:
                    responses.append(e)
            
            # Should have some rate limited responses
            status_codes = [r.status_code for r in responses if hasattr(r, 'status_code')]
            assert 429 in status_codes  # Rate limit exceeded
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test that memory usage remains stable under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            # Make many requests
            for _ in range(100):
                await client.get("/health/")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024
    
    def test_response_time_consistency(self):
        """Test that response times are consistent"""
        client = TestClient(app)
        response_times = []
        
        for _ in range(20):
            start_time = time.time()
            response = client.get("/health/")
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        # Calculate statistics
        avg_time = sum(response_times) / len(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        
        # Response times should be consistent
        assert max_time - min_time < 0.5  # Less than 500ms variation
        assert avg_time < 0.1  # Average less than 100ms


class TestMonitoringSystem:
    """Test monitoring and alerting system validation"""
    
    def test_health_check_endpoints(self):
        """Test that health check endpoints work correctly"""
        client = TestClient(app)
        
        # Basic health check
        response = client.get("/health/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        
        # Liveness check
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"
        
        # Readiness check (may fail in test environment)
        response = client.get("/health/ready")
        assert response.status_code in [200, 503]  # May not be ready in tests
    
    def test_metrics_endpoint(self):
        """Test that metrics endpoint returns Prometheus format"""
        client = TestClient(app)
        response = client.get("/health/metrics")
        
        assert response.status_code == 200
        metrics_text = response.text
        
        # Should contain Prometheus-style metrics
        assert "nfl_analyzer_" in metrics_text
        assert "cpu_usage_percent" in metrics_text or "memory_usage_percent" in metrics_text
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self):
        """Test that metrics are collected correctly"""
        # Test counter increment
        metrics_collector.increment_counter("test_counter", 1.0, {"label": "test"})
        
        # Test gauge setting
        metrics_collector.set_gauge("test_gauge", 42.0, {"label": "test"})
        
        # Verify metrics are stored
        counter_metrics = metrics_collector.get_metrics("test_counter")
        gauge_metrics = metrics_collector.get_metrics("test_gauge")
        
        assert len(counter_metrics) > 0
        assert len(gauge_metrics) > 0
        assert counter_metrics[-1].value == 1.0
        assert gauge_metrics[-1].value == 42.0
    
    @pytest.mark.asyncio
    async def test_alert_creation(self):
        """Test that alerts are created and processed correctly"""
        from app.core.monitoring import AlertType, AlertSeverity
        
        # Create test alert
        alert = await alert_manager.create_alert(
            AlertType.ERROR_RATE,
            AlertSeverity.HIGH,
            "Test Alert",
            "This is a test alert",
            {"test": True}
        )
        
        assert alert.type == AlertType.ERROR_RATE
        assert alert.severity == AlertSeverity.HIGH
        assert alert.title == "Test Alert"
        assert not alert.resolved
        
        # Test alert resolution
        alert_manager.resolve_alert(alert.id)
        assert alert.resolved
        assert alert.resolved_at is not None
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test that performance monitoring works"""
        # Mock some metrics for testing
        metrics_collector.increment_counter("api_requests", 100)
        metrics_collector.increment_counter("api_errors", 10)
        metrics_collector.set_gauge("system_cpu_percent", 85.0)
        
        # Run performance checks
        await performance_monitor.run_checks()
        
        # Check if alerts were created for high CPU
        active_alerts = alert_manager.get_active_alerts()
        cpu_alerts = [a for a in active_alerts if "CPU" in a.title]
        
        # Should have created CPU alert
        assert len(cpu_alerts) > 0
    
    def test_correlation_id_tracking(self):
        """Test that correlation IDs are properly tracked"""
        client = TestClient(app)
        response = client.get("/health/")
        
        # Should have correlation ID in response headers
        assert "X-Correlation-ID" in response.headers
        
        # Correlation ID should be valid UUID format
        correlation_id = response.headers["X-Correlation-ID"]
        assert len(correlation_id) > 0
        assert "-" in correlation_id  # Basic UUID format check
    
    @pytest.mark.asyncio
    async def test_error_tracking(self):
        """Test that errors are properly tracked and monitored"""
        client = TestClient(app)
        
        # Make request that should cause error
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        # Error should be tracked in metrics
        # (In real implementation, this would be verified through metrics)
        
        # Check error response format
        if response.headers.get("content-type", "").startswith("application/json"):
            error = response.json()
            assert "error" in error or "detail" in error


class TestAPICompliance:
    """Test API compliance with standards and best practices"""
    
    def test_http_methods_compliance(self):
        """Test that HTTP methods are used correctly"""
        client = TestClient(app)
        
        # GET should be idempotent
        response1 = client.get("/health/")
        response2 = client.get("/health/")
        assert response1.status_code == response2.status_code
        
        # POST should not be idempotent (for creation)
        # OPTIONS should be supported for CORS
        response = client.options("/health/")
        assert response.status_code in [200, 204]
    
    def test_content_type_headers(self):
        """Test that content-type headers are correct"""
        client = TestClient(app)
        
        # JSON endpoints should return application/json
        response = client.get("/health/")
        assert "application/json" in response.headers.get("content-type", "")
        
        # Metrics endpoint should return text/plain
        response = client.get("/health/metrics")
        assert "text/plain" in response.headers.get("content-type", "")
    
    def test_status_code_usage(self):
        """Test that HTTP status codes are used correctly"""
        client = TestClient(app)
        
        # Successful requests
        response = client.get("/health/")
        assert response.status_code == 200
        
        # Not found
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        # Unauthorized
        response = client.get("/admin/users")
        assert response.status_code == 401
    
    def test_api_versioning(self):
        """Test that API versioning is handled correctly"""
        client = TestClient(app)
        response = client.get("/openapi.json")
        schema = response.json()
        
        # Should have version in API info
        assert "version" in schema["info"]
        version = schema["info"]["version"]
        assert len(version) > 0
        assert "." in version  # Should be semantic versioning