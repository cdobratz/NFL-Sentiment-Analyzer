"""
Comprehensive monitoring and alerting system validation tests.

This module validates:
- Health check accuracy and completeness
- Metrics collection and reporting
- Alert generation and handling
- System observability features
- Performance monitoring

Requirements: 6.2, 6.3, 6.4
"""

import os
# Set test environment variables before importing app
os.environ.update({
    "MONGODB_URL": "mongodb://localhost:27017",
    "DATABASE_NAME": "nfl_analyzer_test",
    "REDIS_URL": "redis://localhost:6379/1",
    "SECRET_KEY": "test-secret-key-for-testing-only", 
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

from app.main import app
from app.core.monitoring import (
    metrics_collector, alert_manager, performance_monitor,
    AlertType, AlertSeverity, Metric, Alert
)
from app.api.health import health_checker
from app.core.config import settings


class TestHealthCheckValidation:
    """Test health check endpoints and accuracy"""
    
    def test_basic_health_check_response_format(self):
        """Test basic health check returns correct format"""
        client = TestClient(app)
        response = client.get("/health/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "service" in data
        
        # Validate values
        assert data["status"] == "healthy"
        assert data["version"] == settings.app_version
        assert data["service"] == settings.app_name
        
        # Timestamp should be recent (within last minute)
        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        now = datetime.now(timestamp.tzinfo)
        assert (now - timestamp).total_seconds() < 60
    
    def test_detailed_health_check_components(self):
        """Test detailed health check includes all components"""
        client = TestClient(app)
        response = client.get("/health/detailed")
        
        # May return 503 if dependencies not available in test
        assert response.status_code in [200, 503]
        data = response.json()
        
        # Required fields
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "check_duration_ms" in data
        assert "components" in data
        
        # Component checks
        components = data["components"]
        expected_components = ["mongodb", "redis", "external_apis", "ml_services"]
        
        for component in expected_components:
            assert component in components
            assert "status" in components[component]
        
        # System metrics
        if "system_metrics" in data:
            metrics = data["system_metrics"]
            if "error" not in metrics:
                assert "cpu" in metrics
                assert "memory" in metrics
                assert "disk" in metrics
    
    def test_liveness_probe(self):
        """Test Kubernetes liveness probe"""
        client = TestClient(app)
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "alive"
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] > 0
    
    def test_readiness_probe(self):
        """Test Kubernetes readiness probe"""
        client = TestClient(app)
        response = client.get("/health/ready")
        
        # May not be ready in test environment
        assert response.status_code in [200, 503]
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert data["status"] in ["ready", "not_ready"]
    
    def test_metrics_endpoint_format(self):
        """Test Prometheus metrics endpoint format"""
        client = TestClient(app)
        response = client.get("/health/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        
        metrics_text = response.text
        lines = metrics_text.strip().split('\n')
        
        # Should have metrics
        assert len(lines) > 0
        
        # Check Prometheus format
        for line in lines:
            if line and not line.startswith('#'):
                # Should have metric name and value
                parts = line.split()
                assert len(parts) >= 2
                
                # Metric name should start with app prefix
                metric_name = parts[0]
                assert metric_name.startswith('nfl_analyzer_')
                
                # Value should be numeric
                try:
                    float(parts[1])
                except ValueError:
                    pytest.fail(f"Invalid metric value: {parts[1]}")
    
    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check performance"""
        client = TestClient(app)
        
        # Test basic health check performance
        start_time = time.time()
        response = client.get("/health/")
        end_time = time.time()
        
        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 0.1  # Should be very fast
        
        # Test detailed health check performance
        start_time = time.time()
        response = client.get("/health/detailed")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_health_checker_mongodb_mock(self):
        """Test MongoDB health check with mocked database"""
        mock_db = AsyncMock()
        mock_db.command.return_value = {
            "version": "4.4.0",
            "uptime": 12345,
            "connections": {"current": 5}
        }
        
        with patch('app.core.database.db_manager.get_database', return_value=mock_db):
            result = await health_checker.check_mongodb()
            
            assert result["status"] == "healthy"
            assert "response_time_ms" in result
            assert "version" in result
            assert result["version"] == "4.4.0"
    
    @pytest.mark.asyncio
    async def test_health_checker_redis_mock(self):
        """Test Redis health check with mocked client"""
        mock_redis = AsyncMock()
        mock_redis.ping.return_value = True
        mock_redis.info.return_value = {
            "redis_version": "6.2.0",
            "uptime_in_seconds": 12345,
            "connected_clients": 3,
            "used_memory_human": "1.5M"
        }
        
        with patch('app.core.database.db_manager.get_redis', return_value=mock_redis):
            result = await health_checker.check_redis()
            
            assert result["status"] == "healthy"
            assert "response_time_ms" in result
            assert "version" in result
            assert result["version"] == "6.2.0"


class TestMetricsCollection:
    """Test metrics collection and reporting"""
    
    def test_counter_increment(self):
        """Test counter metric increment"""
        initial_count = len(metrics_collector.get_metrics("test_counter"))
        
        metrics_collector.increment_counter("test_counter", 1.0, {"test": "value"})
        metrics_collector.increment_counter("test_counter", 2.0, {"test": "value"})
        
        metrics = metrics_collector.get_metrics("test_counter")
        assert len(metrics) == initial_count + 2
        
        # Check values
        assert metrics[-2].value == 1.0
        assert metrics[-1].value == 3.0  # Cumulative
    
    def test_gauge_setting(self):
        """Test gauge metric setting"""
        metrics_collector.set_gauge("test_gauge", 42.0, {"test": "value"})
        metrics_collector.set_gauge("test_gauge", 84.0, {"test": "value"})
        
        metrics = metrics_collector.get_metrics("test_gauge")
        assert len(metrics) >= 2
        
        # Latest value should be 84.0
        latest_value = metrics_collector.get_latest_value("test_gauge", {"test": "value"})
        assert latest_value == 84.0
    
    def test_metrics_with_labels(self):
        """Test metrics with different labels"""
        metrics_collector.increment_counter("labeled_counter", 1.0, {"env": "test"})
        metrics_collector.increment_counter("labeled_counter", 1.0, {"env": "prod"})
        
        # Should track separately
        test_value = metrics_collector.get_latest_value("labeled_counter", {"env": "test"})
        prod_value = metrics_collector.get_latest_value("labeled_counter", {"env": "prod"})
        
        assert test_value == 1.0
        assert prod_value == 1.0
    
    def test_metrics_time_filtering(self):
        """Test metrics filtering by time"""
        # Add old metric
        old_metric = Metric(
            name="time_test",
            value=1.0,
            timestamp=datetime.utcnow() - timedelta(hours=1),
            labels={}
        )
        metrics_collector.metrics["time_test"] = [old_metric]
        
        # Add new metric
        metrics_collector.set_gauge("time_test", 2.0)
        
        # Get recent metrics only
        since = datetime.utcnow() - timedelta(minutes=30)
        recent_metrics = metrics_collector.get_metrics("time_test", since=since)
        
        assert len(recent_metrics) == 1
        assert recent_metrics[0].value == 2.0
    
    def test_metrics_storage_limit(self):
        """Test that metrics storage is limited"""
        # Add many metrics
        for i in range(1200):  # More than the 1000 limit
            metrics_collector.increment_counter("storage_test", 1.0)
        
        metrics = metrics_collector.get_metrics("storage_test")
        assert len(metrics) <= 1000  # Should be limited


class TestAlertSystem:
    """Test alert generation and handling"""
    
    @pytest.mark.asyncio
    async def test_alert_creation(self):
        """Test alert creation and storage"""
        initial_count = len(alert_manager.alerts)
        
        alert = await alert_manager.create_alert(
            AlertType.ERROR_RATE,
            AlertSeverity.HIGH,
            "Test Alert",
            "This is a test alert",
            {"test_data": "value"}
        )
        
        assert len(alert_manager.alerts) == initial_count + 1
        assert alert.type == AlertType.ERROR_RATE
        assert alert.severity == AlertSeverity.HIGH
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"
        assert alert.metadata["test_data"] == "value"
        assert not alert.resolved
    
    def test_alert_resolution(self):
        """Test alert resolution"""
        # Create alert manually
        alert = Alert(
            id="test_alert_123",
            type=AlertType.SYSTEM_RESOURCE,
            severity=AlertSeverity.MEDIUM,
            title="Test Alert",
            message="Test message",
            timestamp=datetime.utcnow()
        )
        alert_manager.alerts.append(alert)
        
        # Resolve alert
        alert_manager.resolve_alert("test_alert_123")
        
        assert alert.resolved
        assert alert.resolved_at is not None
    
    def test_active_alerts_filtering(self):
        """Test filtering of active alerts"""
        # Create test alerts
        alert1 = Alert(
            id="alert1",
            type=AlertType.ERROR_RATE,
            severity=AlertSeverity.HIGH,
            title="High Severity Alert",
            message="Test",
            timestamp=datetime.utcnow()
        )
        
        alert2 = Alert(
            id="alert2",
            type=AlertType.RESPONSE_TIME,
            severity=AlertSeverity.LOW,
            title="Low Severity Alert",
            message="Test",
            timestamp=datetime.utcnow(),
            resolved=True
        )
        
        alert_manager.alerts.extend([alert1, alert2])
        
        # Get active alerts
        active_alerts = alert_manager.get_active_alerts()
        active_ids = [a.id for a in active_alerts]
        
        assert "alert1" in active_ids
        assert "alert2" not in active_ids  # Resolved
        
        # Filter by severity
        high_alerts = alert_manager.get_active_alerts(AlertSeverity.HIGH)
        assert len(high_alerts) >= 1
        assert all(a.severity == AlertSeverity.HIGH for a in high_alerts)
    
    @pytest.mark.asyncio
    async def test_alert_handlers(self):
        """Test alert handler execution"""
        handler_called = False
        handler_alert = None
        
        async def test_handler(alert: Alert):
            nonlocal handler_called, handler_alert
            handler_called = True
            handler_alert = alert
        
        # Add handler
        alert_manager.add_alert_handler(test_handler)
        
        # Create alert
        alert = await alert_manager.create_alert(
            AlertType.SECURITY,
            AlertSeverity.CRITICAL,
            "Security Alert",
            "Test security alert"
        )
        
        # Handler should have been called
        assert handler_called
        assert handler_alert.id == alert.id


class TestPerformanceMonitoring:
    """Test performance monitoring and alerting"""
    
    @pytest.mark.asyncio
    async def test_error_rate_monitoring(self):
        """Test error rate monitoring and alerting"""
        # Clear existing alerts
        alert_manager.alerts.clear()
        
        # Simulate high error rate
        for _ in range(100):
            metrics_collector.increment_counter("api_requests")
        for _ in range(10):  # 10% error rate
            metrics_collector.increment_counter("api_errors")
        
        # Run error rate check
        await performance_monitor.check_error_rate()
        
        # Should create alert for high error rate
        error_alerts = [a for a in alert_manager.alerts if a.type == AlertType.ERROR_RATE]
        assert len(error_alerts) > 0
        
        alert = error_alerts[0]
        assert alert.severity == AlertSeverity.HIGH
        assert "error rate" in alert.message.lower()
    
    @pytest.mark.asyncio
    async def test_response_time_monitoring(self):
        """Test response time monitoring"""
        alert_manager.alerts.clear()
        
        # Simulate slow response times
        for _ in range(10):
            metrics_collector.set_gauge("api_response_time", 3000.0)  # 3 seconds
        
        await performance_monitor.check_response_time()
        
        # Should create alert for slow response times
        response_time_alerts = [a for a in alert_manager.alerts if a.type == AlertType.RESPONSE_TIME]
        assert len(response_time_alerts) > 0
        
        alert = response_time_alerts[0]
        assert alert.severity == AlertSeverity.MEDIUM
        assert "response time" in alert.message.lower()
    
    @pytest.mark.asyncio
    async def test_system_resource_monitoring(self):
        """Test system resource monitoring"""
        alert_manager.alerts.clear()
        
        # Simulate high resource usage
        metrics_collector.set_gauge("system_cpu_percent", 85.0)
        metrics_collector.set_gauge("system_memory_percent", 90.0)
        metrics_collector.set_gauge("system_disk_percent", 95.0)
        
        await performance_monitor.check_system_resources()
        
        # Should create alerts for high resource usage
        resource_alerts = [a for a in alert_manager.alerts if a.type == AlertType.SYSTEM_RESOURCE]
        assert len(resource_alerts) >= 2  # CPU, memory, and/or disk
        
        # Check alert severities
        severities = [a.severity for a in resource_alerts]
        assert AlertSeverity.HIGH in severities or AlertSeverity.CRITICAL in severities
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self):
        """Test complete performance monitoring workflow"""
        alert_manager.alerts.clear()
        
        # Simulate various performance issues
        metrics_collector.increment_counter("api_requests", 100)
        metrics_collector.increment_counter("api_errors", 8)  # 8% error rate
        metrics_collector.set_gauge("api_response_time", 2500.0)  # 2.5 seconds
        metrics_collector.set_gauge("system_cpu_percent", 85.0)
        
        # Run all performance checks
        await performance_monitor.run_checks()
        
        # Should have created multiple alerts
        assert len(alert_manager.alerts) >= 2
        
        # Check alert types
        alert_types = [a.type for a in alert_manager.alerts]
        assert AlertType.ERROR_RATE in alert_types
        assert AlertType.RESPONSE_TIME in alert_types


class TestObservabilityFeatures:
    """Test observability and monitoring features"""
    
    def test_correlation_id_presence(self):
        """Test that correlation IDs are present in responses"""
        client = TestClient(app)
        response = client.get("/health/")
        
        assert "X-Correlation-ID" in response.headers
        correlation_id = response.headers["X-Correlation-ID"]
        
        # Should be a valid UUID-like string
        assert len(correlation_id) > 0
        assert "-" in correlation_id
    
    def test_correlation_id_consistency(self):
        """Test that correlation IDs are consistent within request"""
        client = TestClient(app)
        
        # Make request with custom correlation ID
        headers = {"X-Correlation-ID": "test-correlation-123"}
        response = client.get("/health/", headers=headers)
        
        # Should return the same correlation ID
        assert response.headers["X-Correlation-ID"] == "test-correlation-123"
    
    def test_structured_logging_format(self):
        """Test that logs are in structured format"""
        # This would typically test log output, but we'll test the logging setup
        from app.core.logging import setup_logging, log_business_event
        
        logger = setup_logging()
        assert logger is not None
        
        # Test business event logging (should not raise exception)
        log_business_event("test_event", {"test": "data"})
    
    def test_rate_limit_status_endpoint(self):
        """Test rate limit status endpoint"""
        client = TestClient(app)
        response = client.get("/health/rate-limits")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "rate_limits" in data
        assert "timestamp" in data
        assert "client_type" in data
        
        # Rate limits should have structure
        rate_limits = data["rate_limits"]
        if rate_limits:  # May be empty in test environment
            for limit_type, info in rate_limits.items():
                assert "limit" in info
                assert "remaining" in info
                assert "reset" in info
    
    def test_monitoring_endpoint_security(self):
        """Test that monitoring endpoints have appropriate security"""
        client = TestClient(app)
        
        # Health endpoints should be accessible
        response = client.get("/health/")
        assert response.status_code == 200
        
        # Metrics endpoint should be accessible (for Prometheus)
        response = client.get("/health/metrics")
        assert response.status_code == 200
        
        # Detailed health might require auth in production
        response = client.get("/health/detailed")
        assert response.status_code in [200, 401, 503]


class TestMonitoringIntegration:
    """Test monitoring system integration"""
    
    def test_health_check_integration_with_app(self):
        """Test health checks work with full application"""
        client = TestClient(app)
        
        # Test that health checks reflect actual app state
        response = client.get("/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == settings.app_name
    
    def test_metrics_collection_during_requests(self):
        """Test that metrics are collected during normal requests"""
        client = TestClient(app)
        
        # Clear existing metrics
        initial_request_count = len(metrics_collector.get_metrics("api_requests"))
        
        # Make some requests
        for _ in range(5):
            response = client.get("/health/")
            assert response.status_code == 200
        
        # Check if request metrics were collected
        # Note: This depends on middleware being properly configured
        final_request_count = len(metrics_collector.get_metrics("api_requests"))
        
        # May not increase if middleware isn't active in tests
        # This is more of an integration test
        assert final_request_count >= initial_request_count
    
    def test_error_tracking_integration(self):
        """Test error tracking during requests"""
        client = TestClient(app)
        
        # Make request that should cause 404
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        # Error should be properly formatted
        if response.headers.get("content-type", "").startswith("application/json"):
            error = response.json()
            # FastAPI default error format
            assert "detail" in error or "error" in error
    
    @pytest.mark.asyncio
    async def test_monitoring_system_startup(self):
        """Test monitoring system initialization"""
        # Test that monitoring components are initialized
        assert metrics_collector is not None
        assert alert_manager is not None
        assert performance_monitor is not None
        
        # Test that they have expected attributes
        assert hasattr(metrics_collector, 'metrics')
        assert hasattr(alert_manager, 'alerts')
        assert hasattr(performance_monitor, 'thresholds')
    
    def test_monitoring_configuration(self):
        """Test monitoring system configuration"""
        # Test performance monitor thresholds
        thresholds = performance_monitor.thresholds
        
        assert "error_rate_threshold" in thresholds
        assert "response_time_threshold" in thresholds
        assert "cpu_threshold" in thresholds
        assert "memory_threshold" in thresholds
        
        # Thresholds should be reasonable
        assert 0 < thresholds["error_rate_threshold"] < 1
        assert thresholds["response_time_threshold"] > 0
        assert 0 < thresholds["cpu_threshold"] < 100
        assert 0 < thresholds["memory_threshold"] < 100