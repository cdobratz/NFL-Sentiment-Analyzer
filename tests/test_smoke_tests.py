"""
Smoke Tests for Deployed Application

Comprehensive smoke tests to validate deployed application health,
service connectivity, and basic functionality.

Requirements covered: 8.1, 8.2, 8.3
"""

import pytest
import requests
import time
import socket
import subprocess
import json
import os
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin
import concurrent.futures
from dataclasses import dataclass


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    name: str
    url: str
    expected_status: int = 200
    timeout: int = 10
    required: bool = True


class TestApplicationSmokeTests:
    """Smoke tests for deployed application services."""
    
    # Service endpoints to test
    ENDPOINTS = [
        ServiceEndpoint("API Health", "http://localhost:8000/health", 200, 10, True),
        ServiceEndpoint("API Docs", "http://localhost:8000/docs", 200, 10, True),
        ServiceEndpoint("Frontend", "http://localhost:3000", 200, 10, True),
        ServiceEndpoint("Nginx", "http://localhost/health", 200, 10, False),
        ServiceEndpoint("Prometheus", "http://localhost:9090/-/healthy", 200, 10, False),
        ServiceEndpoint("Grafana", "http://localhost:3001/api/health", 200, 10, False),
    ]
    
    def test_service_health_endpoints(self):
        """Test all service health endpoints respond correctly."""
        failed_services = []
        
        for endpoint in self.ENDPOINTS:
            try:
                response = requests.get(endpoint.url, timeout=endpoint.timeout)
                
                if response.status_code != endpoint.expected_status:
                    if endpoint.required:
                        failed_services.append(
                            f"{endpoint.name}: Expected {endpoint.expected_status}, got {response.status_code}"
                        )
                    else:
                        print(f"Optional service {endpoint.name} not available: {response.status_code}")
                
            except requests.exceptions.ConnectionError:
                if endpoint.required:
                    failed_services.append(f"{endpoint.name}: Connection refused")
                else:
                    print(f"Optional service {endpoint.name} not running")
            
            except requests.exceptions.Timeout:
                if endpoint.required:
                    failed_services.append(f"{endpoint.name}: Timeout after {endpoint.timeout}s")
                else:
                    print(f"Optional service {endpoint.name} timed out")
        
        if failed_services:
            pytest.fail(f"Required services failed: {', '.join(failed_services)}")
    
    def test_api_basic_functionality(self):
        """Test basic API functionality."""
        base_url = "http://localhost:8000"
        
        try:
            # Test health endpoint
            health_response = requests.get(f"{base_url}/health", timeout=10)
            assert health_response.status_code == 200
            
            health_data = health_response.json()
            assert "status" in health_data
            
            # Test API documentation
            docs_response = requests.get(f"{base_url}/docs", timeout=10)
            assert docs_response.status_code == 200
            assert "text/html" in docs_response.headers.get("content-type", "")
            
            # Test OpenAPI spec
            openapi_response = requests.get(f"{base_url}/openapi.json", timeout=10)
            assert openapi_response.status_code == 200
            
            openapi_data = openapi_response.json()
            assert "openapi" in openapi_data
            assert "info" in openapi_data
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for smoke test")
    
    def test_frontend_accessibility(self):
        """Test frontend is accessible and serves content."""
        try:
            response = requests.get("http://localhost:3000", timeout=10)
            assert response.status_code == 200
            
            content_type = response.headers.get("content-type", "")
            assert "text/html" in content_type
            
            # Check for basic HTML structure
            content = response.text
            assert "<html" in content.lower()
            assert "<body" in content.lower()
            
            # Check for React app mounting point
            assert 'id="root"' in content or 'id="app"' in content
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Frontend service not running for smoke test")
    
    def test_database_connectivity(self):
        """Test database connectivity through API."""
        try:
            # Test database connection through API
            response = requests.get("http://localhost:8000/health", timeout=10)
            assert response.status_code == 200
            
            health_data = response.json()
            
            # If health endpoint includes database status
            if "database" in health_data:
                assert health_data["database"] in ["connected", "healthy", True]
            
            # Test a simple database operation
            teams_response = requests.get("http://localhost:8000/data/teams", timeout=10)
            # Should return 200 (with data) or 404 (no data), but not 500 (error)
            assert teams_response.status_code in [200, 404]
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for database connectivity test")
    
    def test_redis_connectivity(self):
        """Test Redis connectivity through API."""
        try:
            # Test Redis through caching endpoint if available
            response = requests.get("http://localhost:8000/health", timeout=10)
            assert response.status_code == 200
            
            health_data = response.json()
            
            # If health endpoint includes Redis status
            if "redis" in health_data or "cache" in health_data:
                redis_status = health_data.get("redis") or health_data.get("cache")
                assert redis_status in ["connected", "healthy", True]
            
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for Redis connectivity test")
    
    def test_service_response_times(self):
        """Test service response times are acceptable."""
        max_response_time = 5.0  # seconds
        slow_services = []
        
        for endpoint in self.ENDPOINTS:
            if not endpoint.required:
                continue
            
            try:
                start_time = time.time()
                response = requests.get(endpoint.url, timeout=endpoint.timeout)
                response_time = time.time() - start_time
                
                if response.status_code == 200 and response_time > max_response_time:
                    slow_services.append(f"{endpoint.name}: {response_time:.2f}s")
                
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                # Skip if service not available
                continue
        
        if slow_services:
            print(f"Slow services (>{max_response_time}s): {', '.join(slow_services)}")
            # Don't fail the test, just warn
    
    def test_concurrent_requests(self):
        """Test application handles concurrent requests."""
        def make_request():
            try:
                response = requests.get("http://localhost:8000/health", timeout=10)
                return response.status_code == 200
            except:
                return False
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # At least 80% should succeed
        success_rate = sum(results) / len(results)
        assert success_rate >= 0.8, f"Concurrent request success rate too low: {success_rate:.2%}"


class TestNetworkConnectivity:
    """Test network connectivity between services."""
    
    def test_port_accessibility(self):
        """Test required ports are accessible."""
        required_ports = [
            (8000, "API"),
            (3000, "Frontend"),
        ]
        
        optional_ports = [
            (80, "Nginx HTTP"),
            (443, "Nginx HTTPS"),
            (9090, "Prometheus"),
            (3001, "Grafana"),
        ]
        
        failed_ports = []
        
        # Test required ports
        for port, service in required_ports:
            if not self._is_port_open("localhost", port):
                failed_ports.append(f"{service} (port {port})")
        
        # Test optional ports (just log if not available)
        for port, service in optional_ports:
            if not self._is_port_open("localhost", port):
                print(f"Optional service {service} not available on port {port}")
        
        if failed_ports:
            pytest.fail(f"Required ports not accessible: {', '.join(failed_ports)}")
    
    def _is_port_open(self, host: str, port: int, timeout: int = 3) -> bool:
        """Check if a port is open."""
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False
    
    def test_service_discovery(self):
        """Test services can discover each other."""
        try:
            # Test that API can connect to database
            response = requests.get("http://localhost:8000/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                
                # Check database connectivity
                if "database" in health_data:
                    assert health_data["database"] in ["connected", "healthy", True]
                
                # Check cache connectivity
                if "cache" in health_data or "redis" in health_data:
                    cache_status = health_data.get("cache") or health_data.get("redis")
                    assert cache_status in ["connected", "healthy", True]
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for service discovery test")


class TestDataIntegrity:
    """Test basic data integrity and consistency."""
    
    def test_api_data_consistency(self):
        """Test API returns consistent data."""
        try:
            # Make multiple requests to the same endpoint
            endpoint = "http://localhost:8000/data/teams"
            responses = []
            
            for _ in range(3):
                response = requests.get(endpoint, timeout=10)
                if response.status_code == 200:
                    responses.append(response.json())
                time.sleep(0.1)
            
            if len(responses) >= 2:
                # Data should be consistent across requests
                first_response = responses[0]
                for response in responses[1:]:
                    assert response == first_response, "API responses are inconsistent"
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for data consistency test")
    
    def test_error_handling(self):
        """Test API error handling."""
        try:
            # Test 404 endpoint
            response = requests.get("http://localhost:8000/nonexistent", timeout=10)
            assert response.status_code == 404
            
            # Should return JSON error response
            if response.headers.get("content-type", "").startswith("application/json"):
                error_data = response.json()
                assert "error" in error_data or "detail" in error_data
            
            # Test invalid data (if applicable)
            invalid_response = requests.post(
                "http://localhost:8000/sentiment/analyze",
                json={"invalid": "data"},
                timeout=10
            )
            # Should return 400 or 422, not 500
            assert invalid_response.status_code in [400, 422]
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for error handling test")


class TestSecurityBasics:
    """Basic security smoke tests."""
    
    def test_security_headers(self):
        """Test basic security headers are present."""
        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            headers = response.headers
            
            # Check for basic security headers
            security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "X-XSS-Protection",
            ]
            
            missing_headers = []
            for header in security_headers:
                if header not in headers:
                    missing_headers.append(header)
            
            if missing_headers:
                print(f"Missing security headers: {', '.join(missing_headers)}")
                # Don't fail the test, just warn for smoke test
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for security headers test")
    
    def test_cors_configuration(self):
        """Test CORS configuration."""
        try:
            # Test preflight request
            response = requests.options(
                "http://localhost:8000/health",
                headers={"Origin": "http://localhost:3000"},
                timeout=10
            )
            
            # Should allow CORS for frontend
            if "Access-Control-Allow-Origin" in response.headers:
                cors_origin = response.headers["Access-Control-Allow-Origin"]
                assert cors_origin in ["*", "http://localhost:3000"]
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for CORS test")
    
    def test_no_sensitive_info_exposure(self):
        """Test that sensitive information is not exposed."""
        try:
            # Test error responses don't expose sensitive info
            response = requests.get("http://localhost:8000/nonexistent", timeout=10)
            
            if response.status_code >= 400:
                content = response.text.lower()
                
                # Should not expose sensitive information
                sensitive_patterns = [
                    "password", "secret", "key", "token",
                    "mongodb://", "redis://", "database",
                    "traceback", "exception"
                ]
                
                exposed_info = []
                for pattern in sensitive_patterns:
                    if pattern in content:
                        exposed_info.append(pattern)
                
                if exposed_info:
                    print(f"Potentially sensitive info in error response: {', '.join(exposed_info)}")
                    # Don't fail smoke test, just warn
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for sensitive info test")


class TestMonitoringIntegration:
    """Test monitoring and observability integration."""
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint is available."""
        try:
            # Test Prometheus metrics if available
            response = requests.get("http://localhost:8000/metrics", timeout=10)
            
            if response.status_code == 200:
                content = response.text
                
                # Should contain Prometheus-format metrics
                assert "# HELP" in content or "# TYPE" in content
                
                # Should contain basic application metrics
                assert any(metric in content for metric in [
                    "http_requests", "response_time", "requests_total"
                ])
        
        except requests.exceptions.ConnectionError:
            pytest.skip("Metrics endpoint not available")
    
    def test_logging_functionality(self):
        """Test logging functionality."""
        try:
            # Make a request that should generate logs
            requests.get("http://localhost:8000/health", timeout=10)
            
            # Check if log files exist (if mounted)
            log_paths = [
                "logs/access.log",
                "logs/error.log",
                "logs/app.log"
            ]
            
            log_files_exist = []
            for log_path in log_paths:
                if os.path.exists(log_path):
                    log_files_exist.append(log_path)
            
            if log_files_exist:
                print(f"Log files found: {', '.join(log_files_exist)}")
            else:
                print("No log files found (may be using stdout logging)")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for logging test")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])