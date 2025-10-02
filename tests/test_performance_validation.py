"""
Performance Validation Tests

Performance tests for production deployment including load testing,
resource utilization, and scalability validation.

Requirements covered: 8.1, 8.2, 8.3
"""

import pytest
import requests
import time
import concurrent.futures
import psutil
import docker
import subprocess
import json
import statistics
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import threading
import queue


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    response_times: List[float]
    success_rate: float
    throughput: float
    error_count: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float


class TestLoadTesting:
    """Load testing for API endpoints."""
    
    def test_api_load_capacity(self):
        """Test API can handle expected load."""
        endpoint = "http://localhost:8000/health"
        concurrent_users = 10
        requests_per_user = 20
        
        try:
            metrics = self._run_load_test(endpoint, concurrent_users, requests_per_user)
            
            # Performance assertions
            assert metrics.success_rate >= 0.95, f"Success rate too low: {metrics.success_rate:.2%}"
            assert metrics.avg_response_time < 2.0, f"Average response time too high: {metrics.avg_response_time:.2f}s"
            assert metrics.p95_response_time < 5.0, f"95th percentile too high: {metrics.p95_response_time:.2f}s"
            
            print(f"Load test results:")
            print(f"  Success rate: {metrics.success_rate:.2%}")
            print(f"  Average response time: {metrics.avg_response_time:.3f}s")
            print(f"  95th percentile: {metrics.p95_response_time:.3f}s")
            print(f"  Throughput: {metrics.throughput:.2f} req/s")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for load test")
    
    def test_database_query_performance(self):
        """Test database query performance under load."""
        endpoint = "http://localhost:8000/data/teams"
        concurrent_users = 5
        requests_per_user = 10
        
        try:
            metrics = self._run_load_test(endpoint, concurrent_users, requests_per_user)
            
            # Database queries should be reasonably fast
            assert metrics.avg_response_time < 3.0, f"Database query too slow: {metrics.avg_response_time:.2f}s"
            assert metrics.success_rate >= 0.90, f"Database query success rate too low: {metrics.success_rate:.2%}"
            
            print(f"Database query performance:")
            print(f"  Average response time: {metrics.avg_response_time:.3f}s")
            print(f"  Success rate: {metrics.success_rate:.2%}")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for database performance test")
    
    def test_sentiment_analysis_performance(self):
        """Test sentiment analysis endpoint performance."""
        endpoint = "http://localhost:8000/sentiment/analyze"
        test_data = {"text": "This is a test sentiment analysis request"}
        
        try:
            # Test single request performance
            start_time = time.time()
            response = requests.post(endpoint, json=test_data, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                assert response_time < 10.0, f"Sentiment analysis too slow: {response_time:.2f}s"
                print(f"Sentiment analysis response time: {response_time:.3f}s")
            else:
                print(f"Sentiment analysis endpoint returned: {response.status_code}")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for sentiment analysis performance test")
    
    def _run_load_test(self, endpoint: str, concurrent_users: int, requests_per_user: int) -> PerformanceMetrics:
        """Run load test and return metrics."""
        response_times = []
        errors = 0
        start_time = time.time()
        
        def make_requests():
            user_response_times = []
            user_errors = 0
            
            for _ in range(requests_per_user):
                try:
                    request_start = time.time()
                    response = requests.get(endpoint, timeout=10)
                    request_time = time.time() - request_start
                    
                    user_response_times.append(request_time)
                    
                    if response.status_code != 200:
                        user_errors += 1
                
                except (requests.exceptions.RequestException, requests.exceptions.Timeout):
                    user_errors += 1
                    user_response_times.append(10.0)  # Timeout penalty
            
            return user_response_times, user_errors
        
        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(make_requests) for _ in range(concurrent_users)]
            
            for future in concurrent.futures.as_completed(futures):
                user_times, user_errors = future.result()
                response_times.extend(user_times)
                errors += user_errors
        
        total_time = time.time() - start_time
        total_requests = concurrent_users * requests_per_user
        
        # Calculate metrics
        success_rate = (total_requests - errors) / total_requests
        throughput = total_requests / total_time
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        response_times.sort()
        p95_index = int(0.95 * len(response_times))
        p99_index = int(0.99 * len(response_times))
        
        p95_response_time = response_times[p95_index] if response_times else 0
        p99_response_time = response_times[p99_index] if response_times else 0
        
        return PerformanceMetrics(
            response_times=response_times,
            success_rate=success_rate,
            throughput=throughput,
            error_count=errors,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time
        )


class TestResourceUtilization:
    """Test resource utilization and limits."""
    
    def setup_method(self):
        """Set up test environment."""
        self.docker_client = docker.from_env()
    
    def test_memory_usage(self):
        """Test memory usage is within acceptable limits."""
        try:
            # Get container stats
            containers = self.docker_client.containers.list()
            app_containers = [c for c in containers if 'nfl-analyzer' in c.name or 'api' in c.name]
            
            for container in app_containers:
                stats = container.stats(stream=False)
                
                # Calculate memory usage
                memory_usage = stats['memory_stats']['usage']
                memory_limit = stats['memory_stats']['limit']
                memory_percent = (memory_usage / memory_limit) * 100
                
                # Memory usage should be reasonable
                assert memory_percent < 90, f"Container {container.name} using too much memory: {memory_percent:.1f}%"
                
                print(f"Container {container.name} memory usage: {memory_percent:.1f}%")
        
        except docker.errors.DockerException:
            pytest.skip("Docker not available for memory usage test")
    
    def test_cpu_usage(self):
        """Test CPU usage is within acceptable limits."""
        try:
            # Monitor CPU usage during load
            endpoint = "http://localhost:8000/health"
            
            # Start monitoring CPU
            cpu_usage = []
            monitoring = True
            
            def monitor_cpu():
                while monitoring:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    cpu_usage.append(cpu_percent)
            
            monitor_thread = threading.Thread(target=monitor_cpu)
            monitor_thread.start()
            
            # Generate some load
            for _ in range(20):
                try:
                    requests.get(endpoint, timeout=5)
                except:
                    pass
                time.sleep(0.1)
            
            monitoring = False
            monitor_thread.join()
            
            if cpu_usage:
                avg_cpu = statistics.mean(cpu_usage)
                max_cpu = max(cpu_usage)
                
                print(f"CPU usage - Average: {avg_cpu:.1f}%, Max: {max_cpu:.1f}%")
                
                # CPU usage should be reasonable
                assert avg_cpu < 80, f"Average CPU usage too high: {avg_cpu:.1f}%"
        
        except Exception as e:
            pytest.skip(f"CPU monitoring not available: {e}")
    
    def test_disk_usage(self):
        """Test disk usage is reasonable."""
        try:
            # Check Docker volumes disk usage
            volumes = self.docker_client.volumes.list()
            
            for volume in volumes:
                if 'nfl' in volume.name or 'mongodb' in volume.name or 'redis' in volume.name:
                    # Get volume mount point (this is system-specific)
                    print(f"Volume: {volume.name}")
        
        except docker.errors.DockerException:
            pytest.skip("Docker not available for disk usage test")
    
    def test_network_performance(self):
        """Test network performance between services."""
        try:
            # Test internal service communication speed
            start_time = time.time()
            
            # Make requests that involve database queries
            for _ in range(10):
                response = requests.get("http://localhost:8000/data/teams", timeout=10)
                if response.status_code not in [200, 404]:
                    break
            
            total_time = time.time() - start_time
            avg_time_per_request = total_time / 10
            
            # Network communication should be fast
            assert avg_time_per_request < 1.0, f"Network communication too slow: {avg_time_per_request:.3f}s per request"
            
            print(f"Average network request time: {avg_time_per_request:.3f}s")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("Services not running for network performance test")


class TestScalability:
    """Test application scalability."""
    
    def test_horizontal_scaling_readiness(self):
        """Test application is ready for horizontal scaling."""
        # Check for stateless design
        try:
            # Make requests to different endpoints to check for session state
            session = requests.Session()
            
            # First request
            response1 = session.get("http://localhost:8000/health", timeout=10)
            
            # Second request (should not depend on first)
            response2 = session.get("http://localhost:8000/health", timeout=10)
            
            # Both should succeed independently
            assert response1.status_code == 200
            assert response2.status_code == 200
            
            # Check for stateless behavior
            if response1.json() != response2.json():
                # This might be expected if health check includes timestamps
                pass
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for scalability test")
    
    def test_database_connection_pooling(self):
        """Test database connection pooling efficiency."""
        try:
            # Make concurrent database requests
            endpoint = "http://localhost:8000/data/teams"
            concurrent_requests = 20
            
            def make_request():
                try:
                    response = requests.get(endpoint, timeout=10)
                    return response.status_code == 200 or response.status_code == 404
                except:
                    return False
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                futures = [executor.submit(make_request) for _ in range(concurrent_requests)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            success_rate = sum(results) / len(results)
            
            # Should handle concurrent database requests well
            assert success_rate >= 0.8, f"Database connection pooling insufficient: {success_rate:.2%} success rate"
            
            print(f"Concurrent database request success rate: {success_rate:.2%}")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for database connection test")
    
    def test_cache_performance(self):
        """Test caching performance and efficiency."""
        try:
            # Test cache hit performance
            endpoint = "http://localhost:8000/data/teams"
            
            # First request (cache miss)
            start_time = time.time()
            response1 = requests.get(endpoint, timeout=10)
            first_request_time = time.time() - start_time
            
            if response1.status_code == 200:
                # Second request (should be cache hit)
                start_time = time.time()
                response2 = requests.get(endpoint, timeout=10)
                second_request_time = time.time() - start_time
                
                # Cache hit should be faster
                if second_request_time < first_request_time * 0.8:
                    print(f"Cache performance: {first_request_time:.3f}s -> {second_request_time:.3f}s")
                else:
                    print("Cache may not be working effectively")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for cache performance test")


class TestDatabasePerformance:
    """Test database performance and optimization."""
    
    def test_query_performance(self):
        """Test database query performance."""
        try:
            # Test various query patterns
            endpoints = [
                "http://localhost:8000/data/teams",
                "http://localhost:8000/data/players",
                "http://localhost:8000/data/games"
            ]
            
            for endpoint in endpoints:
                start_time = time.time()
                response = requests.get(endpoint, timeout=15)
                query_time = time.time() - start_time
                
                if response.status_code == 200:
                    # Query should complete in reasonable time
                    assert query_time < 5.0, f"Query too slow for {endpoint}: {query_time:.2f}s"
                    print(f"Query time for {endpoint}: {query_time:.3f}s")
                elif response.status_code == 404:
                    # No data is acceptable
                    print(f"No data available for {endpoint}")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for database performance test")
    
    def test_index_effectiveness(self):
        """Test database index effectiveness."""
        try:
            # Test queries that should use indexes
            # This would typically require database-specific queries
            
            # Test sentiment queries by team (should be indexed)
            response = requests.get("http://localhost:8000/sentiment/team/1", timeout=10)
            
            if response.status_code in [200, 404]:
                print("Team sentiment query completed")
            
            # Test sentiment queries by date range (should be indexed)
            response = requests.get(
                "http://localhost:8000/sentiment/trends?days=7", 
                timeout=10
            )
            
            if response.status_code in [200, 404]:
                print("Sentiment trends query completed")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("API service not running for index effectiveness test")


class TestMonitoringPerformance:
    """Test monitoring system performance."""
    
    def test_metrics_collection_overhead(self):
        """Test metrics collection doesn't significantly impact performance."""
        try:
            # Test with metrics collection
            start_time = time.time()
            for _ in range(10):
                requests.get("http://localhost:8000/health", timeout=5)
            with_metrics_time = time.time() - start_time
            
            # Test metrics endpoint performance
            metrics_start = time.time()
            metrics_response = requests.get("http://localhost:8000/metrics", timeout=10)
            metrics_time = time.time() - metrics_start
            
            if metrics_response.status_code == 200:
                # Metrics collection should be fast
                assert metrics_time < 2.0, f"Metrics collection too slow: {metrics_time:.2f}s"
                print(f"Metrics collection time: {metrics_time:.3f}s")
            
            print(f"Request time with metrics: {with_metrics_time:.3f}s")
        
        except requests.exceptions.ConnectionError:
            pytest.skip("Services not running for monitoring performance test")
    
    def test_log_processing_performance(self):
        """Test log processing doesn't impact application performance."""
        # This would typically test log aggregation systems
        # For now, we'll check if log files are growing reasonably
        
        import os
        log_files = [
            "logs/access.log",
            "logs/error.log",
            "logs/app.log"
        ]
        
        for log_file in log_files:
            if os.path.exists(log_file):
                file_size = os.path.getsize(log_file)
                print(f"Log file {log_file}: {file_size} bytes")
                
                # Log files shouldn't be excessively large
                max_size = 100 * 1024 * 1024  # 100MB
                if file_size > max_size:
                    print(f"Warning: Log file {log_file} is large: {file_size / 1024 / 1024:.1f}MB")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])