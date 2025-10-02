"""
Load testing and performance validation for the NFL Sentiment Analyzer API.

This module provides comprehensive load testing capabilities including:
- Stress testing with high concurrent load
- Performance benchmarking
- Resource usage monitoring
- Scalability testing

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
import statistics
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
import httpx
from fastapi.testclient import TestClient
import psutil

from app.main import app


class LoadTestResult:
    """Container for load test results"""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.status_codes: List[int] = []
        self.errors: List[str] = []
        self.start_time: float = 0
        self.end_time: float = 0
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
    
    @property
    def total_requests(self) -> int:
        return len(self.response_times) + len(self.errors)
    
    @property
    def successful_requests(self) -> int:
        return len([code for code in self.status_codes if 200 <= code < 400])
    
    @property
    def failed_requests(self) -> int:
        return len([code for code in self.status_codes if code >= 400]) + len(self.errors)
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def avg_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)
    
    @property
    def p95_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.quantiles(self.response_times, n=20)[18]  # 95th percentile
    
    @property
    def p99_response_time(self) -> float:
        if not self.response_times:
            return 0.0
        return statistics.quantiles(self.response_times, n=100)[98]  # 99th percentile
    
    @property
    def requests_per_second(self) -> float:
        duration = self.end_time - self.start_time
        if duration == 0:
            return 0.0
        return self.total_requests / duration


class LoadTester:
    """Load testing utility for API endpoints"""
    
    def __init__(self, base_url: str = "http://test"):
        self.base_url = base_url
        self.process = psutil.Process(os.getpid())
    
    async def run_concurrent_requests(
        self,
        endpoint: str,
        num_requests: int,
        concurrent_users: int,
        headers: Dict[str, str] = None,
        method: str = "GET",
        payload: Dict[str, Any] = None
    ) -> LoadTestResult:
        """Run concurrent requests against an endpoint"""
        result = LoadTestResult()
        result.start_time = time.time()
        
        # Create semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def make_request(client: httpx.AsyncClient) -> Tuple[float, int, str]:
            """Make a single request and return timing and status"""
            async with semaphore:
                try:
                    start = time.time()
                    
                    if method.upper() == "GET":
                        response = await client.get(endpoint, headers=headers)
                    elif method.upper() == "POST":
                        response = await client.post(endpoint, json=payload, headers=headers)
                    else:
                        raise ValueError(f"Unsupported method: {method}")
                    
                    end = time.time()
                    return end - start, response.status_code, ""
                    
                except Exception as e:
                    return 0.0, 0, str(e)
        
        # Monitor system resources during test
        async def monitor_resources():
            """Monitor CPU and memory usage during test"""
            while result.end_time == 0:
                try:
                    cpu_percent = self.process.cpu_percent()
                    memory_info = self.process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    
                    result.cpu_usage.append(cpu_percent)
                    result.memory_usage.append(memory_mb)
                    
                    await asyncio.sleep(0.1)  # Monitor every 100ms
                except:
                    break
        
        # Start resource monitoring
        monitor_task = asyncio.create_task(monitor_resources())
        
        try:
            async with httpx.AsyncClient(app=app, base_url=self.base_url) as client:
                # Create all request tasks
                tasks = [make_request(client) for _ in range(num_requests)]
                
                # Execute all requests
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for res in results:
                    if isinstance(res, Exception):
                        result.errors.append(str(res))
                    else:
                        response_time, status_code, error = res
                        if error:
                            result.errors.append(error)
                        else:
                            result.response_times.append(response_time)
                            result.status_codes.append(status_code)
        
        finally:
            result.end_time = time.time()
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        return result
    
    async def stress_test(
        self,
        endpoint: str,
        duration_seconds: int,
        concurrent_users: int,
        headers: Dict[str, str] = None
    ) -> LoadTestResult:
        """Run stress test for specified duration"""
        result = LoadTestResult()
        result.start_time = time.time()
        end_time = result.start_time + duration_seconds
        
        semaphore = asyncio.Semaphore(concurrent_users)
        
        async def continuous_requests():
            """Make continuous requests until end time"""
            async with httpx.AsyncClient(app=app, base_url=self.base_url) as client:
                while time.time() < end_time:
                    async with semaphore:
                        try:
                            start = time.time()
                            response = await client.get(endpoint, headers=headers)
                            end = time.time()
                            
                            result.response_times.append(end - start)
                            result.status_codes.append(response.status_code)
                            
                        except Exception as e:
                            result.errors.append(str(e))
        
        # Start continuous requests
        await continuous_requests()
        result.end_time = time.time()
        
        return result


class TestLoadPerformance:
    """Load testing and performance validation tests"""
    
    @pytest.fixture
    def load_tester(self):
        """Create load tester instance"""
        return LoadTester()
    
    @pytest.mark.asyncio
    async def test_health_endpoint_load(self, load_tester):
        """Test health endpoint under load"""
        result = await load_tester.run_concurrent_requests(
            endpoint="/health/",
            num_requests=100,
            concurrent_users=10
        )
        
        # Assertions for performance
        assert result.success_rate >= 0.95  # 95% success rate
        assert result.avg_response_time < 0.5  # Average response time under 500ms
        assert result.p95_response_time < 1.0  # 95th percentile under 1 second
        assert result.requests_per_second > 50  # At least 50 RPS
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis_load(self, load_tester):
        """Test sentiment analysis endpoint under load"""
        payload = {
            "text": "This team is playing great today!",
            "context": {"team_id": "test_team"}
        }
        
        result = await load_tester.run_concurrent_requests(
            endpoint="/sentiment/analyze",
            num_requests=50,
            concurrent_users=5,
            method="POST",
            payload=payload
        )
        
        # More lenient requirements for complex endpoint
        assert result.success_rate >= 0.90  # 90% success rate
        assert result.avg_response_time < 2.0  # Average response time under 2 seconds
        assert result.p95_response_time < 5.0  # 95th percentile under 5 seconds
    
    @pytest.mark.asyncio
    async def test_rate_limiting_under_load(self, load_tester):
        """Test that rate limiting works correctly under high load"""
        # Make requests that should trigger rate limiting
        result = await load_tester.run_concurrent_requests(
            endpoint="/health/",
            num_requests=200,  # Exceed typical rate limits
            concurrent_users=20
        )
        
        # Should have some rate limited responses (429)
        rate_limited = len([code for code in result.status_codes if code == 429])
        assert rate_limited > 0, "Rate limiting should be triggered under high load"
        
        # But some requests should still succeed
        successful = len([code for code in result.status_codes if 200 <= code < 300])
        assert successful > 0, "Some requests should still succeed"
    
    @pytest.mark.asyncio
    async def test_memory_stability_under_load(self, load_tester):
        """Test that memory usage remains stable under load"""
        initial_memory = load_tester.process.memory_info().rss / 1024 / 1024  # MB
        
        result = await load_tester.run_concurrent_requests(
            endpoint="/health/",
            num_requests=200,
            concurrent_users=20
        )
        
        final_memory = load_tester.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100, f"Memory increased by {memory_increase:.2f}MB"
        
        # Check for memory leaks by looking at peak usage
        if result.memory_usage:
            peak_memory = max(result.memory_usage)
            assert peak_memory - initial_memory < 150, "Peak memory usage too high"
    
    @pytest.mark.asyncio
    async def test_response_time_consistency(self, load_tester):
        """Test response time consistency under varying load"""
        # Test with different load levels
        load_levels = [1, 5, 10, 20]
        results = []
        
        for concurrent_users in load_levels:
            result = await load_tester.run_concurrent_requests(
                endpoint="/health/",
                num_requests=50,
                concurrent_users=concurrent_users
            )
            results.append((concurrent_users, result.avg_response_time))
        
        # Response times should not increase dramatically with load
        response_times = [rt for _, rt in results]
        
        # The highest response time should not be more than 5x the lowest
        min_time = min(response_times)
        max_time = max(response_times)
        
        assert max_time / min_time < 5.0, "Response time degradation too severe under load"
    
    @pytest.mark.asyncio
    async def test_concurrent_different_endpoints(self, load_tester):
        """Test performance when hitting different endpoints concurrently"""
        endpoints = ["/health/", "/health/live", "/health/ready", "/"]
        
        async def test_endpoint(endpoint: str) -> LoadTestResult:
            return await load_tester.run_concurrent_requests(
                endpoint=endpoint,
                num_requests=25,
                concurrent_users=5
            )
        
        # Test all endpoints concurrently
        tasks = [test_endpoint(endpoint) for endpoint in endpoints]
        results = await asyncio.gather(*tasks)
        
        # All endpoints should perform reasonably
        for i, result in enumerate(results):
            endpoint = endpoints[i]
            assert result.success_rate >= 0.90, f"Endpoint {endpoint} success rate too low"
            assert result.avg_response_time < 1.0, f"Endpoint {endpoint} response time too high"
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, load_tester):
        """Test performance under sustained load"""
        # Run for 30 seconds with moderate load
        result = await load_tester.stress_test(
            endpoint="/health/",
            duration_seconds=10,  # Reduced for testing
            concurrent_users=5
        )
        
        assert result.success_rate >= 0.95
        assert result.avg_response_time < 0.5
        assert result.requests_per_second > 20
        
        # Check that performance didn't degrade over time
        if len(result.response_times) > 10:
            first_half = result.response_times[:len(result.response_times)//2]
            second_half = result.response_times[len(result.response_times)//2:]
            
            avg_first = statistics.mean(first_half)
            avg_second = statistics.mean(second_half)
            
            # Second half should not be significantly slower
            assert avg_second / avg_first < 2.0, "Performance degraded over time"
    
    def test_synchronous_load_with_test_client(self):
        """Test load using synchronous TestClient for comparison"""
        client = TestClient(app)
        response_times = []
        
        start_time = time.time()
        
        # Make 50 sequential requests
        for _ in range(50):
            request_start = time.time()
            response = client.get("/health/")
            request_end = time.time()
            
            assert response.status_code == 200
            response_times.append(request_end - request_start)
        
        end_time = time.time()
        
        # Calculate metrics
        avg_response_time = statistics.mean(response_times)
        total_time = end_time - start_time
        rps = 50 / total_time
        
        # Performance assertions
        assert avg_response_time < 0.1  # Should be fast for sequential requests
        assert rps > 100  # Should handle at least 100 RPS sequentially
    
    @pytest.mark.asyncio
    async def test_error_handling_under_load(self, load_tester):
        """Test error handling when system is under load"""
        # First, put system under load
        load_task = asyncio.create_task(
            load_tester.run_concurrent_requests(
                endpoint="/health/",
                num_requests=100,
                concurrent_users=10
            )
        )
        
        # While under load, test error scenarios
        await asyncio.sleep(0.1)  # Let load start
        
        error_result = await load_tester.run_concurrent_requests(
            endpoint="/nonexistent-endpoint",
            num_requests=10,
            concurrent_users=2
        )
        
        # Wait for load test to complete
        load_result = await load_task
        
        # Error responses should still be fast and consistent
        assert error_result.avg_response_time < 0.5
        assert all(code == 404 for code in error_result.status_codes)
        
        # Load test should still perform well
        assert load_result.success_rate >= 0.90


class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    def test_response_time_benchmarks(self):
        """Test that response times meet benchmark requirements"""
        client = TestClient(app)
        
        benchmarks = {
            "/health/": 0.05,  # 50ms
            "/health/live": 0.05,  # 50ms
            "/": 0.1,  # 100ms
        }
        
        for endpoint, max_time in benchmarks.items():
            times = []
            for _ in range(10):
                start = time.time()
                response = client.get(endpoint)
                end = time.time()
                
                assert response.status_code == 200
                times.append(end - start)
            
            avg_time = statistics.mean(times)
            assert avg_time < max_time, f"Endpoint {endpoint} too slow: {avg_time:.3f}s > {max_time}s"
    
    def test_memory_usage_benchmarks(self):
        """Test memory usage benchmarks"""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        client = TestClient(app)
        
        # Make many requests to test memory stability
        for _ in range(100):
            response = client.get("/health/")
            assert response.status_code == 200
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal for simple requests
        assert memory_increase < 10, f"Memory increased by {memory_increase:.2f}MB"
    
    def test_cpu_usage_efficiency(self):
        """Test CPU usage efficiency"""
        process = psutil.Process(os.getpid())
        
        # Measure CPU usage during requests
        cpu_before = process.cpu_percent()
        
        client = TestClient(app)
        start_time = time.time()
        
        # Make requests for 1 second
        request_count = 0
        while time.time() - start_time < 1.0:
            response = client.get("/health/")
            assert response.status_code == 200
            request_count += 1
        
        cpu_after = process.cpu_percent()
        
        # Should be able to handle reasonable number of requests
        assert request_count > 50, f"Only handled {request_count} requests per second"
        
        # CPU usage should be reasonable (this is approximate)
        # Note: CPU measurement in tests can be unreliable
        if cpu_after > 0:  # Only check if we got a measurement
            assert cpu_after < 90, f"CPU usage too high: {cpu_after}%"