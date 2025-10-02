# API Documentation and Monitoring Tests

This directory contains comprehensive tests for API documentation accuracy, completeness, rate limiting, authentication mechanisms, load testing, and monitoring system validation.

## Overview

These tests validate the requirements specified in task 9.2:
- **API Documentation**: Test accuracy and completeness of OpenAPI/Swagger documentation
- **Rate Limiting**: Verify rate limiting mechanisms work correctly under various conditions
- **Authentication**: Validate JWT and API key authentication systems
- **Load Testing**: Test API performance under high traffic and concurrent load
- **Monitoring**: Validate health checks, metrics collection, and alerting systems

## Test Files

### `test_api_documentation_monitoring.py`
Main test suite covering:
- **TestAPIDocumentation**: OpenAPI schema generation, Swagger UI accessibility, endpoint documentation completeness
- **TestRateLimiting**: Rate limit enforcement, user-based quotas, API key limits
- **TestAuthentication**: JWT authentication, API key validation, security headers
- **TestLoadTesting**: Concurrent request handling, rate limiting under load
- **TestMonitoringSystem**: Health checks, metrics endpoints, correlation ID tracking
- **TestAPICompliance**: HTTP method compliance, content types, status codes

### `test_load_performance.py`
Dedicated load testing suite:
- **LoadTester**: Utility class for running concurrent and stress tests
- **TestLoadPerformance**: Performance tests under various load conditions
- **TestPerformanceBenchmarks**: Response time and resource usage benchmarks

### `test_monitoring_validation.py`
Comprehensive monitoring system validation:
- **TestHealthCheckValidation**: Health endpoint accuracy and performance
- **TestMetricsCollection**: Metrics collection and storage validation
- **TestAlertSystem**: Alert generation and handling
- **TestPerformanceMonitoring**: Performance monitoring and alerting
- **TestObservabilityFeatures**: Correlation IDs, structured logging
- **TestMonitoringIntegration**: End-to-end monitoring integration

## Running Tests

### Run All Tests
```bash
python tests/run_api_documentation_monitoring_tests.py
```

### Run Specific Test Categories
```bash
# API Documentation tests only
python tests/run_api_documentation_monitoring_tests.py docs

# Rate limiting tests only
python tests/run_api_documentation_monitoring_tests.py rate-limit

# Authentication tests only
python tests/run_api_documentation_monitoring_tests.py auth

# Load testing only
python tests/run_api_documentation_monitoring_tests.py load

# Monitoring validation only
python tests/run_api_documentation_monitoring_tests.py monitoring

# Quick validation for CI/CD
python tests/run_api_documentation_monitoring_tests.py quick
```

### Run Individual Test Files
```bash
# Run with pytest directly
pytest tests/test_api_documentation_monitoring.py -v
pytest tests/test_load_performance.py -v
pytest tests/test_monitoring_validation.py -v

# Run specific test classes
pytest tests/test_api_documentation_monitoring.py::TestAPIDocumentation -v
pytest tests/test_load_performance.py::TestLoadPerformance -v
pytest tests/test_monitoring_validation.py::TestHealthCheckValidation -v
```

## Test Categories

### 1. API Documentation Tests

**Purpose**: Validate that API documentation is accurate, complete, and accessible.

**Key Tests**:
- OpenAPI schema generation and structure
- Swagger UI and ReDoc accessibility
- Endpoint documentation completeness
- Security scheme documentation
- Error response examples

**Requirements Covered**: 6.2

### 2. Rate Limiting Tests

**Purpose**: Ensure rate limiting mechanisms work correctly and protect the API from abuse.

**Key Tests**:
- Rate limit enforcement within and beyond limits
- User-based and API key-based rate limits
- Multiple rate limit types (per minute/hour/day)
- Rate limit headers in responses
- Rate limiting under high load

**Requirements Covered**: 6.3

### 3. Authentication Tests

**Purpose**: Validate authentication mechanisms and security measures.

**Key Tests**:
- JWT token authentication
- API key authentication
- Authentication error handling
- Security headers presence
- CORS configuration

**Requirements Covered**: 6.3

### 4. Load Testing

**Purpose**: Test API performance under high traffic and concurrent load.

**Key Tests**:
- Concurrent request handling
- Response time consistency
- Memory usage stability
- Rate limiting under load
- Sustained load performance
- Resource usage monitoring

**Requirements Covered**: 6.4

### 5. Monitoring System Validation

**Purpose**: Ensure monitoring and alerting systems work correctly.

**Key Tests**:
- Health check endpoints accuracy
- Metrics collection and reporting
- Alert generation and handling
- Performance monitoring
- Observability features (correlation IDs, logging)
- System integration

**Requirements Covered**: 6.4

## Performance Benchmarks

### Response Time Benchmarks
- Health endpoints: < 50ms average
- Simple endpoints: < 100ms average
- Complex endpoints: < 2000ms average
- 95th percentile: < 1000ms for simple endpoints

### Load Testing Benchmarks
- Minimum 50 RPS for health endpoints
- 95% success rate under normal load
- 90% success rate under high load
- Memory increase < 100MB during load tests

### Rate Limiting Benchmarks
- Default: 100 requests per minute per IP
- User accounts: Higher limits based on role
- API keys: Custom limits per key
- Rate limit headers present in all responses

## Monitoring Validation

### Health Checks
- Basic health check: Always available, < 100ms
- Detailed health check: Component status, < 5s
- Liveness probe: Process health
- Readiness probe: Dependency health
- Metrics endpoint: Prometheus format

### Metrics Collection
- Request counters and response times
- Error rates and status codes
- System resource usage (CPU, memory, disk)
- Business metrics (sentiment analysis, user activity)

### Alerting
- Error rate alerts (> 5% error rate)
- Response time alerts (> 2s average)
- System resource alerts (> 80% CPU, > 85% memory)
- Custom business metric alerts

## Test Environment Setup

### Prerequisites
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx psutil

# Optional: Install coverage reporting
pip install pytest-cov
```

### Environment Variables
```bash
# Test configuration
export TESTING=true
export DATABASE_NAME=nfl_analyzer_test
export REDIS_URL=redis://localhost:6379/1
```

### Mock Services
Tests use mocked external services:
- MongoDB: Mocked database operations
- Redis: Mocked caching operations
- External APIs: Mocked responses

## Continuous Integration

### Quick Validation
For CI/CD pipelines, use the quick validation:
```bash
python tests/run_api_documentation_monitoring_tests.py quick
```

This runs essential tests that validate:
- OpenAPI schema generation
- Basic rate limiting
- Health check functionality

### Full Test Suite
For comprehensive validation:
```bash
python tests/run_api_documentation_monitoring_tests.py
```

## Troubleshooting

### Common Issues

1. **Redis Connection Errors**
   - Ensure Redis is running for integration tests
   - Use mocked Redis for unit tests

2. **MongoDB Connection Errors**
   - Ensure MongoDB is available for integration tests
   - Use mocked database for unit tests

3. **Load Test Failures**
   - Adjust concurrent user limits for test environment
   - Check system resources during tests

4. **Rate Limiting Test Failures**
   - Clear Redis cache between tests
   - Ensure rate limiting middleware is active

### Debug Mode
Run tests with verbose output:
```bash
pytest tests/test_api_documentation_monitoring.py -v -s
```

Add debugging to specific tests:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Test Coverage

Target coverage levels:
- API Documentation: 100% (all endpoints documented)
- Rate Limiting: 95% (core functionality)
- Authentication: 95% (security critical)
- Load Testing: 90% (performance validation)
- Monitoring: 95% (observability critical)

Generate coverage report:
```bash
pytest tests/test_api_documentation_monitoring.py --cov=app --cov-report=html
```

## Contributing

When adding new tests:

1. Follow existing test patterns and naming conventions
2. Add appropriate docstrings explaining test purpose
3. Include both positive and negative test cases
4. Mock external dependencies appropriately
5. Update this README with new test descriptions

### Test Naming Convention
- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Use descriptive names that explain what is being tested

### Assertion Guidelines
- Use specific assertions that clearly indicate what failed
- Include helpful error messages
- Test both success and failure scenarios
- Validate response formats and data types