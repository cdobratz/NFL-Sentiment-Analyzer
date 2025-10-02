# Deployment and Infrastructure Tests

This directory contains comprehensive tests for validating the deployment and infrastructure setup of the NFL Sentiment Analyzer application.

## Overview

The deployment tests ensure that:
- Docker containers build correctly and follow best practices
- Multi-service orchestration works properly
- Production configuration is secure and optimized
- Application health and basic functionality work after deployment
- Security configurations are properly implemented
- Performance meets acceptable standards

## Test Suites

### 1. Docker Build Tests (`test_docker_builds.py`)

Tests Docker container builds and image optimization:

- **Dockerfile Best Practices**: Validates multi-stage builds, non-root users, health checks
- **Build Optimization**: Tests layer caching, image size, build context optimization
- **Security**: Validates base image security, user permissions, secrets handling
- **Performance**: Tests build caching effectiveness and requirements optimization

**Key Tests:**
- `test_dockerfile_best_practices()` - Validates Dockerfile follows best practices
- `test_image_layer_optimization()` - Ensures optimal layer structure
- `test_build_caching()` - Validates build cache effectiveness
- `test_user_security()` - Ensures containers run as non-root

### 2. Deployment Infrastructure Tests (`test_deployment_infrastructure.py`)

Tests multi-service orchestration and configuration:

- **Container Builds**: Validates all Docker images build successfully
- **Service Orchestration**: Tests Docker Compose configuration and service dependencies
- **Production Configuration**: Validates environment variables and security settings
- **Network Configuration**: Tests service communication and network setup
- **Volume Management**: Validates data persistence and volume configuration

**Key Tests:**
- `test_docker_compose_file_validation()` - Validates Compose file syntax
- `test_service_dependencies()` - Tests service dependency configuration
- `test_production_optimizations()` - Validates production-specific settings
- `test_volume_configuration()` - Tests data persistence setup

### 3. Smoke Tests (`test_smoke_tests.py`)

Basic health and functionality tests for deployed application:

- **Service Health**: Tests all service health endpoints
- **API Functionality**: Validates basic API operations
- **Database Connectivity**: Tests database connections through API
- **Network Connectivity**: Validates service-to-service communication
- **Data Integrity**: Tests basic data consistency

**Key Tests:**
- `test_service_health_endpoints()` - Tests all service health checks
- `test_api_basic_functionality()` - Validates core API functionality
- `test_database_connectivity()` - Tests database connection through API
- `test_concurrent_requests()` - Tests application under concurrent load

### 4. Security Validation Tests (`test_security_validation.py`)

Comprehensive security tests for production deployment:

- **Container Security**: Tests container security configurations
- **Network Security**: Validates network security settings
- **Configuration Security**: Tests secure configuration management
- **Database Security**: Validates database security settings
- **Application Security**: Tests application-level security measures

**Key Tests:**
- `test_container_runs_as_non_root()` - Ensures containers use non-root users
- `test_secrets_management()` - Validates secrets are properly managed
- `test_ssl_configuration()` - Tests SSL/TLS configuration
- `test_security_headers()` - Validates security headers are present

### 5. Performance Validation Tests (`test_performance_validation.py`)

Performance and scalability validation:

- **Load Testing**: Tests application under expected load
- **Resource Utilization**: Monitors CPU, memory, and disk usage
- **Scalability**: Tests horizontal scaling readiness
- **Database Performance**: Validates database query performance
- **Monitoring Performance**: Tests monitoring system overhead

**Key Tests:**
- `test_api_load_capacity()` - Tests API under concurrent load
- `test_memory_usage()` - Monitors container memory usage
- `test_horizontal_scaling_readiness()` - Tests stateless design
- `test_database_connection_pooling()` - Validates connection pooling

## Running the Tests

### Prerequisites

Before running the tests, ensure you have:

1. **Docker and Docker Compose installed**
2. **Python 3.11+ with required dependencies**
3. **Application built and optionally running**

### Quick Start

Run all deployment tests:
```bash
python tests/run_deployment_tests.py
```

Run with verbose output:
```bash
python tests/run_deployment_tests.py --verbose
```

Run specific test suite:
```bash
python tests/run_deployment_tests.py --test docker_builds
```

### Individual Test Suites

Run individual test suites using pytest:

```bash
# Docker build tests
pytest tests/test_docker_builds.py -v

# Infrastructure tests
pytest tests/test_deployment_infrastructure.py -v

# Smoke tests (requires running services)
pytest tests/test_smoke_tests.py -v

# Security validation
pytest tests/test_security_validation.py -v

# Performance validation (requires running services)
pytest tests/test_performance_validation.py -v
```

### Test Runner Options

The `run_deployment_tests.py` script supports several options:

- `--verbose, -v`: Verbose output with detailed test results
- `--fail-fast, -x`: Stop execution on first failure
- `--test TEST, -t TEST`: Run specific test suite
- `--report FILE, -r FILE`: Generate JSON report (default: deployment_test_report.json)

## Test Categories

### Required Tests
These tests must pass for production deployment:
- Docker build tests
- Infrastructure configuration tests
- Basic smoke tests
- Security validation tests

### Optional Tests
These tests provide additional validation but don't block deployment:
- Performance validation tests
- Advanced monitoring tests

## Environment Setup

### For Build Tests
No running services required - tests build containers from scratch.

### For Smoke and Performance Tests
Services should be running:

```bash
# Start services
docker-compose up -d

# Wait for services to be ready
sleep 30

# Run tests
python tests/run_deployment_tests.py
```

### Test Environment Variables

Set these environment variables for comprehensive testing:

```bash
export SECRET_KEY="test-secret-key"
export MONGODB_URL="mongodb://localhost:27017"
export REDIS_URL="redis://localhost:6379"
export DEBUG="true"
export ENVIRONMENT="test"
```

## Interpreting Results

### Success Criteria

**All Required Tests Pass**: Application is ready for production deployment.

**Some Optional Tests Fail**: Consider investigating but doesn't block deployment.

**Required Tests Fail**: Fix issues before deploying to production.

### Common Issues

1. **Docker Build Failures**
   - Check Dockerfile syntax
   - Verify build context and .dockerignore
   - Ensure base images are available

2. **Service Health Failures**
   - Verify services are running
   - Check service dependencies
   - Validate environment variables

3. **Security Test Failures**
   - Review security configurations
   - Check secrets management
   - Validate SSL/TLS setup

4. **Performance Test Failures**
   - Check resource limits
   - Verify database indexing
   - Review caching configuration

## Continuous Integration

### GitHub Actions Example

```yaml
name: Deployment Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  deployment-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    
    - name: Run deployment tests
      run: |
        python tests/run_deployment_tests.py --verbose --report deployment_report.json
    
    - name: Upload test report
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: deployment-test-report
        path: deployment_report.json
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        
        stage('Build Images') {
            steps {
                sh 'docker-compose build'
            }
        }
        
        stage('Deployment Tests') {
            steps {
                sh 'python tests/run_deployment_tests.py --verbose --report deployment_report.json'
            }
            post {
                always {
                    archiveArtifacts artifacts: 'deployment_report.json', fingerprint: true
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: '.',
                        reportFiles: 'deployment_report.json',
                        reportName: 'Deployment Test Report'
                    ])
                }
            }
        }
    }
}
```

## Extending the Tests

### Adding New Test Suites

1. Create new test file in `tests/` directory
2. Follow naming convention: `test_[category]_[description].py`
3. Add test suite to `run_deployment_tests.py`
4. Update this README with test documentation

### Custom Test Configuration

Create `pytest.ini` for custom pytest configuration:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --disable-warnings
    --tb=short
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    security: marks tests as security tests
```

## Troubleshooting

### Common Test Failures

1. **Docker not available**: Install Docker and ensure it's running
2. **Services not responding**: Check if services are started and healthy
3. **Permission errors**: Ensure proper file permissions and user access
4. **Network connectivity**: Verify service networking and port exposure
5. **Resource constraints**: Check available system resources (CPU, memory, disk)

### Debug Mode

Run tests with maximum verbosity:
```bash
pytest tests/test_deployment_infrastructure.py -vvv --tb=long --capture=no
```

### Test Isolation

Run tests in isolation to avoid interference:
```bash
pytest tests/test_docker_builds.py --forked
```

## Security Considerations

- Tests may create temporary containers and networks
- Sensitive information should not be logged or exposed
- Test credentials should be different from production
- Clean up test resources after execution
- Validate that tests don't expose security vulnerabilities

## Performance Considerations

- Some tests may be resource-intensive
- Performance tests require adequate system resources
- Consider running performance tests separately in CI/CD
- Monitor test execution time and optimize as needed

## Maintenance

- Review and update tests regularly
- Keep test dependencies up to date
- Monitor test reliability and flakiness
- Update documentation when adding new tests
- Validate tests against new Docker and infrastructure versions