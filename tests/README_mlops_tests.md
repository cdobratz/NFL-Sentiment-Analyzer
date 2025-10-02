# MLOps Testing and Validation Suite

This directory contains comprehensive tests for the MLOps pipeline components, including model validation, deployment procedures, data drift detection, and performance monitoring.

## Test Structure

### Test Files

- **`test_mlops_validation.py`** - Core model validation tests with benchmark datasets
- **`test_mlops_deployment.py`** - Model deployment and rollback procedure tests
- **`test_mlops_data_drift.py`** - Data drift detection and performance monitoring tests
- **`test_mlops_integration.py`** - End-to-end MLOps pipeline integration tests
- **`test_mlops_config.py`** - Test configuration and utilities
- **`run_mlops_tests.py`** - Test runner script for executing MLOps tests

### Test Categories

#### 1. Model Validation (`TestModelValidation`)
- **Accuracy Validation**: Tests model performance against benchmark datasets
- **Performance Metrics**: Validates latency, throughput, and resource usage
- **Robustness Testing**: Tests model behavior with edge cases and adversarial inputs
- **Bias Detection**: Validates model fairness across different team contexts

#### 2. Model Deployment (`TestModelDeploymentStrategies`)
- **Deployment Strategies**: Tests immediate, blue-green, canary, and A/B test deployments
- **Rollback Procedures**: Validates automatic and manual rollback mechanisms
- **Health Monitoring**: Tests deployment health checks and status tracking
- **Configuration Validation**: Tests deployment configuration and resource limits

#### 3. Data Drift Detection (`TestDataDriftDetection`)
- **Statistical Tests**: Kolmogorov-Smirnov, Population Stability Index, Jensen-Shannon divergence
- **Performance Monitoring**: Tests performance threshold monitoring and alerting
- **Retraining Triggers**: Validates automated retraining trigger mechanisms
- **Trend Analysis**: Tests performance trend analysis and forecasting

#### 4. Integration Testing (`TestMLOpsEndToEndWorkflow`)
- **Complete Lifecycle**: Tests full model lifecycle from training to production
- **Service Integration**: Tests integration between MLOps components
- **Error Handling**: Tests error handling and resilience mechanisms
- **Feature Store Integration**: Tests feature store operations and data flow

## Running Tests

### Prerequisites

Install required dependencies:
```bash
pip install pytest pytest-asyncio pytest-cov numpy pandas scikit-learn
```

### Test Execution Options

#### 1. Run All Tests
```bash
python tests/run_mlops_tests.py --all
```

#### 2. Run Specific Category
```bash
python tests/run_mlops_tests.py --category model_validation
python tests/run_mlops_tests.py --category deployment
python tests/run_mlops_tests.py --category data_drift
python tests/run_mlops_tests.py --category integration
```

#### 3. Run Specific Tests
```bash
python tests/run_mlops_tests.py --tests tests/test_mlops_validation.py::TestModelValidation::test_validate_model_accuracy
```

#### 4. Validate Environment
```bash
python tests/run_mlops_tests.py --validate-env
```

#### 5. Verbose Output
```bash
python tests/run_mlops_tests.py --all --verbose
```

### Direct Pytest Execution

You can also run tests directly with pytest:

```bash
# Run all MLOps tests
pytest tests/test_mlops_*.py -v

# Run specific test file
pytest tests/test_mlops_validation.py -v

# Run with coverage
pytest tests/test_mlops_*.py --cov=app/services/mlops --cov-report=html

# Run tests with specific markers
pytest -m "model_validation" -v
pytest -m "deployment" -v
pytest -m "data_drift" -v
```

## Test Configuration

### Environment Variables

Set these environment variables for external service integration (optional):

```bash
export HUGGINGFACE_TOKEN="your_hf_token"
export WANDB_API_KEY="your_wandb_key"
export HOPSWORKS_API_KEY="your_hopsworks_key"
```

### Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.mlops` - General MLOps tests
- `@pytest.mark.model_validation` - Model validation tests
- `@pytest.mark.deployment` - Deployment tests
- `@pytest.mark.data_drift` - Data drift tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.external` - Tests requiring external services (skipped by default)

### Mock Services

Tests use mock services by default to avoid dependencies on external services:

- **HuggingFace Service**: Mocked model predictions and registrations
- **W&B Service**: Mocked experiment tracking
- **Hopsworks Service**: Mocked feature store operations
- **Deployment Service**: Mocked deployment operations

## Test Data

### Benchmark Datasets

The test suite includes several benchmark datasets:

- **Small Dataset**: 100 samples for quick validation
- **Medium Dataset**: 1,000 samples for comprehensive testing
- **Large Dataset**: 10,000 samples for performance testing

### Synthetic Data Generation

Tests generate synthetic data for:

- Performance metrics time series
- Data drift simulation
- Feature store data
- Sentiment analysis results

## Expected Test Results

### Model Validation Tests
- **Accuracy**: Models should achieve >80% accuracy on benchmark datasets
- **Latency**: Predictions should complete within 100ms
- **Robustness**: Models should handle edge cases without errors
- **Bias**: Models should show <10% bias across different team contexts

### Deployment Tests
- **Success Rate**: Deployments should succeed >95% of the time
- **Rollback Time**: Rollbacks should complete within 60 seconds
- **Health Checks**: Health monitoring should detect issues within 30 seconds

### Data Drift Tests
- **Detection Accuracy**: Drift detection should identify significant changes
- **False Positive Rate**: <5% false positives for normal data variations
- **Alert Generation**: Alerts should be generated for threshold violations

### Integration Tests
- **End-to-End Success**: Complete workflows should execute without errors
- **Service Communication**: All service integrations should work correctly
- **Error Recovery**: System should recover gracefully from failures

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Test Timeouts**: Some tests may take longer on slower systems
3. **Mock Service Issues**: Check that mock services are properly configured
4. **File Permissions**: Ensure test runner script is executable

### Debug Mode

Run tests with additional debugging:

```bash
pytest tests/test_mlops_validation.py -v -s --tb=long
```

### Test Isolation

Run tests in isolation to avoid interference:

```bash
pytest tests/test_mlops_validation.py::TestModelValidation::test_validate_model_accuracy --forked
```

## Continuous Integration

### GitHub Actions

Example workflow for CI:

```yaml
name: MLOps Tests
on: [push, pull_request]
jobs:
  mlops-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio pytest-cov
      - name: Run MLOps tests
        run: python tests/run_mlops_tests.py --all
```

### Test Reports

Test results are automatically saved to JSON files with timestamps:
- `mlops_test_results_YYYYMMDD_HHMMSS.json`

## Performance Benchmarks

### Expected Test Execution Times

- **Model Validation**: ~2-3 minutes
- **Deployment Tests**: ~1-2 minutes  
- **Data Drift Tests**: ~3-4 minutes
- **Integration Tests**: ~4-5 minutes
- **Total Suite**: ~10-15 minutes

### Resource Requirements

- **Memory**: ~2GB RAM for full test suite
- **CPU**: Multi-core recommended for parallel test execution
- **Disk**: ~1GB for test data and temporary files

## Contributing

### Adding New Tests

1. Follow the existing test structure and naming conventions
2. Use appropriate pytest markers
3. Include comprehensive docstrings
4. Add mock services for external dependencies
5. Update this README with new test descriptions

### Test Guidelines

- Tests should be deterministic and repeatable
- Use fixtures for common test data
- Mock external services to avoid dependencies
- Include both positive and negative test cases
- Test error conditions and edge cases

## Support

For issues with the MLOps testing suite:

1. Check the troubleshooting section above
2. Review test logs and error messages
3. Ensure all dependencies are properly installed
4. Verify test environment configuration

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [MLOps Best Practices](https://ml-ops.org/)
- [Model Validation Techniques](https://developers.google.com/machine-learning/testing-debugging/pipeline/deploying)