# Data Ingestion Service Tests

This document describes the comprehensive test suite implemented for the data ingestion service as part of task 4.3.

## Overview

The test suite covers all aspects of the data ingestion service including unit tests, integration tests, and performance tests. The tests ensure reliable data collection, processing, rate limiting, and error handling.

## Test Files Created

### 1. `test_data_ingestion_unit.py`
**Purpose**: Unit tests for individual methods and components without external dependencies.

**Test Classes**:
- `TestRateLimiter`: Tests the rate limiting functionality
- `TestRawDataItem`: Tests the data structure for raw data items
- `TestDataIngestionService`: Tests core service functionality

**Key Test Areas**:
- Rate limiter initialization and behavior
- Service lifecycle (start/stop)
- Twitter data collection and processing
- ESPN data collection (scoreboard and news)
- Betting lines collection (DraftKings, MGM)
- Raw data processing and storage
- Duplicate detection
- Error handling

**Sample Tests**:
```python
def test_rate_limiter_initialization(self):
    """Test rate limiter initializes correctly."""
    limiter = RateLimiter(max_requests=100, time_window=60)
    assert limiter.max_requests == 100
    assert limiter.time_window == 60

async def test_service_initialization(self):
    """Test service initializes correctly."""
    assert self.service.session is None
    assert self.service.is_running is False
    assert len(self.service.rate_limiters) == 4
```

### 2. `test_data_ingestion_integration.py`
**Purpose**: Integration tests for end-to-end data pipeline with external API mocks.

**Test Classes**:
- `TestDataIngestionServiceIntegration`: End-to-end pipeline tests
- `TestGlobalServiceInstance`: Tests for the global service instance

**Key Test Areas**:
- Full Twitter data collection and processing pipeline
- Full ESPN data collection and processing pipeline
- Full betting lines collection and processing pipeline
- Mixed data sources processing
- Duplicate handling across the pipeline
- Error recovery and resilience
- Concurrent data collection from multiple sources

**Sample Tests**:
```python
async def test_full_twitter_pipeline(self):
    """Test complete Twitter data collection and processing pipeline."""
    # Mock Twitter API response with realistic data
    # Test data collection, processing, and storage
    # Verify statistics and data integrity

async def test_mixed_data_sources_pipeline(self):
    """Test processing data from multiple sources simultaneously."""
    # Create mixed raw data items from different sources
    # Process all data types in one batch
    # Verify correct handling of each data type
```

### 3. `test_data_ingestion_performance.py`
**Purpose**: Performance tests for rate limiting, error handling, and performance under load.

**Test Classes**:
- `TestRateLimiterPerformance`: Rate limiter performance and behavior under load
- `TestDataIngestionServicePerformance`: Service performance under various conditions
- `TestErrorHandlingPerformance`: Error handling performance and resilience

**Key Test Areas**:
- Rate limiter timing accuracy and concurrent request handling
- Large dataset processing performance
- Memory usage with large datasets
- Concurrent API calls performance
- Error recovery performance
- High error rate handling

**Sample Tests**:
```python
async def test_rate_limiter_timing_accuracy(self):
    """Test that rate limiter timing is accurate."""
    # Test precise timing of rate limiting
    # Verify delays are within expected ranges

async def test_batch_data_processing_performance(self):
    """Test batch data processing performance."""
    # Process 100 mixed data items
    # Verify processing completes within time limits
    # Check memory usage and performance metrics
```

## Test Coverage

### Core Functionality Tested
✅ **Rate Limiting**: All rate limiter functionality including timing, cleanup, and concurrent requests  
✅ **Data Collection**: Twitter, ESPN, and betting APIs with proper mocking  
✅ **Data Processing**: Raw data processing, validation, and storage  
✅ **Error Handling**: API failures, network errors, database errors  
✅ **Duplicate Detection**: Efficient duplicate checking across data types  
✅ **Service Lifecycle**: Start/stop operations and session management  

### External API Mocking
✅ **Twitter API v2**: Complete response structure with users and tweets  
✅ **ESPN API**: Scoreboard and news endpoints with realistic data  
✅ **Betting APIs**: Mock implementations for DraftKings and MGM  
✅ **Database Operations**: MongoDB collection operations  
✅ **HTTP Client**: aiohttp session mocking with proper async context managers  

### Performance Testing
✅ **Rate Limiting Performance**: Timing accuracy and concurrent handling  
✅ **Large Dataset Processing**: 100+ items with performance benchmarks  
✅ **Memory Usage**: Memory leak detection and usage monitoring  
✅ **Error Recovery**: Performance under high error rates  
✅ **Concurrent Operations**: Multiple API calls simultaneously  

## Requirements Satisfied

The test suite satisfies all requirements from task 4.3:

### ✅ Create unit tests for data collection and processing methods
- Comprehensive unit tests for all data collection methods
- Individual method testing with proper isolation
- Mock external dependencies for reliable testing

### ✅ Mock external API responses for reliable testing
- Twitter API v2 responses with complete data structures
- ESPN API responses for both scoreboard and news
- Betting API responses with realistic odds data
- Proper async context manager mocking for HTTP clients

### ✅ Test rate limiting and error handling for API failures
- Rate limiter accuracy and behavior under load
- API timeout and network error handling
- Database error recovery and resilience
- High error rate performance testing

### ✅ Implement integration tests for end-to-end data pipeline
- Complete pipeline tests from data collection to storage
- Mixed data source processing
- Duplicate handling across the pipeline
- Concurrent data collection and processing

## Running the Tests

### Run All Data Ingestion Tests
```bash
python -m pytest tests/test_data_ingestion_*.py -v
```

### Run Specific Test Categories
```bash
# Unit tests only
python -m pytest tests/test_data_ingestion_unit.py -v

# Integration tests only
python -m pytest tests/test_data_ingestion_integration.py -v

# Performance tests only
python -m pytest tests/test_data_ingestion_performance.py -v
```

### Run Specific Test Classes
```bash
# Rate limiter tests
python -m pytest tests/test_data_ingestion_unit.py::TestRateLimiter -v

# Service initialization tests
python -m pytest tests/test_data_ingestion_unit.py::TestDataIngestionService::test_service_initialization -v
```

## Test Environment Setup

The tests require the following environment variables to be set:
```bash
SECRET_KEY=test-secret-key
MONGODB_URL=mongodb://localhost:27017/test
DATABASE_NAME=test_db
REDIS_URL=redis://localhost:6379/1
TWITTER_BEARER_TOKEN=test_token
```

These are automatically set in the test files to ensure consistent test execution.

## Notes and Limitations

1. **Async Context Manager Mocking**: Some complex async HTTP mocking scenarios may require additional setup for full integration testing.

2. **Database Integration**: Tests use mocked database operations. For full integration testing, consider using test containers with real MongoDB instances.

3. **External API Rate Limits**: Tests use mocked responses to avoid hitting real API rate limits during testing.

4. **Performance Benchmarks**: Performance test thresholds are set conservatively and may need adjustment based on deployment environment.

## Future Enhancements

1. **Real API Integration Tests**: Add optional tests that hit real APIs with test credentials
2. **Database Integration Tests**: Add tests with real MongoDB test containers
3. **Load Testing**: Add more comprehensive load testing scenarios
4. **Monitoring Integration**: Add tests for monitoring and alerting functionality

The test suite provides comprehensive coverage of the data ingestion service functionality and ensures reliable operation under various conditions.