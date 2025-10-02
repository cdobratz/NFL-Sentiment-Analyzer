# Sentiment Analysis Testing Suite

This directory contains comprehensive tests for the NFL sentiment analysis system, covering unit tests, integration tests, and performance tests as specified in task 3.3.

## Test Files Overview

### 1. `test_sentiment_unit.py`
**Unit tests for sentiment analysis algorithms and scoring**

- **TestNFLSentimentConfig**: Tests NFL-specific configuration and keyword management
  - Configuration initialization and keyword aggregation
  - Sentiment weight calculations for different keyword types
  - Text categorization based on NFL-specific content

- **TestSentimentAnalysisService**: Tests core sentiment analysis service functionality
  - Text preprocessing and NFL context extraction
  - Sentiment scoring algorithms with positive/negative/neutral detection
  - NFL-specific adjustments for injury and trade contexts
  - Emotion scores and aspect sentiment calculations
  - Batch processing capabilities

- **TestNFLSentimentEngine**: Tests enhanced NFL sentiment analysis engine
  - Enhanced context extraction with team, player, and game information
  - Confidence score calculation with NFL-specific factors
  - Performance metrics tracking
  - Batch processing with concurrency limits
  - NFL-specific aspect sentiment analysis

### 2. `test_sentiment_integration.py`
**Integration tests for sentiment API endpoints**

- **TestSentimentAnalysisEndpoint**: Tests `/sentiment/analyze` endpoint
  - Successful sentiment analysis with database storage
  - Error handling for service failures and invalid requests
  - Authentication and authorization integration

- **TestBatchSentimentAnalysisEndpoint**: Tests `/sentiment/analyze/batch` endpoint
  - Batch processing with multiple texts
  - Validation of batch size limits
  - Database operations for batch results

- **TestTeamSentimentEndpoint**: Tests `/sentiment/team/{team_id}` endpoint
  - Team sentiment aggregation and trend analysis
  - Filtering by sources, categories, and time periods
  - Handling of non-existent teams and empty data

- **TestPlayerSentimentEndpoint**: Tests `/sentiment/player/{player_id}` endpoint
  - Player-specific sentiment analysis and aggregation
  - Position-based sentiment weighting

- **TestGameSentimentEndpoint**: Tests `/sentiment/game/{game_id}` endpoint
  - Game-specific sentiment analysis
  - Home/away team sentiment breakdown

- **TestSentimentTrendsEndpoint**: Tests `/sentiment/trends` endpoint
  - Time-based sentiment trend analysis
  - Configurable intervals and filtering options

- **TestSentimentAggregationEndpoint**: Tests `/sentiment/aggregate` endpoint
  - Custom aggregation queries with flexible parameters

- **TestBackgroundTaskIntegration**: Tests background task integration
- **TestErrorHandlingIntegration**: Tests comprehensive error handling

### 3. `test_sentiment_performance.py`
**Performance tests for batch processing capabilities**

- **TestSentimentAnalysisPerformance**: Tests core performance characteristics
  - Single text analysis performance (< 1 second)
  - Batch processing performance for different sizes (10, 50, 100 texts)
  - Text length impact on processing time
  - Concurrent analysis performance
  - Memory usage efficiency
  - Keyword processing performance impact

- **TestNFLSentimentEnginePerformance**: Tests enhanced engine performance
  - Enhanced analysis with detailed breakdowns
  - Batch processing with concurrency limits
  - Context extraction performance
  - Confidence calculation performance
  - Performance metrics tracking
  - Large batch memory efficiency

- **TestPerformanceBenchmarks**: Benchmark tests for regression detection
  - Throughput benchmarks for different batch sizes
  - Latency percentiles for SLA validation (P50 < 500ms, P95 < 1s, P99 < 2s)
  - Scalability testing with increasing concurrent load

## Key Features Tested

### NFL-Specific Context Processing
- Team mention extraction and recognition
- Player position identification
- Injury and trade context detection
- Game situation analysis (playoff, primetime, rivalry, clutch)
- Performance metrics extraction

### Keyword Weighting System
- Positive performance keywords (touchdown, amazing, elite, etc.)
- Negative performance keywords (fumble, terrible, bust, etc.)
- Injury-related keywords with appropriate sentiment adjustment
- Trade and coaching keywords with neutral-to-mixed sentiment
- Betting and fantasy keywords with specialized weighting

### Batch Processing Capabilities
- Concurrent processing with configurable limits
- Memory-efficient processing of large batches
- Error handling and graceful degradation
- Performance optimization for high-throughput scenarios

### API Integration
- Complete endpoint coverage with realistic scenarios
- Database integration with MongoDB operations
- Authentication and authorization testing
- Error handling and edge case coverage

## Performance Benchmarks

The tests validate the following performance requirements:

- **Single Analysis**: < 1 second per text
- **Batch Processing**: 
  - Small batches (10 texts): < 5 seconds
  - Medium batches (50 texts): < 15 seconds  
  - Large batches (100 texts): < 30 seconds
- **Memory Usage**: < 200MB increase for 100-text batches
- **Latency SLAs**:
  - P50: < 500ms
  - P95: < 1 second
  - P99: < 2 seconds

## Running the Tests

```bash
# Run all sentiment tests
python -m pytest tests/test_sentiment_*.py -v

# Run specific test categories
python -m pytest tests/test_sentiment_unit.py -v          # Unit tests
python -m pytest tests/test_sentiment_integration.py -v   # Integration tests
python -m pytest tests/test_sentiment_performance.py -v   # Performance tests

# Run with coverage
python -m pytest tests/test_sentiment_*.py --cov=app.services.sentiment_service --cov=app.services.nfl_sentiment_engine --cov=app.api.sentiment
```

## Test Coverage

The test suite provides comprehensive coverage of:

- ✅ NFL-specific keyword processing and weighting
- ✅ Context extraction and enhancement
- ✅ Sentiment scoring algorithms
- ✅ Batch processing capabilities
- ✅ API endpoint integration
- ✅ Database operations
- ✅ Error handling and edge cases
- ✅ Performance characteristics and benchmarks
- ✅ Memory efficiency and scalability

All tests are designed to be independent, fast-running, and provide clear feedback on system functionality and performance.