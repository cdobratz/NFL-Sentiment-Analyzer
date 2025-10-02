"""
Performance tests for data ingestion service.
Tests rate limiting, error handling, and performance under load.
"""
import pytest
import asyncio
import time
from unittest.mock import AsyncMock, patch, Mock
from datetime import datetime, timedelta
from typing import List, Dict, Any
import aiohttp

# Set test environment variables before importing
import os
os.environ.setdefault('SECRET_KEY', 'test-secret-key')
os.environ.setdefault('MONGODB_URL', 'mongodb://localhost:27017/test')
os.environ.setdefault('DATABASE_NAME', 'test_db')
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/1')
os.environ.setdefault('TWITTER_BEARER_TOKEN', 'test_token')

from app.services.data_ingestion_service import (
    DataIngestionService, DataSource, RawDataItem, RateLimiter
)


class TestRateLimiterPerformance:
    """Test rate limiter performance and behavior under load."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_timing_accuracy(self):
        """Test that rate limiter timing is accurate."""
        limiter = RateLimiter(max_requests=2, time_window=1)
        
        # Make requests up to limit
        start_time = time.time()
        await limiter.acquire()
        await limiter.acquire()
        
        # Third request should be delayed
        await limiter.acquire()
        end_time = time.time()
        
        # Should have waited approximately 1 second
        elapsed = end_time - start_time
        assert 0.8 <= elapsed <= 1.5  # Allow some tolerance for timing
    
    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_requests(self):
        """Test rate limiter with concurrent requests."""
        limiter = RateLimiter(max_requests=5, time_window=2)
        
        async def make_request():
            await limiter.acquire()
            return time.time()
        
        # Make 10 concurrent requests
        start_time = time.time()
        tasks = [make_request() for _ in range(10)]
        timestamps = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # First 5 should be immediate, next 5 should be delayed
        immediate_requests = sum(1 for ts in timestamps if ts - start_time < 0.1)
        delayed_requests = len(timestamps) - immediate_requests
        
        assert immediate_requests <= 5
        assert delayed_requests >= 5
        assert end_time - start_time >= 1.5  # Should take at least 1.5 seconds
    
    @pytest.mark.asyncio
    async def test_rate_limiter_burst_handling(self):
        """Test rate limiter handling burst requests."""
        limiter = RateLimiter(max_requests=3, time_window=1)
        
        # First burst - should be immediate
        start_time = time.time()
        for _ in range(3):
            await limiter.acquire()
        first_burst_time = time.time() - start_time
        
        # Second burst - should be delayed
        start_time = time.time()
        for _ in range(3):
            await limiter.acquire()
        second_burst_time = time.time() - start_time
        
        assert first_burst_time < 0.1  # First burst immediate
        assert second_burst_time >= 0.8  # Second burst delayed
    
    @pytest.mark.asyncio
    async def test_rate_limiter_cleanup_performance(self):
        """Test rate limiter cleanup performance with many old requests."""
        limiter = RateLimiter(max_requests=100, time_window=1)
        
        # Add many old requests
        old_time = datetime.now() - timedelta(seconds=5)
        limiter.requests = [old_time] * 1000
        
        # New request should clean up old ones efficiently
        start_time = time.time()
        await limiter.acquire()
        cleanup_time = time.time() - start_time
        
        # Cleanup should be fast even with many old requests
        assert cleanup_time < 0.1
        assert len(limiter.requests) == 1  # Only new request remains


class TestDataIngestionServicePerformance:
    """Test data ingestion service performance under various conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataIngestionService()
    
    @pytest.mark.asyncio
    async def test_twitter_data_processing_performance(self):
        """Test Twitter data processing performance with large datasets."""
        # Create large mock response
        tweets = []
        users = []
        
        for i in range(100):  # 100 tweets
            tweets.append({
                "id": f"tweet_{i}",
                "text": f"This is test tweet number {i} about NFL",
                "created_at": "2024-01-15T20:00:00Z",
                "author_id": f"user_{i}",
                "public_metrics": {"like_count": i * 2},
                "lang": "en"
            })
            
            users.append({
                "id": f"user_{i}",
                "name": f"User {i}",
                "username": f"user{i}",
                "verified": i % 10 == 0,  # Every 10th user is verified
                "public_metrics": {"followers_count": i * 100}
            })
        
        mock_response_data = {
            "data": tweets,
            "includes": {"users": users}
        }
        
        # Measure processing time
        start_time = time.time()
        result = self.service._process_twitter_data(mock_response_data)
        processing_time = time.time() - start_time
        
        # Verify results
        assert len(result) == 100
        assert all(item.source == DataSource.TWITTER for item in result)
        
        # Performance should be reasonable (less than 1 second for 100 tweets)
        assert processing_time < 1.0
        
        # Verify data integrity
        for i, item in enumerate(result):
            assert item.content["id"] == f"tweet_{i}"
            assert item.content["author"]["name"] == f"User {i}"
    
    @pytest.mark.asyncio
    async def test_espn_data_processing_performance(self):
        """Test ESPN data processing performance with large datasets."""
        # Create large mock scoreboard response
        events = []
        for i in range(50):  # 50 games
            events.append({
                "id": f"game_{i}",
                "name": f"Team A{i} vs Team B{i}",
                "shortName": f"A{i} vs B{i}",
                "date": "2024-01-15T20:00:00Z",
                "status": {"type": {"completed": i % 2 == 0}},
                "competitions": [{"competitors": []}],
                "season": {"type": "regular"},
                "week": {"number": i % 18 + 1}
            })
        
        mock_response_data = {"events": events}
        
        # Measure processing time
        start_time = time.time()
        result = self.service._process_espn_scoreboard(mock_response_data)
        processing_time = time.time() - start_time
        
        # Verify results
        assert len(result) == 50
        assert all(item.source == DataSource.ESPN for item in result)
        
        # Performance should be reasonable
        assert processing_time < 0.5
    
    @pytest.mark.asyncio
    async def test_batch_data_processing_performance(self):
        """Test batch data processing performance."""
        # Create mixed batch of raw data items
        raw_items = []
        
        # Add tweets
        for i in range(50):
            raw_items.append(RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={
                    "id": f"tweet_{i}",
                    "text": f"Tweet {i}",
                    "created_at": "2024-01-15T20:00:00Z",
                    "author": {"name": f"User {i}"},
                    "metrics": {"like_count": i},
                    "context_annotations": []
                },
                timestamp=datetime.now(),
                metadata={"lang": "en"}
            ))
        
        # Add news articles
        for i in range(25):
            raw_items.append(RawDataItem(
                source=DataSource.ESPN,
                data_type="news",
                content={
                    "id": f"news_{i}",
                    "headline": f"News {i}",
                    "description": f"Description {i}",
                    "published": "2024-01-15T19:00:00Z"
                },
                timestamp=datetime.now(),
                metadata={"source": "espn_news"}
            ))
        
        # Add games
        for i in range(25):
            raw_items.append(RawDataItem(
                source=DataSource.ESPN,
                data_type="game",
                content={
                    "id": f"game_{i}",
                    "name": f"Game {i}",
                    "date": "2024-01-15T20:00:00Z",
                    "status": {"completed": False},
                    "competitions": []
                },
                timestamp=datetime.now(),
                metadata={"source": "espn_scoreboard"}
            ))
        
        # Mock database operations
        mock_db = AsyncMock()
        mock_db.raw_tweets.find_one.return_value = None
        mock_db.raw_news.find_one.return_value = None
        mock_db.raw_games.find_one.return_value = None
        mock_db.raw_tweets.insert_one.return_value = None
        mock_db.raw_news.insert_one.return_value = None
        mock_db.raw_games.insert_one.return_value = None
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            # Measure processing time
            start_time = time.time()
            stats = await self.service.process_raw_data(raw_items)
            processing_time = time.time() - start_time
            
            # Verify results
            assert stats["processed"] == 100
            assert stats["tweets"] == 50
            assert stats["news"] == 25
            assert stats["games"] == 25
            assert stats["errors"] == 0
            
            # Performance should be reasonable (less than 2 seconds for 100 items)
            assert processing_time < 2.0
    
    @pytest.mark.asyncio
    async def test_concurrent_api_calls_performance(self):
        """Test performance of concurrent API calls."""
        # Mock responses
        mock_twitter_response = {
            "data": [{"id": "123", "text": "test", "created_at": "2024-01-15T20:00:00Z", "author_id": "user1"}],
            "includes": {"users": [{"id": "user1", "name": "Test", "username": "test"}]}
        }
        
        mock_espn_scoreboard = {"events": []}
        mock_espn_news = {"articles": []}
        
        # Mock HTTP responses with delays to simulate network latency
        async def mock_get(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate 100ms network delay
            mock_response = AsyncMock()
            mock_response.status = 200
            
            url = args[0] if args else kwargs.get('url', '')
            if 'twitter.com' in url:
                mock_response.json.return_value = mock_twitter_response
            elif 'scoreboard' in url:
                mock_response.json.return_value = mock_espn_scoreboard
            else:
                mock_response.json.return_value = mock_espn_news
            
            return mock_response
        
        mock_session = AsyncMock()
        mock_session.get.side_effect = lambda *args, **kwargs: mock_get(*args, **kwargs).__aenter__()
        
        with patch('app.services.data_ingestion_service.settings') as mock_settings:
            mock_settings.twitter_bearer_token = "test_token"
            
            # Start service
            await self.service.start()
            self.service.session = mock_session
            
            try:
                # Test concurrent calls
                start_time = time.time()
                
                tasks = [
                    self.service.collect_twitter_data(["NFL"]),
                    self.service.fetch_espn_data(),
                    self.service.get_betting_lines(["draftkings"])
                ]
                
                results = await asyncio.gather(*tasks)
                concurrent_time = time.time() - start_time
                
                # Test sequential calls
                start_time = time.time()
                
                await self.service.collect_twitter_data(["NFL"])
                await self.service.fetch_espn_data()
                await self.service.get_betting_lines(["draftkings"])
                
                sequential_time = time.time() - start_time
                
                # Concurrent should be significantly faster
                assert concurrent_time < sequential_time * 0.7  # At least 30% faster
                
            finally:
                await self.service.stop()
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_large_datasets(self):
        """Test memory usage doesn't grow excessively with large datasets."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large amounts of data
        for batch in range(10):  # 10 batches
            raw_items = []
            
            # Create 100 items per batch
            for i in range(100):
                raw_items.append(RawDataItem(
                    source=DataSource.TWITTER,
                    data_type="tweet",
                    content={
                        "id": f"tweet_{batch}_{i}",
                        "text": f"Large tweet content {batch}_{i} " * 10,  # Make content larger
                        "created_at": "2024-01-15T20:00:00Z",
                        "author": {"name": f"User {i}"},
                        "metrics": {"like_count": i},
                        "context_annotations": []
                    },
                    timestamp=datetime.now(),
                    metadata={"lang": "en", "batch": batch}
                ))
            
            # Mock database
            mock_db = AsyncMock()
            mock_db.raw_tweets.find_one.return_value = None
            mock_db.raw_tweets.insert_one.return_value = None
            
            with patch('app.services.data_ingestion_service.get_database') as mock_get_db:
                mock_get_db.return_value = mock_db
                
                # Process batch
                await self.service.process_raw_data(raw_items)
            
            # Clear references to help garbage collection
            del raw_items
        
        # Check memory usage after processing
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
    
    @pytest.mark.asyncio
    async def test_error_recovery_performance(self):
        """Test performance of error recovery mechanisms."""
        # Create data that will cause some errors
        raw_items = []
        for i in range(50):
            raw_items.append(RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={"id": f"tweet_{i}", "text": f"Tweet {i}"},
                timestamp=datetime.now(),
                metadata={}
            ))
        
        # Mock database with intermittent errors
        mock_db = AsyncMock()
        mock_db.raw_tweets.find_one.return_value = None
        
        # Make every 5th insert fail
        def mock_insert(doc):
            if int(doc["unique_id"].split("_")[1]) % 5 == 0:
                raise Exception("Simulated database error")
            return None
        
        mock_db.raw_tweets.insert_one.side_effect = mock_insert
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            # Measure processing time with errors
            start_time = time.time()
            stats = await self.service.process_raw_data(raw_items)
            processing_time = time.time() - start_time
            
            # Verify error handling
            assert stats["processed"] == 40  # 10 errors out of 50
            assert stats["errors"] == 10
            
            # Processing should still be reasonably fast despite errors
            assert processing_time < 1.0
    
    @pytest.mark.asyncio
    async def test_duplicate_detection_performance(self):
        """Test performance of duplicate detection with large datasets."""
        # Create items with some duplicates
        raw_items = []
        for i in range(100):
            # Every 10th item is a duplicate
            item_id = f"tweet_{i // 10}"
            raw_items.append(RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={"id": item_id, "text": f"Tweet {item_id}"},
                timestamp=datetime.now(),
                metadata={}
            ))
        
        # Mock database - simulate existing items for duplicates
        mock_db = AsyncMock()
        
        def mock_find_one(query):
            unique_id = query["unique_id"]
            # Return existing document for every 10th item
            if int(unique_id.split("_")[1]) % 10 == 0:
                return {"unique_id": unique_id}
            return None
        
        mock_db.raw_tweets.find_one.side_effect = mock_find_one
        mock_db.raw_tweets.insert_one.return_value = None
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            # Measure processing time
            start_time = time.time()
            stats = await self.service.process_raw_data(raw_items)
            processing_time = time.time() - start_time
            
            # Verify duplicate detection
            assert stats["duplicates"] > 0
            assert stats["processed"] + stats["duplicates"] == 100
            
            # Performance should be reasonable even with duplicate checks
            assert processing_time < 1.0


class TestErrorHandlingPerformance:
    """Test error handling performance and resilience."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataIngestionService()
    
    @pytest.mark.asyncio
    async def test_api_timeout_handling(self):
        """Test handling of API timeouts."""
        # Mock session that times out
        mock_session = AsyncMock()
        mock_session.get.side_effect = asyncio.TimeoutError("Request timeout")
        
        with patch('app.services.data_ingestion_service.settings') as mock_settings:
            mock_settings.twitter_bearer_token = "test_token"
            
            self.service.session = mock_session
            
            # Test timeout handling
            start_time = time.time()
            result = await self.service.collect_twitter_data(["NFL"])
            handling_time = time.time() - start_time
            
            # Should return empty list and handle timeout gracefully
            assert result == []
            # Should not hang indefinitely
            assert handling_time < 5.0
    
    @pytest.mark.asyncio
    async def test_network_error_resilience(self):
        """Test resilience to network errors."""
        # Mock session with various network errors
        mock_session = AsyncMock()
        
        errors = [
            aiohttp.ClientError("Connection error"),
            aiohttp.ClientTimeout(),
            Exception("Generic network error")
        ]
        
        mock_session.get.side_effect = errors
        
        with patch('app.services.data_ingestion_service.settings') as mock_settings:
            mock_settings.twitter_bearer_token = "test_token"
            
            self.service.session = mock_session
            
            # Test error resilience
            for _ in range(3):
                start_time = time.time()
                result = await self.service.collect_twitter_data(["NFL"])
                handling_time = time.time() - start_time
                
                # Should handle errors gracefully
                assert result == []
                assert handling_time < 2.0  # Should fail fast
    
    @pytest.mark.asyncio
    async def test_database_error_recovery(self):
        """Test recovery from database errors."""
        raw_items = [
            RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={"id": "test", "text": "test"},
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        # Mock database with various errors
        mock_db = AsyncMock()
        mock_db.raw_tweets.find_one.side_effect = [
            Exception("Connection lost"),
            Exception("Timeout"),
            None  # Finally succeeds
        ]
        mock_db.raw_tweets.insert_one.return_value = None
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            # Test database error handling
            start_time = time.time()
            stats = await self.service.process_raw_data(raw_items)
            handling_time = time.time() - start_time
            
            # Should handle database errors gracefully
            assert stats["errors"] == 1  # One error recorded
            assert handling_time < 1.0  # Should not retry indefinitely
    
    @pytest.mark.asyncio
    async def test_high_error_rate_performance(self):
        """Test performance when error rate is high."""
        # Create many items
        raw_items = []
        for i in range(100):
            raw_items.append(RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={"id": f"tweet_{i}", "text": f"Tweet {i}"},
                timestamp=datetime.now(),
                metadata={}
            ))
        
        # Mock database with high error rate (80% failure)
        mock_db = AsyncMock()
        mock_db.raw_tweets.find_one.return_value = None
        
        def mock_insert(doc):
            item_num = int(doc["unique_id"].split("_")[1])
            if item_num % 5 != 0:  # 80% failure rate
                raise Exception("Database error")
            return None
        
        mock_db.raw_tweets.insert_one.side_effect = mock_insert
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            # Test high error rate handling
            start_time = time.time()
            stats = await self.service.process_raw_data(raw_items)
            handling_time = time.time() - start_time
            
            # Should handle high error rate without significant performance degradation
            assert stats["errors"] == 80  # 80% error rate
            assert stats["processed"] == 20  # 20% success rate
            assert handling_time < 2.0  # Should still be reasonably fast