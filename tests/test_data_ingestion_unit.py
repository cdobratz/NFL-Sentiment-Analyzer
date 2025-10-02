"""
Unit tests for data ingestion service.
Tests individual methods and components without external dependencies.
"""
import pytest
import asyncio
import os
from unittest.mock import MagicMock, AsyncMock, patch, Mock
from datetime import datetime, timedelta
from typing import List, Dict, Any
import aiohttp
import json

# Set test environment variables before importing
os.environ.setdefault('SECRET_KEY', 'test-secret-key')
os.environ.setdefault('MONGODB_URL', 'mongodb://localhost:27017/test')
os.environ.setdefault('DATABASE_NAME', 'test_db')
os.environ.setdefault('REDIS_URL', 'redis://localhost:6379/1')
os.environ.setdefault('TWITTER_BEARER_TOKEN', 'test_token')

from app.services.data_ingestion_service import (
    DataIngestionService, DataSource, RawDataItem, RateLimiter
)


class TestRateLimiter:
    """Test rate limiter functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes correctly."""
        limiter = RateLimiter(max_requests=100, time_window=60)
        
        assert limiter.max_requests == 100
        assert limiter.time_window == 60
        assert limiter.requests == []
    
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_first_request(self):
        """Test rate limiter allows first request immediately."""
        limiter = RateLimiter(max_requests=10, time_window=60)
        
        start_time = datetime.now()
        await limiter.acquire()
        end_time = datetime.now()
        
        # Should not wait for first request
        assert (end_time - start_time).total_seconds() < 0.1
        assert len(limiter.requests) == 1
    
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_within_limit(self):
        """Test rate limiter allows requests within limit."""
        limiter = RateLimiter(max_requests=5, time_window=60)
        
        # Make 5 requests quickly
        for _ in range(5):
            await limiter.acquire()
        
        assert len(limiter.requests) == 5
    
    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_exceeds_limit(self):
        """Test rate limiter blocks when limit exceeded."""
        limiter = RateLimiter(max_requests=2, time_window=1)
        
        # Make 2 requests to hit limit
        await limiter.acquire()
        await limiter.acquire()
        
        # Third request should be delayed
        start_time = datetime.now()
        await limiter.acquire()
        end_time = datetime.now()
        
        # Should have waited at least some time
        assert (end_time - start_time).total_seconds() >= 0.5
    
    @pytest.mark.asyncio
    async def test_rate_limiter_cleanup_old_requests(self):
        """Test rate limiter cleans up old requests."""
        limiter = RateLimiter(max_requests=2, time_window=1)
        
        # Add old request manually
        old_time = datetime.now() - timedelta(seconds=2)
        limiter.requests.append(old_time)
        
        await limiter.acquire()
        
        # Old request should be cleaned up
        assert len(limiter.requests) == 1
        assert limiter.requests[0] > old_time


class TestRawDataItem:
    """Test RawDataItem data class."""
    
    def test_raw_data_item_creation(self):
        """Test RawDataItem can be created correctly."""
        timestamp = datetime.now()
        item = RawDataItem(
            source=DataSource.TWITTER,
            data_type="tweet",
            content={"id": "123", "text": "test"},
            timestamp=timestamp,
            metadata={"lang": "en"}
        )
        
        assert item.source == DataSource.TWITTER
        assert item.data_type == "tweet"
        assert item.content["id"] == "123"
        assert item.timestamp == timestamp
        assert item.metadata["lang"] == "en"


class TestDataIngestionService:
    """Test main data ingestion service functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataIngestionService()
    
    def test_service_initialization(self):
        """Test service initializes correctly."""
        assert self.service.session is None
        assert self.service.is_running is False
        assert len(self.service.rate_limiters) == 4
        assert DataSource.TWITTER in self.service.rate_limiters
        assert DataSource.ESPN in self.service.rate_limiters
        assert DataSource.DRAFTKINGS in self.service.rate_limiters
        assert DataSource.MGM in self.service.rate_limiters
    
    @pytest.mark.asyncio
    async def test_start_service(self):
        """Test service starts correctly."""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value = AsyncMock()
            
            await self.service.start()
            
            assert self.service.session is not None
            assert self.service.is_running is True
            mock_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_service(self):
        """Test service stops correctly."""
        # Set up service as if it's running
        mock_session = AsyncMock()
        self.service.session = mock_session
        self.service.is_running = True
        
        await self.service.stop()
        
        assert self.service.is_running is False
        mock_session.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stop_service_no_session(self):
        """Test service stops correctly when no session exists."""
        self.service.is_running = True
        
        await self.service.stop()
        
        assert self.service.is_running is False
    
    @pytest.mark.asyncio
    async def test_collect_twitter_data_no_token(self):
        """Test Twitter data collection without bearer token."""
        with patch('app.services.data_ingestion_service.settings') as mock_settings:
            mock_settings.twitter_bearer_token = None
            
            result = await self.service.collect_twitter_data(["test"])
            
            assert result == []
    
    @pytest.mark.asyncio
    async def test_collect_twitter_data_success(self):
        """Test successful Twitter data collection."""
        mock_response_data = {
            "data": [
                {
                    "id": "1234567890",
                    "text": "Great game by the team!",
                    "created_at": "2024-01-15T20:00:00Z",
                    "author_id": "user123",
                    "public_metrics": {"like_count": 10},
                    "lang": "en"
                }
            ],
            "includes": {
                "users": [
                    {
                        "id": "user123",
                        "name": "Test User",
                        "username": "testuser",
                        "verified": False,
                        "public_metrics": {"followers_count": 100}
                    }
                ]
            }
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_response_data
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('app.services.data_ingestion_service.settings') as mock_settings:
            mock_settings.twitter_bearer_token = "test_token"
            
            self.service.session = mock_session
            
            result = await self.service.collect_twitter_data(["NFL", "football"])
            
            assert len(result) == 1
            assert result[0].source == DataSource.TWITTER
            assert result[0].data_type == "tweet"
            assert result[0].content["id"] == "1234567890"
            assert result[0].content["text"] == "Great game by the team!"
            assert result[0].content["author"]["name"] == "Test User"
    
    @pytest.mark.asyncio
    async def test_collect_twitter_data_api_error(self):
        """Test Twitter data collection with API error."""
        mock_response = AsyncMock()
        mock_response.status = 429  # Rate limit exceeded
        mock_response.text.return_value = "Rate limit exceeded"
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('app.services.data_ingestion_service.settings') as mock_settings:
            mock_settings.twitter_bearer_token = "test_token"
            
            self.service.session = mock_session
            
            result = await self.service.collect_twitter_data(["NFL"])
            
            assert result == []
    
    @pytest.mark.asyncio
    async def test_collect_twitter_data_exception(self):
        """Test Twitter data collection with exception."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("Network error")
        
        with patch('app.services.data_ingestion_service.settings') as mock_settings:
            mock_settings.twitter_bearer_token = "test_token"
            
            self.service.session = mock_session
            
            result = await self.service.collect_twitter_data(["NFL"])
            
            assert result == []
    
    def test_process_twitter_data_empty(self):
        """Test processing empty Twitter data."""
        data = {"data": []}
        
        result = self.service._process_twitter_data(data)
        
        assert result == []
    
    def test_process_twitter_data_with_users(self):
        """Test processing Twitter data with user information."""
        data = {
            "data": [
                {
                    "id": "123",
                    "text": "Test tweet",
                    "created_at": "2024-01-15T20:00:00Z",
                    "author_id": "user1",
                    "public_metrics": {"like_count": 5},
                    "lang": "en"
                }
            ],
            "includes": {
                "users": [
                    {
                        "id": "user1",
                        "name": "Test User",
                        "username": "testuser",
                        "verified": True,
                        "public_metrics": {"followers_count": 1000}
                    }
                ]
            }
        }
        
        result = self.service._process_twitter_data(data)
        
        assert len(result) == 1
        item = result[0]
        assert item.source == DataSource.TWITTER
        assert item.data_type == "tweet"
        assert item.content["id"] == "123"
        assert item.content["author"]["name"] == "Test User"
        assert item.content["author"]["verified"] is True
    
    def test_process_twitter_data_missing_users(self):
        """Test processing Twitter data with missing user information."""
        data = {
            "data": [
                {
                    "id": "123",
                    "text": "Test tweet",
                    "created_at": "2024-01-15T20:00:00Z",
                    "author_id": "user1",
                    "public_metrics": {"like_count": 5}
                }
            ]
        }
        
        result = self.service._process_twitter_data(data)
        
        assert len(result) == 1
        item = result[0]
        assert item.content["author"]["name"] == ""
        assert item.content["author"]["verified"] is False
    
    @pytest.mark.asyncio
    async def test_fetch_espn_data_default(self):
        """Test fetching ESPN data with default parameters."""
        with patch.object(self.service, '_fetch_espn_scoreboard') as mock_scoreboard, \
             patch.object(self.service, '_fetch_espn_news') as mock_news:
            
            mock_scoreboard.return_value = [Mock()]
            mock_news.return_value = [Mock()]
            
            result = await self.service.fetch_espn_data()
            
            mock_scoreboard.assert_called_once()
            mock_news.assert_called_once()
            assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_fetch_espn_data_with_game_ids(self):
        """Test fetching ESPN data with specific game IDs."""
        with patch.object(self.service, '_fetch_espn_scoreboard') as mock_scoreboard, \
             patch.object(self.service, '_fetch_espn_news') as mock_news:
            
            mock_news.return_value = [Mock()]
            
            result = await self.service.fetch_espn_data(game_ids=["game1", "game2"])
            
            mock_scoreboard.assert_not_called()
            mock_news.assert_called_once()
            assert len(result) == 1
    
    @pytest.mark.asyncio
    async def test_fetch_espn_scoreboard_success(self):
        """Test successful ESPN scoreboard fetch."""
        mock_response_data = {
            "events": [
                {
                    "id": "game1",
                    "name": "Team A vs Team B",
                    "shortName": "A vs B",
                    "date": "2024-01-15T20:00:00Z",
                    "status": {"type": {"completed": False}},
                    "competitions": [],
                    "season": {"type": "regular"},
                    "week": {"number": 1}
                }
            ]
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_response_data
        
        # Create proper async context manager mock
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        self.service.session = mock_session
        
        result = await self.service._fetch_espn_scoreboard()
        
        assert len(result) == 1
        assert result[0].source == DataSource.ESPN
        assert result[0].data_type == "game"
        assert result[0].content["id"] == "game1"
    
    @pytest.mark.asyncio
    async def test_fetch_espn_scoreboard_error(self):
        """Test ESPN scoreboard fetch with error."""
        mock_response = AsyncMock()
        mock_response.status = 500
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        self.service.session = mock_session
        
        result = await self.service._fetch_espn_scoreboard()
        
        assert result == []
    
    @pytest.mark.asyncio
    async def test_fetch_espn_scoreboard_exception(self):
        """Test ESPN scoreboard fetch with exception."""
        mock_session = AsyncMock()
        mock_session.get.side_effect = Exception("Network error")
        
        self.service.session = mock_session
        
        result = await self.service._fetch_espn_scoreboard()
        
        assert result == []
    
    def test_process_espn_scoreboard_empty(self):
        """Test processing empty ESPN scoreboard data."""
        data = {"events": []}
        
        result = self.service._process_espn_scoreboard(data)
        
        assert result == []
    
    def test_process_espn_scoreboard_with_events(self):
        """Test processing ESPN scoreboard data with events."""
        data = {
            "events": [
                {
                    "id": "game1",
                    "name": "Team A vs Team B",
                    "shortName": "A vs B",
                    "date": "2024-01-15T20:00:00Z",
                    "status": {"type": {"completed": False}},
                    "competitions": [],
                    "season": {"type": "regular"},
                    "week": {"number": 1}
                }
            ]
        }
        
        result = self.service._process_espn_scoreboard(data)
        
        assert len(result) == 1
        item = result[0]
        assert item.source == DataSource.ESPN
        assert item.data_type == "game"
        assert item.content["id"] == "game1"
        assert item.metadata["season_type"] == "regular"
        assert item.metadata["week_number"] == 1
    
    @pytest.mark.asyncio
    async def test_fetch_espn_news_success(self):
        """Test successful ESPN news fetch."""
        mock_response_data = {
            "articles": [
                {
                    "id": "article1",
                    "headline": "Test Headline",
                    "description": "Test description",
                    "story": "Test story content",
                    "published": "2024-01-15T20:00:00Z",
                    "categories": [{"description": "NFL"}],
                    "type": "story"
                }
            ]
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_response_data
        
        # Create proper async context manager mock
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.get.return_value.__aexit__ = AsyncMock(return_value=None)
        
        self.service.session = mock_session
        
        result = await self.service._fetch_espn_news()
        
        assert len(result) == 1
        assert result[0].source == DataSource.ESPN
        assert result[0].data_type == "news"
        assert result[0].content["headline"] == "Test Headline"
    
    def test_process_espn_news_empty(self):
        """Test processing empty ESPN news data."""
        data = {"articles": []}
        
        result = self.service._process_espn_news(data)
        
        assert result == []
    
    def test_process_espn_news_with_articles(self):
        """Test processing ESPN news data with articles."""
        data = {
            "articles": [
                {
                    "id": "article1",
                    "headline": "Test Headline",
                    "description": "Test description",
                    "published": "2024-01-15T20:00:00Z",
                    "type": "story"
                }
            ]
        }
        
        result = self.service._process_espn_news(data)
        
        assert len(result) == 1
        item = result[0]
        assert item.source == DataSource.ESPN
        assert item.data_type == "news"
        assert item.content["headline"] == "Test Headline"
        assert item.metadata["type"] == "story"
    
    @pytest.mark.asyncio
    async def test_get_betting_lines_default(self):
        """Test getting betting lines with default sportsbooks."""
        with patch.object(self.service, '_fetch_draftkings_lines') as mock_dk, \
             patch.object(self.service, '_fetch_mgm_lines') as mock_mgm:
            
            mock_dk.return_value = [Mock()]
            mock_mgm.return_value = [Mock()]
            
            result = await self.service.get_betting_lines()
            
            mock_dk.assert_called_once()
            mock_mgm.assert_called_once()
            assert len(result) == 2
    
    @pytest.mark.asyncio
    async def test_get_betting_lines_specific_sportsbooks(self):
        """Test getting betting lines from specific sportsbooks."""
        with patch.object(self.service, '_fetch_draftkings_lines') as mock_dk, \
             patch.object(self.service, '_fetch_mgm_lines') as mock_mgm:
            
            mock_dk.return_value = [Mock()]
            
            result = await self.service.get_betting_lines(["draftkings"])
            
            mock_dk.assert_called_once()
            mock_mgm.assert_not_called()
            assert len(result) == 1
    
    @pytest.mark.asyncio
    async def test_fetch_draftkings_lines(self):
        """Test fetching DraftKings betting lines (mock implementation)."""
        result = await self.service._fetch_draftkings_lines()
        
        assert len(result) == 1
        item = result[0]
        assert item.source == DataSource.DRAFTKINGS
        assert item.data_type == "betting_line"
        assert item.metadata["sportsbook"] == "DraftKings"
        assert "spread" in item.content
        assert "over_under" in item.content
    
    @pytest.mark.asyncio
    async def test_fetch_mgm_lines(self):
        """Test fetching MGM betting lines (mock implementation)."""
        result = await self.service._fetch_mgm_lines()
        
        assert len(result) == 1
        item = result[0]
        assert item.source == DataSource.MGM
        assert item.data_type == "betting_line"
        assert item.metadata["sportsbook"] == "MGM"
        assert "spread" in item.content
        assert "over_under" in item.content
    
    @pytest.mark.asyncio
    async def test_process_raw_data_empty(self):
        """Test processing empty raw data list."""
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db:
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db
            
            result = await self.service.process_raw_data([])
            
            assert result["processed"] == 0
            assert result["duplicates"] == 0
            assert result["errors"] == 0
    
    @pytest.mark.asyncio
    async def test_process_raw_data_with_tweets(self):
        """Test processing raw data with tweets."""
        raw_items = [
            RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={"id": "123", "text": "test"},
                timestamp=datetime.now(),
                metadata={"lang": "en"}
            )
        ]
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db, \
             patch.object(self.service, '_is_duplicate') as mock_is_duplicate, \
             patch.object(self.service, '_store_tweet') as mock_store:
            
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db
            mock_is_duplicate.return_value = False
            mock_store.return_value = None
            
            result = await self.service.process_raw_data(raw_items)
            
            assert result["processed"] == 1
            assert result["tweets"] == 1
            assert result["duplicates"] == 0
            assert result["errors"] == 0
            mock_store.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_raw_data_with_duplicates(self):
        """Test processing raw data with duplicates."""
        raw_items = [
            RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={"id": "123", "text": "test"},
                timestamp=datetime.now(),
                metadata={"lang": "en"}
            )
        ]
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db, \
             patch.object(self.service, '_is_duplicate') as mock_is_duplicate:
            
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db
            mock_is_duplicate.return_value = True
            
            result = await self.service.process_raw_data(raw_items)
            
            assert result["processed"] == 0
            assert result["duplicates"] == 1
            assert result["errors"] == 0
    
    @pytest.mark.asyncio
    async def test_process_raw_data_with_errors(self):
        """Test processing raw data with errors."""
        raw_items = [
            RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={"id": "123", "text": "test"},
                timestamp=datetime.now(),
                metadata={"lang": "en"}
            )
        ]
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db, \
             patch.object(self.service, '_is_duplicate') as mock_is_duplicate, \
             patch.object(self.service, '_store_tweet') as mock_store:
            
            mock_db = AsyncMock()
            mock_get_db.return_value = mock_db
            mock_is_duplicate.return_value = False
            mock_store.side_effect = Exception("Database error")
            
            result = await self.service.process_raw_data(raw_items)
            
            assert result["processed"] == 0
            assert result["errors"] == 1
    
    @pytest.mark.asyncio
    async def test_is_duplicate_tweet_exists(self):
        """Test duplicate detection for existing tweet."""
        mock_db = AsyncMock()
        # Mock the collection access properly
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = {"unique_id": "123"}
        mock_db.__getitem__.return_value = mock_collection
        
        item = RawDataItem(
            source=DataSource.TWITTER,
            data_type="tweet",
            content={"id": "123", "text": "test"},
            timestamp=datetime.now(),
            metadata={}
        )
        
        result = await self.service._is_duplicate(mock_db, item)
        
        assert result is True
        mock_collection.find_one.assert_called_once_with({"unique_id": "123"})
    
    @pytest.mark.asyncio
    async def test_is_duplicate_tweet_not_exists(self):
        """Test duplicate detection for new tweet."""
        mock_db = AsyncMock()
        # Mock the collection access properly
        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_db.__getitem__.return_value = mock_collection
        
        item = RawDataItem(
            source=DataSource.TWITTER,
            data_type="tweet",
            content={"id": "123", "text": "test"},
            timestamp=datetime.now(),
            metadata={}
        )
        
        result = await self.service._is_duplicate(mock_db, item)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_is_duplicate_no_unique_id(self):
        """Test duplicate detection with no unique ID."""
        mock_db = AsyncMock()
        
        item = RawDataItem(
            source=DataSource.TWITTER,
            data_type="tweet",
            content={"text": "test"},  # No ID
            timestamp=datetime.now(),
            metadata={}
        )
        
        result = await self.service._is_duplicate(mock_db, item)
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_store_tweet(self):
        """Test storing tweet data."""
        mock_db = AsyncMock()
        
        item = RawDataItem(
            source=DataSource.TWITTER,
            data_type="tweet",
            content={
                "id": "123",
                "text": "test tweet",
                "created_at": "2024-01-15T20:00:00Z",
                "author": {"name": "Test User"},
                "metrics": {"like_count": 5},
                "context_annotations": []
            },
            timestamp=datetime.now(),
            metadata={"lang": "en"}
        )
        
        await self.service._store_tweet(mock_db, item)
        
        mock_db.raw_tweets.insert_one.assert_called_once()
        call_args = mock_db.raw_tweets.insert_one.call_args[0][0]
        assert call_args["unique_id"] == "123"
        assert call_args["text"] == "test tweet"
        assert call_args["source"] == DataSource.TWITTER
    
    @pytest.mark.asyncio
    async def test_store_news(self):
        """Test storing news data."""
        mock_db = AsyncMock()
        
        item = RawDataItem(
            source=DataSource.ESPN,
            data_type="news",
            content={
                "id": "article1",
                "headline": "Test Headline",
                "description": "Test description",
                "published": "2024-01-15T20:00:00Z"
            },
            timestamp=datetime.now(),
            metadata={"source": "espn_news"}
        )
        
        await self.service._store_news(mock_db, item)
        
        mock_db.raw_news.insert_one.assert_called_once()
        call_args = mock_db.raw_news.insert_one.call_args[0][0]
        assert call_args["unique_id"] == "article1"
        assert call_args["headline"] == "Test Headline"
        assert call_args["source"] == DataSource.ESPN
    
    @pytest.mark.asyncio
    async def test_store_game_data(self):
        """Test storing game data."""
        mock_db = AsyncMock()
        
        item = RawDataItem(
            source=DataSource.ESPN,
            data_type="game",
            content={
                "id": "game1",
                "name": "Team A vs Team B",
                "date": "2024-01-15T20:00:00Z",
                "status": {"completed": False},
                "competitions": []
            },
            timestamp=datetime.now(),
            metadata={"source": "espn_scoreboard"}
        )
        
        await self.service._store_game_data(mock_db, item)
        
        mock_db.raw_games.insert_one.assert_called_once()
        call_args = mock_db.raw_games.insert_one.call_args[0][0]
        assert call_args["unique_id"] == "game1"
        assert call_args["name"] == "Team A vs Team B"
        assert call_args["source"] == DataSource.ESPN
    
    @pytest.mark.asyncio
    async def test_store_betting_line(self):
        """Test storing betting line data."""
        mock_db = AsyncMock()
        
        item = RawDataItem(
            source=DataSource.DRAFTKINGS,
            data_type="betting_line",
            content={
                "game_id": "game1",
                "home_team": "Team A",
                "away_team": "Team B",
                "spread": -3.5,
                "over_under": 45.5
            },
            timestamp=datetime.now(),
            metadata={"sportsbook": "DraftKings"}
        )
        
        await self.service._store_betting_line(mock_db, item)
        
        mock_db.raw_betting_lines.insert_one.assert_called_once()
        call_args = mock_db.raw_betting_lines.insert_one.call_args[0][0]
        assert call_args["unique_id"] == f"game1_{DataSource.DRAFTKINGS}"
        assert call_args["home_team"] == "Team A"
        assert call_args["spread"] == -3.5
        assert call_args["source"] == DataSource.DRAFTKINGS