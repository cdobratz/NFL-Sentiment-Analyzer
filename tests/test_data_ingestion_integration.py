"""
Integration tests for data ingestion service.
Tests end-to-end data pipeline with external API mocks.
"""
import pytest
import asyncio
import os
from unittest.mock import AsyncMock, patch, Mock
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
    DataIngestionService, DataSource, RawDataItem, data_ingestion_service
)


class TestDataIngestionServiceIntegration:
    """Integration tests for the complete data ingestion pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = DataIngestionService()
    
    @pytest.mark.asyncio
    async def test_full_twitter_pipeline(self):
        """Test complete Twitter data collection and processing pipeline."""
        # Mock Twitter API response
        mock_twitter_response = {
            "data": [
                {
                    "id": "1234567890",
                    "text": "Amazing touchdown by the Chiefs! #NFL",
                    "created_at": "2024-01-15T20:00:00Z",
                    "author_id": "user123",
                    "public_metrics": {"like_count": 25, "retweet_count": 5},
                    "context_annotations": [
                        {"domain": {"name": "Sport"}, "entity": {"name": "NFL"}}
                    ],
                    "lang": "en"
                },
                {
                    "id": "1234567891",
                    "text": "Terrible fumble by the quarterback",
                    "created_at": "2024-01-15T20:05:00Z",
                    "author_id": "user456",
                    "public_metrics": {"like_count": 10, "retweet_count": 2},
                    "lang": "en"
                }
            ],
            "includes": {
                "users": [
                    {
                        "id": "user123",
                        "name": "NFL Fan",
                        "username": "nflfan123",
                        "verified": False,
                        "public_metrics": {"followers_count": 1500}
                    },
                    {
                        "id": "user456",
                        "name": "Sports Critic",
                        "username": "sportscritic",
                        "verified": True,
                        "public_metrics": {"followers_count": 50000}
                    }
                ]
            }
        }
        
        # Mock database
        mock_db = AsyncMock()
        mock_db.raw_tweets.find_one.return_value = None  # No duplicates
        mock_db.raw_tweets.insert_one.return_value = None
        
        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_twitter_response
        
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        
        with patch('app.services.data_ingestion_service.settings') as mock_settings, \
             patch('app.services.data_ingestion_service.get_database') as mock_get_db:
            
            mock_settings.twitter_bearer_token = "test_token"
            mock_get_db.return_value = mock_db
            
            # Start service
            await self.service.start()
            self.service.session = mock_session
            
            try:
                # Collect Twitter data
                raw_items = await self.service.collect_twitter_data(
                    keywords=["NFL", "Chiefs", "touchdown"],
                    max_results=50
                )
                
                # Verify data collection
                assert len(raw_items) == 2
                assert all(item.source == DataSource.TWITTER for item in raw_items)
                assert all(item.data_type == "tweet" for item in raw_items)
                
                # Verify first tweet
                first_tweet = raw_items[0]
                assert first_tweet.content["id"] == "1234567890"
                assert "touchdown" in first_tweet.content["text"]
                assert first_tweet.content["author"]["verified"] is False
                assert first_tweet.content["metrics"]["like_count"] == 25
                
                # Verify second tweet
                second_tweet = raw_items[1]
                assert second_tweet.content["id"] == "1234567891"
                assert "fumble" in second_tweet.content["text"]
                assert second_tweet.content["author"]["verified"] is True
                
                # Process raw data
                stats = await self.service.process_raw_data(raw_items)
                
                # Verify processing stats
                assert stats["processed"] == 2
                assert stats["tweets"] == 2
                assert stats["duplicates"] == 0
                assert stats["errors"] == 0
                
                # Verify database calls
                assert mock_db.raw_tweets.insert_one.call_count == 2
                
            finally:
                await self.service.stop()
    
    @pytest.mark.asyncio
    async def test_full_espn_pipeline(self):
        """Test complete ESPN data collection and processing pipeline."""
        # Mock ESPN scoreboard response
        mock_scoreboard_response = {
            "events": [
                {
                    "id": "401547439",
                    "name": "Kansas City Chiefs vs Buffalo Bills",
                    "shortName": "KC vs BUF",
                    "date": "2024-01-21T18:30:00Z",
                    "status": {
                        "type": {"completed": False, "description": "Scheduled"}
                    },
                    "competitions": [
                        {
                            "competitors": [
                                {
                                    "team": {
                                        "id": "12",
                                        "abbreviation": "KC",
                                        "displayName": "Kansas City Chiefs"
                                    },
                                    "homeAway": "home"
                                },
                                {
                                    "team": {
                                        "id": "2",
                                        "abbreviation": "BUF",
                                        "displayName": "Buffalo Bills"
                                    },
                                    "homeAway": "away"
                                }
                            ]
                        }
                    ],
                    "season": {"type": "postseason", "year": 2024},
                    "week": {"number": 2}
                }
            ]
        }
        
        # Mock ESPN news response
        mock_news_response = {
            "articles": [
                {
                    "id": "39284756",
                    "headline": "Chiefs prepare for playoff showdown",
                    "description": "Kansas City Chiefs gear up for crucial playoff game",
                    "story": "The Kansas City Chiefs are preparing for their upcoming playoff game...",
                    "published": "2024-01-20T15:30:00Z",
                    "categories": [{"description": "NFL"}],
                    "type": "story",
                    "links": {"web": {"href": "https://espn.com/article/123"}}
                }
            ]
        }
        
        # Mock database
        mock_db = AsyncMock()
        mock_db.raw_games.find_one.return_value = None
        mock_db.raw_news.find_one.return_value = None
        mock_db.raw_games.insert_one.return_value = None
        mock_db.raw_news.insert_one.return_value = None
        
        # Mock HTTP responses
        mock_scoreboard_response_obj = AsyncMock()
        mock_scoreboard_response_obj.status = 200
        mock_scoreboard_response_obj.json.return_value = mock_scoreboard_response
        
        mock_news_response_obj = AsyncMock()
        mock_news_response_obj.status = 200
        mock_news_response_obj.json.return_value = mock_news_response
        
        mock_session = AsyncMock()
        mock_session.get.side_effect = [
            mock_scoreboard_response_obj.__aenter__.return_value,
            mock_news_response_obj.__aenter__.return_value
        ]
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            # Start service
            await self.service.start()
            self.service.session = mock_session
            
            try:
                # Fetch ESPN data
                raw_items = await self.service.fetch_espn_data()
                
                # Verify data collection
                assert len(raw_items) == 2  # 1 game + 1 news article
                
                # Find game and news items
                game_items = [item for item in raw_items if item.data_type == "game"]
                news_items = [item for item in raw_items if item.data_type == "news"]
                
                assert len(game_items) == 1
                assert len(news_items) == 1
                
                # Verify game data
                game_item = game_items[0]
                assert game_item.source == DataSource.ESPN
                assert game_item.content["id"] == "401547439"
                assert "Chiefs" in game_item.content["name"]
                assert game_item.metadata["season_type"] == "postseason"
                assert game_item.metadata["week_number"] == 2
                
                # Verify news data
                news_item = news_items[0]
                assert news_item.source == DataSource.ESPN
                assert news_item.content["id"] == "39284756"
                assert "Chiefs" in news_item.content["headline"]
                assert news_item.metadata["type"] == "story"
                
                # Process raw data
                stats = await self.service.process_raw_data(raw_items)
                
                # Verify processing stats
                assert stats["processed"] == 2
                assert stats["games"] == 1
                assert stats["news"] == 1
                assert stats["duplicates"] == 0
                assert stats["errors"] == 0
                
            finally:
                await self.service.stop()
    
    @pytest.mark.asyncio
    async def test_full_betting_pipeline(self):
        """Test complete betting lines collection and processing pipeline."""
        # Mock database
        mock_db = AsyncMock()
        mock_db.raw_betting_lines.find_one.return_value = None
        mock_db.raw_betting_lines.insert_one.return_value = None
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            # Start service
            await self.service.start()
            
            try:
                # Get betting lines (mock implementation)
                raw_items = await self.service.get_betting_lines(["draftkings", "mgm"])
                
                # Verify data collection
                assert len(raw_items) == 2  # DraftKings + MGM
                
                # Find DraftKings and MGM items
                dk_items = [item for item in raw_items if item.source == DataSource.DRAFTKINGS]
                mgm_items = [item for item in raw_items if item.source == DataSource.MGM]
                
                assert len(dk_items) == 1
                assert len(mgm_items) == 1
                
                # Verify DraftKings data
                dk_item = dk_items[0]
                assert dk_item.data_type == "betting_line"
                assert dk_item.metadata["sportsbook"] == "DraftKings"
                assert "spread" in dk_item.content
                assert "over_under" in dk_item.content
                assert "home_moneyline" in dk_item.content
                
                # Verify MGM data
                mgm_item = mgm_items[0]
                assert mgm_item.data_type == "betting_line"
                assert mgm_item.metadata["sportsbook"] == "MGM"
                assert "spread" in mgm_item.content
                
                # Process raw data
                stats = await self.service.process_raw_data(raw_items)
                
                # Verify processing stats
                assert stats["processed"] == 2
                assert stats["betting_lines"] == 2
                assert stats["duplicates"] == 0
                assert stats["errors"] == 0
                
            finally:
                await self.service.stop()
    
    @pytest.mark.asyncio
    async def test_mixed_data_sources_pipeline(self):
        """Test processing data from multiple sources simultaneously."""
        # Create mixed raw data items
        raw_items = [
            RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={
                    "id": "tweet123",
                    "text": "Great game!",
                    "created_at": "2024-01-15T20:00:00Z",
                    "author": {"name": "Fan"},
                    "metrics": {"like_count": 5},
                    "context_annotations": []
                },
                timestamp=datetime.now(),
                metadata={"lang": "en"}
            ),
            RawDataItem(
                source=DataSource.ESPN,
                data_type="news",
                content={
                    "id": "news123",
                    "headline": "Team wins big",
                    "description": "Great victory",
                    "published": "2024-01-15T19:00:00Z"
                },
                timestamp=datetime.now(),
                metadata={"source": "espn_news"}
            ),
            RawDataItem(
                source=DataSource.ESPN,
                data_type="game",
                content={
                    "id": "game123",
                    "name": "Team A vs Team B",
                    "date": "2024-01-15T20:00:00Z",
                    "status": {"completed": True},
                    "competitions": []
                },
                timestamp=datetime.now(),
                metadata={"source": "espn_scoreboard"}
            ),
            RawDataItem(
                source=DataSource.DRAFTKINGS,
                data_type="betting_line",
                content={
                    "game_id": "game123",
                    "home_team": "Team A",
                    "away_team": "Team B",
                    "spread": -3.5,
                    "over_under": 45.5
                },
                timestamp=datetime.now(),
                metadata={"sportsbook": "DraftKings"}
            )
        ]
        
        # Mock database
        mock_db = AsyncMock()
        mock_db.raw_tweets.find_one.return_value = None
        mock_db.raw_news.find_one.return_value = None
        mock_db.raw_games.find_one.return_value = None
        mock_db.raw_betting_lines.find_one.return_value = None
        
        mock_db.raw_tweets.insert_one.return_value = None
        mock_db.raw_news.insert_one.return_value = None
        mock_db.raw_games.insert_one.return_value = None
        mock_db.raw_betting_lines.insert_one.return_value = None
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            # Process mixed data
            stats = await self.service.process_raw_data(raw_items)
            
            # Verify processing stats
            assert stats["processed"] == 4
            assert stats["tweets"] == 1
            assert stats["news"] == 1
            assert stats["games"] == 1
            assert stats["betting_lines"] == 1
            assert stats["duplicates"] == 0
            assert stats["errors"] == 0
            
            # Verify all database operations were called
            mock_db.raw_tweets.insert_one.assert_called_once()
            mock_db.raw_news.insert_one.assert_called_once()
            mock_db.raw_games.insert_one.assert_called_once()
            mock_db.raw_betting_lines.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_duplicate_handling_pipeline(self):
        """Test handling of duplicate data across the pipeline."""
        # Create duplicate raw data items
        raw_items = [
            RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={"id": "duplicate123", "text": "Test tweet"},
                timestamp=datetime.now(),
                metadata={}
            ),
            RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={"id": "unique456", "text": "Another tweet"},
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        # Mock database - first tweet is duplicate, second is new
        mock_db = AsyncMock()
        mock_db.raw_tweets.find_one.side_effect = [
            {"unique_id": "duplicate123"},  # First call returns existing
            None  # Second call returns None (new)
        ]
        mock_db.raw_tweets.insert_one.return_value = None
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            # Process data with duplicates
            stats = await self.service.process_raw_data(raw_items)
            
            # Verify duplicate handling
            assert stats["processed"] == 1  # Only one processed
            assert stats["duplicates"] == 1  # One duplicate found
            assert stats["tweets"] == 1  # One tweet stored
            assert stats["errors"] == 0
            
            # Verify only one insert was called (for the non-duplicate)
            mock_db.raw_tweets.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_handling_pipeline(self):
        """Test error handling throughout the pipeline."""
        # Create raw data items
        raw_items = [
            RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={"id": "tweet1", "text": "Test"},
                timestamp=datetime.now(),
                metadata={}
            ),
            RawDataItem(
                source=DataSource.ESPN,
                data_type="news",
                content={"id": "news1", "headline": "Test"},
                timestamp=datetime.now(),
                metadata={}
            )
        ]
        
        # Mock database with errors
        mock_db = AsyncMock()
        mock_db.raw_tweets.find_one.return_value = None
        mock_db.raw_news.find_one.return_value = None
        
        # First insert succeeds, second fails
        mock_db.raw_tweets.insert_one.return_value = None
        mock_db.raw_news.insert_one.side_effect = Exception("Database error")
        
        with patch('app.services.data_ingestion_service.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            # Process data with errors
            stats = await self.service.process_raw_data(raw_items)
            
            # Verify error handling
            assert stats["processed"] == 1  # One succeeded
            assert stats["errors"] == 1  # One failed
            assert stats["tweets"] == 1  # Tweet was processed
            assert stats["news"] == 0  # News failed
    
    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self):
        """Test rate limiting integration with API calls."""
        # Mock settings
        with patch('app.services.data_ingestion_service.settings') as mock_settings:
            mock_settings.twitter_bearer_token = "test_token"
            
            # Mock session with rate limit response
            mock_response = AsyncMock()
            mock_response.status = 429  # Rate limit exceeded
            mock_response.text.return_value = "Rate limit exceeded"
            
            mock_session = AsyncMock()
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            # Start service
            await self.service.start()
            self.service.session = mock_session
            
            try:
                # Attempt Twitter data collection
                result = await self.service.collect_twitter_data(["NFL"])
                
                # Should return empty list due to rate limit
                assert result == []
                
                # Verify rate limiter was called
                assert len(self.service.rate_limiters[DataSource.TWITTER].requests) > 0
                
            finally:
                await self.service.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_data_collection(self):
        """Test concurrent data collection from multiple sources."""
        # Mock responses for different sources
        mock_twitter_response = {
            "data": [{"id": "123", "text": "test", "created_at": "2024-01-15T20:00:00Z", "author_id": "user1"}],
            "includes": {"users": [{"id": "user1", "name": "Test User", "username": "test"}]}
        }
        
        mock_espn_scoreboard = {"events": []}
        mock_espn_news = {"articles": []}
        
        # Mock HTTP responses
        mock_twitter_resp = AsyncMock()
        mock_twitter_resp.status = 200
        mock_twitter_resp.json.return_value = mock_twitter_response
        
        mock_espn_resp1 = AsyncMock()
        mock_espn_resp1.status = 200
        mock_espn_resp1.json.return_value = mock_espn_scoreboard
        
        mock_espn_resp2 = AsyncMock()
        mock_espn_resp2.status = 200
        mock_espn_resp2.json.return_value = mock_espn_news
        
        mock_session = AsyncMock()
        mock_session.get.side_effect = [
            mock_twitter_resp.__aenter__.return_value,
            mock_espn_resp1.__aenter__.return_value,
            mock_espn_resp2.__aenter__.return_value
        ]
        
        with patch('app.services.data_ingestion_service.settings') as mock_settings:
            mock_settings.twitter_bearer_token = "test_token"
            
            # Start service
            await self.service.start()
            self.service.session = mock_session
            
            try:
                # Collect data from multiple sources concurrently
                twitter_task = self.service.collect_twitter_data(["NFL"])
                espn_task = self.service.fetch_espn_data()
                betting_task = self.service.get_betting_lines(["draftkings"])
                
                # Wait for all tasks to complete
                twitter_data, espn_data, betting_data = await asyncio.gather(
                    twitter_task, espn_task, betting_task
                )
                
                # Verify all data was collected
                assert len(twitter_data) == 1
                assert len(espn_data) == 0  # Empty responses
                assert len(betting_data) == 1  # Mock betting data
                
                # Verify different sources
                assert twitter_data[0].source == DataSource.TWITTER
                assert betting_data[0].source == DataSource.DRAFTKINGS
                
            finally:
                await self.service.stop()


class TestGlobalServiceInstance:
    """Test the global data ingestion service instance."""
    
    def test_global_instance_exists(self):
        """Test that global service instance exists."""
        assert data_ingestion_service is not None
        assert isinstance(data_ingestion_service, DataIngestionService)
    
    @pytest.mark.asyncio
    async def test_global_instance_lifecycle(self):
        """Test global service instance lifecycle management."""
        # Ensure service is stopped initially
        if data_ingestion_service.is_running:
            await data_ingestion_service.stop()
        
        assert data_ingestion_service.is_running is False
        
        # Start service
        await data_ingestion_service.start()
        assert data_ingestion_service.is_running is True
        assert data_ingestion_service.session is not None
        
        # Stop service
        await data_ingestion_service.stop()
        assert data_ingestion_service.is_running is False
    
    @pytest.mark.asyncio
    async def test_global_instance_multiple_starts(self):
        """Test that multiple starts don't cause issues."""
        # Ensure service is stopped initially
        if data_ingestion_service.is_running:
            await data_ingestion_service.stop()
        
        # Start service multiple times
        await data_ingestion_service.start()
        first_session = data_ingestion_service.session
        
        await data_ingestion_service.start()
        second_session = data_ingestion_service.session
        
        # Should reuse the same session
        assert first_session == second_session
        assert data_ingestion_service.is_running is True
        
        # Clean up
        await data_ingestion_service.stop()