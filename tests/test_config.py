"""Test configuration and utilities."""

import os
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock


class TestConfig:
    """Test configuration settings."""
    
    # Database settings
    MONGODB_URL = os.getenv("TEST_MONGODB_URL", "mongodb://localhost:27017")
    DATABASE_NAME = "nfl_analyzer_test"
    
    # Redis settings
    REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")
    
    # Auth settings
    SECRET_KEY = "test-secret-key-for-testing-only"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # External API settings (use mock endpoints in tests)
    TWITTER_API_URL = "http://mock-twitter-api"
    ESPN_API_URL = "http://mock-espn-api"
    DRAFTKINGS_API_URL = "http://mock-draftkings-api"
    
    # ML/AI settings
    HUGGINGFACE_MODEL_NAME = "mock-sentiment-model"
    HOPSWORKS_API_KEY = "mock-hopsworks-key"
    WANDB_API_KEY = "mock-wandb-key"


class MockExternalServices:
    """Mock external services for testing."""
    
    @staticmethod
    def mock_twitter_api() -> AsyncMock:
        """Mock Twitter API client."""
        mock_client = AsyncMock()
        mock_client.get_tweets.return_value = {
            "data": [
                {
                    "id": "1234567890",
                    "text": "Great game by the team today! #NFL",
                    "created_at": "2024-01-15T20:00:00Z",
                    "author_id": "user123",
                    "public_metrics": {
                        "retweet_count": 10,
                        "like_count": 25,
                        "reply_count": 5
                    }
                }
            ]
        }
        return mock_client
    
    @staticmethod
    def mock_espn_api() -> AsyncMock:
        """Mock ESPN API client."""
        mock_client = AsyncMock()
        mock_client.get_games.return_value = {
            "events": [
                {
                    "id": "game_1",
                    "name": "Team A vs Team B",
                    "date": "2024-01-15T20:00:00Z",
                    "status": {
                        "type": {"completed": False}
                    },
                    "competitions": [
                        {
                            "competitors": [
                                {
                                    "team": {
                                        "id": "team_1",
                                        "abbreviation": "TEA",
                                        "displayName": "Team A"
                                    },
                                    "homeAway": "home"
                                },
                                {
                                    "team": {
                                        "id": "team_2",
                                        "abbreviation": "TEB",
                                        "displayName": "Team B"
                                    },
                                    "homeAway": "away"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        return mock_client
    
    @staticmethod
    def mock_betting_api() -> AsyncMock:
        """Mock betting API client."""
        mock_client = AsyncMock()
        mock_client.get_odds.return_value = {
            "games": [
                {
                    "game_id": "game_1",
                    "commence_time": "2024-01-15T20:00:00Z",
                    "bookmakers": [
                        {
                            "key": "draftkings",
                            "title": "DraftKings",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "Team A", "price": -150},
                                        {"name": "Team B", "price": 130}
                                    ]
                                },
                                {
                                    "key": "spreads",
                                    "outcomes": [
                                        {"name": "Team A", "price": -110, "point": -3.5},
                                        {"name": "Team B", "price": -110, "point": 3.5}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        return mock_client
    
    @staticmethod
    def mock_sentiment_model() -> MagicMock:
        """Mock sentiment analysis model."""
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "sentiment": "POSITIVE",
            "confidence": 0.85,
            "scores": {
                "POSITIVE": 0.85,
                "NEGATIVE": 0.10,
                "NEUTRAL": 0.05
            }
        }
        return mock_model


class TestDataFactory:
    """Factory for creating test data."""
    
    @staticmethod
    def create_user_data(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create test user data."""
        data = {
            "id": "user_1",
            "email": "test@example.com",
            "username": "testuser",
            "role": "USER",
            "created_at": "2024-01-15T10:00:00Z",
            "preferences": {
                "favorite_teams": ["team_1"],
                "notifications_enabled": True
            }
        }
        if overrides:
            data.update(overrides)
        return data
    
    @staticmethod
    def create_team_data(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create test team data."""
        data = {
            "id": "team_1",
            "name": "Test Team",
            "abbreviation": "TT",
            "conference": "NFC",
            "division": "North",
            "current_sentiment": 0.75,
            "sentiment_trend": [
                {"timestamp": "2024-01-15T10:00:00Z", "sentiment": 0.70},
                {"timestamp": "2024-01-15T11:00:00Z", "sentiment": 0.75}
            ]
        }
        if overrides:
            data.update(overrides)
        return data
    
    @staticmethod
    def create_sentiment_data(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create test sentiment analysis data."""
        data = {
            "id": "sentiment_1",
            "text": "This team is playing amazingly well!",
            "sentiment": "POSITIVE",
            "confidence": 0.85,
            "team_id": "team_1",
            "player_id": None,
            "game_id": "game_1",
            "source": "TWITTER",
            "timestamp": "2024-01-15T12:00:00Z",
            "context": {
                "hashtags": ["#NFL", "#TeamA"],
                "mentions": ["@TeamA"],
                "betting_related": False
            }
        }
        if overrides:
            data.update(overrides)
        return data
    
    @staticmethod
    def create_game_data(overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create test game data."""
        data = {
            "id": "game_1",
            "home_team_id": "team_1",
            "away_team_id": "team_2",
            "game_date": "2024-01-15T20:00:00Z",
            "week": 1,
            "season": 2024,
            "status": "scheduled",
            "betting_lines": [
                {
                    "sportsbook": "DraftKings",
                    "spread": {"home": -3.5, "away": 3.5},
                    "total": {"over": 45.5, "under": 45.5},
                    "moneyline": {"home": -150, "away": 130},
                    "updated_at": "2024-01-15T18:00:00Z"
                }
            ],
            "sentiment_summary": {
                "home_team_sentiment": 0.75,
                "away_team_sentiment": 0.65,
                "overall_sentiment": 0.70,
                "total_mentions": 150
            }
        }
        if overrides:
            data.update(overrides)
        return data