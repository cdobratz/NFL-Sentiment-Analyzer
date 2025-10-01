import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock


# Test settings
class TestSettings:
    MONGODB_URL = "mongodb://localhost:27017"
    DATABASE_NAME = "nfl_analyzer_test"
    REDIS_URL = "redis://localhost:6379/1"
    SECRET_KEY = "test-secret-key"
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings():
    """Provide test settings."""
    return TestSettings()


# Database fixtures will be added when needed for integration tests


@pytest.fixture
def mock_db():
    """Mock database for unit tests."""
    mock_db = MagicMock()
    mock_db.users = MagicMock()
    mock_db.sentiment_analyses = MagicMock()
    mock_db.teams = MagicMock()
    mock_db.games = MagicMock()
    mock_db.players = MagicMock()
    return mock_db


@pytest.fixture
def mock_redis():
    """Mock Redis client for unit tests."""
    mock_redis = AsyncMock()
    return mock_redis


# Note: FastAPI app fixtures will be added when needed for integration tests
# For now, we focus on unit tests that don't require the full app setup


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "username": "testuser",
        "password": "testpassword123",
        "role": "USER"
    }


@pytest.fixture
def sample_team_data():
    """Sample team data for testing."""
    return {
        "id": "team_1",
        "name": "Test Team",
        "abbreviation": "TT",
        "conference": "NFC",
        "division": "North",
        "current_sentiment": 0.75
    }


@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment analysis data for testing."""
    return {
        "text": "This team is playing amazingly well!",
        "sentiment": "POSITIVE",
        "confidence": 0.85,
        "team_id": "team_1",
        "source": "TWITTER",
        "context": {
            "game_id": "game_1",
            "player_mentions": ["player_1"]
        }
    }


@pytest.fixture
def sample_game_data():
    """Sample game data for testing."""
    return {
        "id": "game_1",
        "home_team_id": "team_1",
        "away_team_id": "team_2",
        "game_date": "2024-01-15T20:00:00Z",
        "week": 1,
        "betting_lines": [
            {
                "sportsbook": "DraftKings",
                "spread": -3.5,
                "over_under": 45.5,
                "moneyline": {"home": -150, "away": +130}
            }
        ]
    }


@pytest.fixture
def auth_headers(sample_user_data):
    """Generate auth headers for testing."""
    # This would normally create a real JWT token
    # For now, we'll mock it
    return {"Authorization": "Bearer test-token"}


# Mock external API responses
@pytest.fixture
def mock_twitter_response():
    """Mock Twitter API response."""
    return {
        "data": [
            {
                "id": "1234567890",
                "text": "Great game by the team today!",
                "created_at": "2024-01-15T20:00:00Z",
                "author_id": "user123"
            }
        ]
    }


@pytest.fixture
def mock_espn_response():
    """Mock ESPN API response."""
    return {
        "events": [
            {
                "id": "game_1",
                "name": "Team A vs Team B",
                "date": "2024-01-15T20:00:00Z",
                "competitions": [
                    {
                        "competitors": [
                            {"team": {"abbreviation": "TEA", "displayName": "Team A"}},
                            {"team": {"abbreviation": "TEB", "displayName": "Team B"}}
                        ]
                    }
                ]
            }
        ]
    }


@pytest.fixture
def mock_betting_response():
    """Mock betting API response."""
    return {
        "games": [
            {
                "game_id": "game_1",
                "sportsbooks": [
                    {
                        "name": "DraftKings",
                        "odds": {
                            "spread": {"home": -3.5, "away": 3.5},
                            "total": {"over": 45.5, "under": 45.5},
                            "moneyline": {"home": -150, "away": 130}
                        }
                    }
                ]
            }
        ]
    }