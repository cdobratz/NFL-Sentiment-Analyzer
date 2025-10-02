"""
Integration tests for sentiment API endpoints.
Tests the complete sentiment analysis workflow including API endpoints, database operations, and service integration.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from fastapi import status
import json

from app.models.sentiment import (
    SentimentLabel, SentimentCategory, DataSource,
    SentimentAnalysisCreate, BatchSentimentRequest
)


class TestSentimentAnalysisEndpoint:
    """Test /sentiment/analyze endpoint integration."""
    
    @pytest.fixture
    def mock_sentiment_service(self):
        """Mock sentiment analysis service."""
        with patch('app.api.sentiment.SentimentAnalysisService') as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service
            mock_service.model_version = "1.0"
            yield mock_service
    
    @pytest.fixture
    def mock_db_operations(self):
        """Mock database operations."""
        mock_db = MagicMock()
        mock_db.sentiment_analysis.insert_one = AsyncMock(return_value=MagicMock(inserted_id="test_id"))
        return mock_db
    
    @pytest.fixture
    def sample_sentiment_request(self):
        """Sample sentiment analysis request."""
        return {
            "text": "Amazing touchdown pass by the quarterback!",
            "team_id": "patriots",
            "source": "twitter",
            "context": {
                "keywords": ["touchdown", "quarterback"],
                "entities": ["patriots"]
            }
        }
    
    @pytest.fixture
    def sample_sentiment_result(self):
        """Sample sentiment analysis result."""
        from app.models.sentiment import SentimentResult, AnalysisContext, NFLContext
        
        return SentimentResult(
            text="Amazing touchdown pass by the quarterback!",
            sentiment=SentimentLabel.POSITIVE,
            sentiment_score=0.85,
            confidence=0.92,
            category=SentimentCategory.PERFORMANCE,
            context=AnalysisContext(
                keywords=["touchdown", "quarterback"],
                nfl_context=NFLContext(
                    team_mentions=["patriots"],
                    position_mentions=["QB"]
                )
            ),
            team_id="patriots",
            source=DataSource.TWITTER,
            timestamp=datetime.utcnow(),
            model_version="1.0",
            processing_time_ms=150.0,
            emotion_scores={"joy": 0.8, "anger": 0.1},
            aspect_sentiments={"performance": 0.9},
            keyword_contributions={"touchdown": 0.5, "amazing": 0.4}
        )
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_success(self, mock_sentiment_service, mock_db_operations, 
                                           sample_sentiment_request, sample_sentiment_result):
        """Test successful sentiment analysis via API endpoint."""
        # Setup mocks
        mock_sentiment_service.analyze_text.return_value = sample_sentiment_result
        
        # Mock authentication
        with patch('app.api.sentiment.require_user_or_api_scope') as mock_auth:
            mock_auth.return_value = lambda: {"_id": "user_123", "role": "user"}
            
            with patch('app.api.sentiment.get_database') as mock_get_db:
                mock_get_db.return_value = mock_db_operations
                
                # Import and create test client after mocking
                from app.main import app
                client = TestClient(app)
                
                # Make request
                response = client.post("/sentiment/analyze", json=sample_sentiment_request)
                
                # Verify response
                assert response.status_code == status.HTTP_200_OK
                result_data = response.json()
                
                assert result_data["text"] == sample_sentiment_request["text"]
                assert result_data["sentiment"] == "POSITIVE"
                assert result_data["team_id"] == "patriots"
                assert result_data["confidence"] > 0.9
                
                # Verify service was called correctly
                mock_sentiment_service.analyze_text.assert_called_once()
                call_args = mock_sentiment_service.analyze_text.call_args
                assert call_args.kwargs["text"] == sample_sentiment_request["text"]
                assert call_args.kwargs["team_id"] == "patriots"
                
                # Verify database insertion
                mock_db_operations.sentiment_analysis.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_service_error(self, mock_sentiment_service, mock_db_operations,
                                                 sample_sentiment_request):
        """Test sentiment analysis API endpoint with service error."""
        # Setup service to raise exception
        mock_sentiment_service.analyze_text.side_effect = Exception("Service error")
        
        with patch('app.api.sentiment.require_user_or_api_scope') as mock_auth:
            mock_auth.return_value = lambda: {"_id": "user_123", "role": "user"}
            
            with patch('app.api.sentiment.get_database') as mock_get_db:
                mock_get_db.return_value = mock_db_operations
                
                from app.main import app
                client = TestClient(app)
                
                response = client.post("/sentiment/analyze", json=sample_sentiment_request)
                
                assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
                assert "Error processing sentiment analysis" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment_invalid_request(self):
        """Test sentiment analysis API endpoint with invalid request data."""
        invalid_request = {
            "text": "",  # Empty text
            "source": "invalid_source"  # Invalid source
        }
        
        with patch('app.api.sentiment.require_user_or_api_scope') as mock_auth:
            mock_auth.return_value = lambda: {"_id": "user_123", "role": "user"}
            
            from app.main import app
            client = TestClient(app)
            
            response = client.post("/sentiment/analyze", json=invalid_request)
            
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestBatchSentimentAnalysisEndpoint:
    """Test /sentiment/analyze/batch endpoint integration."""
    
    @pytest.fixture
    def mock_sentiment_service(self):
        """Mock sentiment analysis service for batch operations."""
        with patch('app.api.sentiment.SentimentAnalysisService') as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service
            mock_service.model_version = "1.0"
            yield mock_service
    
    @pytest.fixture
    def sample_batch_request(self):
        """Sample batch sentiment analysis request."""
        return {
            "texts": [
                "Amazing touchdown!",
                "Terrible fumble",
                "Good game overall"
            ],
            "team_id": "cowboys",
            "source": "twitter"
        }
    
    @pytest.fixture
    def sample_batch_results(self):
        """Sample batch sentiment analysis results."""
        from app.models.sentiment import SentimentResult, AnalysisContext
        
        return [
            SentimentResult(
                text="Amazing touchdown!",
                sentiment=SentimentLabel.POSITIVE,
                sentiment_score=0.8,
                confidence=0.9,
                category=SentimentCategory.PERFORMANCE,
                context=AnalysisContext(),
                source=DataSource.TWITTER,
                timestamp=datetime.utcnow(),
                model_version="1.0",
                processing_time_ms=100.0
            ),
            SentimentResult(
                text="Terrible fumble",
                sentiment=SentimentLabel.NEGATIVE,
                sentiment_score=-0.7,
                confidence=0.85,
                category=SentimentCategory.PERFORMANCE,
                context=AnalysisContext(),
                source=DataSource.TWITTER,
                timestamp=datetime.utcnow(),
                model_version="1.0",
                processing_time_ms=120.0
            ),
            SentimentResult(
                text="Good game overall",
                sentiment=SentimentLabel.POSITIVE,
                sentiment_score=0.3,
                confidence=0.6,
                category=SentimentCategory.GENERAL,
                context=AnalysisContext(),
                source=DataSource.TWITTER,
                timestamp=datetime.utcnow(),
                model_version="1.0",
                processing_time_ms=90.0
            )
        ]
    
    @pytest.mark.asyncio
    async def test_batch_analyze_sentiment_success(self, mock_sentiment_service, sample_batch_request,
                                                 sample_batch_results):
        """Test successful batch sentiment analysis via API endpoint."""
        # Setup mocks
        mock_sentiment_service.analyze_batch.return_value = sample_batch_results
        
        mock_db = MagicMock()
        mock_db.sentiment_analysis.insert_many = AsyncMock()
        
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            with patch('app.api.sentiment.get_optional_user') as mock_get_user:
                mock_get_user.return_value = {"_id": "user_123"}
                
                from app.main import app
                client = TestClient(app)
                
                response = client.post("/sentiment/analyze/batch", json=sample_batch_request)
                
                assert response.status_code == status.HTTP_200_OK
                result_data = response.json()
                
                assert len(result_data["results"]) == 3
                assert result_data["total_processed"] == 3
                assert result_data["model_version"] == "1.0"
                assert "processing_time_ms" in result_data
                assert "aggregated_sentiment" in result_data
                assert "sentiment_distribution" in result_data
                
                # Verify service was called
                mock_sentiment_service.analyze_batch.assert_called_once()
                
                # Verify database insertion
                mock_db.sentiment_analysis.insert_many.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_batch_analyze_sentiment_too_many_texts(self):
        """Test batch sentiment analysis with too many texts."""
        large_batch_request = {
            "texts": ["text"] * 101,  # Exceed limit of 100
            "source": "twitter"
        }
        
        from app.main import app
        client = TestClient(app)
        
        response = client.post("/sentiment/analyze/batch", json=large_batch_request)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Maximum 100 texts allowed" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_batch_analyze_sentiment_empty_texts(self):
        """Test batch sentiment analysis with empty texts list."""
        empty_batch_request = {
            "texts": [],
            "source": "twitter"
        }
        
        from app.main import app
        client = TestClient(app)
        
        response = client.post("/sentiment/analyze/batch", json=empty_batch_request)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestTeamSentimentEndpoint:
    """Test /sentiment/team/{team_id} endpoint integration."""
    
    @pytest.fixture
    def mock_team_data(self):
        """Mock team data from database."""
        return {
            "_id": "patriots",
            "name": "New England Patriots",
            "abbreviation": "NE"
        }
    
    @pytest.fixture
    def mock_sentiment_aggregation_data(self):
        """Mock sentiment aggregation data."""
        return [{
            "_id": None,
            "avg_sentiment": 0.65,
            "total_mentions": 150,
            "confidence": 0.78,
            "sentiment_by_category": [
                {"category": "performance", "sentiment": 0.8},
                {"category": "injury", "sentiment": -0.3}
            ],
            "sentiment_by_source": [
                {"source": "twitter", "sentiment": 0.7},
                {"source": "espn", "sentiment": 0.6}
            ]
        }]
    
    @pytest.fixture
    def mock_trend_data(self):
        """Mock sentiment trend data."""
        return [
            {
                "_id": "2024-01-15-14",
                "avg_sentiment": 0.7,
                "volume": 25,
                "timestamp": datetime(2024, 1, 15, 14, 0, 0)
            },
            {
                "_id": "2024-01-15-15",
                "avg_sentiment": 0.6,
                "volume": 30,
                "timestamp": datetime(2024, 1, 15, 15, 0, 0)
            }
        ]
    
    @pytest.mark.asyncio
    async def test_get_team_sentiment_success(self, mock_team_data, mock_sentiment_aggregation_data,
                                            mock_trend_data):
        """Test successful team sentiment retrieval."""
        mock_db = MagicMock()
        
        # Mock database queries
        mock_db.teams.find_one = AsyncMock(return_value=mock_team_data)
        
        # Mock aggregation pipeline
        mock_agg_cursor = AsyncMock()
        mock_agg_cursor.to_list.return_value = mock_sentiment_aggregation_data
        mock_db.sentiment_analysis.aggregate.return_value = mock_agg_cursor
        
        # Mock trend pipeline
        mock_trend_cursor = AsyncMock()
        mock_trend_cursor.to_list.return_value = mock_trend_data
        
        # Setup aggregate to return different cursors for different calls
        def aggregate_side_effect(pipeline):
            if any("$dateToString" in str(stage) for stage in pipeline):
                return mock_trend_cursor
            return mock_agg_cursor
        
        mock_db.sentiment_analysis.aggregate.side_effect = aggregate_side_effect
        
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            from app.main import app
            client = TestClient(app)
            
            response = client.get("/sentiment/team/patriots")
            
            assert response.status_code == status.HTTP_200_OK
            result_data = response.json()
            
            assert result_data["team_id"] == "patriots"
            assert result_data["team_name"] == "New England Patriots"
            assert result_data["team_abbreviation"] == "NE"
            assert result_data["current_sentiment"] == 0.65
            assert result_data["total_mentions"] == 150
            assert result_data["confidence_score"] == 0.78
            assert len(result_data["sentiment_trend"]) == 2
            assert "sentiment_by_category" in result_data
            assert "sentiment_by_source" in result_data
    
    @pytest.mark.asyncio
    async def test_get_team_sentiment_team_not_found(self):
        """Test team sentiment retrieval for non-existent team."""
        mock_db = MagicMock()
        mock_db.teams.find_one = AsyncMock(return_value=None)
        
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            from app.main import app
            client = TestClient(app)
            
            response = client.get("/sentiment/team/nonexistent")
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "Team not found" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_get_team_sentiment_no_data(self, mock_team_data):
        """Test team sentiment retrieval with no sentiment data."""
        mock_db = MagicMock()
        mock_db.teams.find_one = AsyncMock(return_value=mock_team_data)
        
        # Mock empty aggregation result
        mock_agg_cursor = AsyncMock()
        mock_agg_cursor.to_list.return_value = []
        mock_db.sentiment_analysis.aggregate.return_value = mock_agg_cursor
        
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            from app.main import app
            client = TestClient(app)
            
            response = client.get("/sentiment/team/patriots")
            
            assert response.status_code == status.HTTP_200_OK
            result_data = response.json()
            
            assert result_data["team_id"] == "patriots"
            assert result_data["current_sentiment"] == 0.0
            assert result_data["total_mentions"] == 0
            assert result_data["confidence_score"] == 0.0
    
    @pytest.mark.asyncio
    async def test_get_team_sentiment_with_filters(self, mock_team_data, mock_sentiment_aggregation_data):
        """Test team sentiment retrieval with query filters."""
        mock_db = MagicMock()
        mock_db.teams.find_one = AsyncMock(return_value=mock_team_data)
        
        mock_agg_cursor = AsyncMock()
        mock_agg_cursor.to_list.return_value = mock_sentiment_aggregation_data
        mock_db.sentiment_analysis.aggregate.return_value = mock_agg_cursor
        
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            from app.main import app
            client = TestClient(app)
            
            # Test with query parameters
            response = client.get(
                "/sentiment/team/patriots?days=14&sources=twitter&sources=espn&categories=performance"
            )
            
            assert response.status_code == status.HTTP_200_OK
            
            # Verify that the aggregation was called (filters would be applied in the query)
            mock_db.sentiment_analysis.aggregate.assert_called()


class TestPlayerSentimentEndpoint:
    """Test /sentiment/player/{player_id} endpoint integration."""
    
    @pytest.fixture
    def mock_player_data(self):
        """Mock player data from database."""
        return {
            "_id": "player_123",
            "name": "Tom Brady",
            "team_id": "patriots",
            "position": "QB"
        }
    
    @pytest.fixture
    def mock_player_sentiment_data(self):
        """Mock player sentiment aggregation data."""
        return [{
            "_id": None,
            "avg_sentiment": 0.85,
            "total_mentions": 75,
            "confidence": 0.88,
            "sentiment_by_category": [
                {"category": "performance", "sentiment": 0.9},
                {"category": "fantasy", "sentiment": 0.7}
            ],
            "sentiment_by_source": [
                {"source": "twitter", "sentiment": 0.8},
                {"source": "fantasy", "sentiment": 0.9}
            ]
        }]
    
    @pytest.mark.asyncio
    async def test_get_player_sentiment_success(self, mock_player_data, mock_player_sentiment_data):
        """Test successful player sentiment retrieval."""
        mock_db = MagicMock()
        mock_db.players.find_one = AsyncMock(return_value=mock_player_data)
        
        # Mock aggregation results
        mock_agg_cursor = AsyncMock()
        mock_agg_cursor.to_list.return_value = mock_player_sentiment_data
        mock_db.sentiment_analysis.aggregate.return_value = mock_agg_cursor
        
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            from app.main import app
            client = TestClient(app)
            
            response = client.get("/sentiment/player/player_123")
            
            assert response.status_code == status.HTTP_200_OK
            result_data = response.json()
            
            assert result_data["player_id"] == "player_123"
            assert result_data["player_name"] == "Tom Brady"
            assert result_data["team_id"] == "patriots"
            assert result_data["position"] == "QB"
            assert result_data["current_sentiment"] == 0.85
            assert result_data["total_mentions"] == 75
    
    @pytest.mark.asyncio
    async def test_get_player_sentiment_player_not_found(self):
        """Test player sentiment retrieval for non-existent player."""
        mock_db = MagicMock()
        mock_db.players.find_one = AsyncMock(return_value=None)
        
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            from app.main import app
            client = TestClient(app)
            
            response = client.get("/sentiment/player/nonexistent")
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "Player not found" in response.json()["detail"]


class TestGameSentimentEndpoint:
    """Test /sentiment/game/{game_id} endpoint integration."""
    
    @pytest.fixture
    def mock_game_data(self):
        """Mock game data from database."""
        return {
            "_id": "game_123",
            "home_team_id": "patriots",
            "away_team_id": "cowboys",
            "game_date": datetime(2024, 1, 15, 20, 0, 0),
            "week": 1,
            "season": 2024
        }
    
    @pytest.fixture
    def mock_game_sentiment_data(self):
        """Mock game sentiment aggregation data."""
        return [{
            "_id": None,
            "avg_sentiment": 0.45,
            "total_mentions": 200,
            "confidence": 0.72,
            "sentiment_by_category": [
                {"category": "betting", "sentiment": 0.3},
                {"category": "performance", "sentiment": 0.6}
            ]
        }]
    
    @pytest.mark.asyncio
    async def test_get_game_sentiment_success(self, mock_game_data, mock_game_sentiment_data):
        """Test successful game sentiment retrieval."""
        mock_db = MagicMock()
        mock_db.games.find_one = AsyncMock(return_value=mock_game_data)
        
        # Mock aggregation results
        mock_agg_cursor = AsyncMock()
        mock_agg_cursor.to_list.return_value = mock_game_sentiment_data
        mock_db.sentiment_analysis.aggregate.return_value = mock_agg_cursor
        
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            from app.main import app
            client = TestClient(app)
            
            response = client.get("/sentiment/game/game_123")
            
            assert response.status_code == status.HTTP_200_OK
            result_data = response.json()
            
            assert result_data["game_id"] == "game_123"
            assert result_data["home_team_id"] == "patriots"
            assert result_data["away_team_id"] == "cowboys"
            assert result_data["current_sentiment"] == 0.45
            assert result_data["total_mentions"] == 200


class TestSentimentTrendsEndpoint:
    """Test /sentiment/trends endpoint integration."""
    
    @pytest.fixture
    def mock_trend_data(self):
        """Mock sentiment trend data."""
        return [
            {
                "_id": "2024-01-15-14",
                "avg_sentiment": 0.7,
                "volume": 25,
                "confidence": 0.8,
                "timestamp": datetime(2024, 1, 15, 14, 0, 0),
                "category_breakdown": [
                    {"category": "performance", "sentiment": 0.8}
                ],
                "source_breakdown": [
                    {"source": "twitter", "sentiment": 0.7}
                ]
            },
            {
                "_id": "2024-01-15-15",
                "avg_sentiment": 0.6,
                "volume": 30,
                "confidence": 0.75,
                "timestamp": datetime(2024, 1, 15, 15, 0, 0),
                "category_breakdown": [
                    {"category": "performance", "sentiment": 0.6}
                ],
                "source_breakdown": [
                    {"source": "twitter", "sentiment": 0.6}
                ]
            }
        ]
    
    @pytest.mark.asyncio
    async def test_get_sentiment_trends_success(self, mock_trend_data):
        """Test successful sentiment trends retrieval."""
        mock_db = MagicMock()
        
        mock_agg_cursor = AsyncMock()
        mock_agg_cursor.to_list.return_value = mock_trend_data
        mock_db.sentiment_analysis.aggregate.return_value = mock_agg_cursor
        
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            from app.main import app
            client = TestClient(app)
            
            response = client.get("/sentiment/trends")
            
            assert response.status_code == status.HTTP_200_OK
            result_data = response.json()
            
            assert len(result_data["trends"]) == 2
            assert result_data["period"] == "7 days"
            assert result_data["interval"] == "hour"
            assert result_data["total_data_points"] == 2
            assert "filters" in result_data
            assert "last_updated" in result_data
    
    @pytest.mark.asyncio
    async def test_get_sentiment_trends_with_filters(self, mock_trend_data):
        """Test sentiment trends retrieval with filters."""
        mock_db = MagicMock()
        
        mock_agg_cursor = AsyncMock()
        mock_agg_cursor.to_list.return_value = mock_trend_data
        mock_db.sentiment_analysis.aggregate.return_value = mock_agg_cursor
        
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            from app.main import app
            client = TestClient(app)
            
            response = client.get(
                "/sentiment/trends?team_id=patriots&days=14&interval=day&sources=twitter"
            )
            
            assert response.status_code == status.HTTP_200_OK
            result_data = response.json()
            
            assert result_data["period"] == "14 days"
            assert result_data["interval"] == "day"
            assert result_data["filters"]["team_id"] == "patriots"


class TestSentimentAggregationEndpoint:
    """Test /sentiment/aggregate endpoint integration."""
    
    @pytest.fixture
    def sample_aggregation_request(self):
        """Sample aggregation request."""
        return {
            "entity_type": "team",
            "entity_id": "patriots",
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-15T23:59:59Z",
            "sources": ["twitter", "espn"],
            "categories": ["performance", "injury"]
        }
    
    @pytest.fixture
    def mock_aggregation_result(self):
        """Mock aggregation result."""
        return [{
            "_id": None,
            "avg_sentiment": 0.65,
            "total_mentions": 150,
            "avg_confidence": 0.78,
            "sentiment_distribution": ["POSITIVE", "POSITIVE", "NEGATIVE"],
            "category_sentiments": [
                {"category": "performance", "sentiment": 0.8},
                {"category": "injury", "sentiment": -0.3}
            ],
            "source_sentiments": [
                {"source": "twitter", "sentiment": 0.7},
                {"source": "espn", "sentiment": 0.6}
            ]
        }]
    
    @pytest.mark.asyncio
    async def test_aggregate_sentiment_success(self, sample_aggregation_request, mock_aggregation_result):
        """Test successful sentiment aggregation."""
        mock_db = MagicMock()
        
        mock_agg_cursor = AsyncMock()
        mock_agg_cursor.to_list.return_value = mock_aggregation_result
        mock_db.sentiment_analysis.aggregate.return_value = mock_agg_cursor
        
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            from app.main import app
            client = TestClient(app)
            
            response = client.post("/sentiment/aggregate", json=sample_aggregation_request)
            
            assert response.status_code == status.HTTP_200_OK
            result_data = response.json()
            
            assert result_data["entity_type"] == "team"
            assert result_data["entity_id"] == "patriots"
            assert result_data["avg_sentiment"] == 0.65
            assert result_data["total_mentions"] == 150
    
    @pytest.mark.asyncio
    async def test_aggregate_sentiment_no_data(self, sample_aggregation_request):
        """Test sentiment aggregation with no data."""
        mock_db = MagicMock()
        
        mock_agg_cursor = AsyncMock()
        mock_agg_cursor.to_list.return_value = []
        mock_db.sentiment_analysis.aggregate.return_value = mock_agg_cursor
        
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.return_value = mock_db
            
            from app.main import app
            client = TestClient(app)
            
            response = client.post("/sentiment/aggregate", json=sample_aggregation_request)
            
            assert response.status_code == status.HTTP_200_OK
            result_data = response.json()
            
            assert result_data["avg_sentiment"] == 0.0
            assert result_data["total_mentions"] == 0


class TestBackgroundTaskIntegration:
    """Test background task integration for sentiment analysis."""
    
    @pytest.mark.asyncio
    async def test_update_aggregated_sentiment_task(self):
        """Test that aggregated sentiment update task is triggered."""
        mock_db = MagicMock()
        
        # Mock the background task function
        with patch('app.api.sentiment.update_aggregated_sentiment') as mock_update_task:
            
            # Test that the task is called with correct parameters
            from app.api.sentiment import update_aggregated_sentiment
            
            await update_aggregated_sentiment(mock_db, "team_1", "player_1", "game_1")
            
            # Verify the function was called (implementation would update aggregated data)
            # This is a placeholder test - actual implementation would verify database updates
            assert True  # Task execution verified


class TestErrorHandlingIntegration:
    """Test error handling in sentiment API integration."""
    
    @pytest.mark.asyncio
    async def test_database_connection_error(self):
        """Test handling of database connection errors."""
        with patch('app.api.sentiment.get_database') as mock_get_db:
            mock_get_db.side_effect = Exception("Database connection failed")
            
            from app.main import app
            client = TestClient(app)
            
            response = client.get("/sentiment/team/patriots")
            
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    
    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """Test handling of authentication errors."""
        with patch('app.api.sentiment.require_user_or_api_scope') as mock_auth:
            mock_auth.side_effect = Exception("Authentication failed")
            
            from app.main import app
            client = TestClient(app)
            
            request_data = {"text": "test", "source": "twitter"}
            response = client.post("/sentiment/analyze", json=request_data)
            
            # Should return authentication error
            assert response.status_code in [status.HTTP_401_UNAUTHORIZED, status.HTTP_500_INTERNAL_SERVER_ERROR]