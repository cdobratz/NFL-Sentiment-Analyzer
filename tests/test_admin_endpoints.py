"""
Integration tests for admin API endpoints.
Tests the complete admin functionality through FastAPI endpoints.
"""
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI, status
from datetime import datetime, timedelta
from bson import ObjectId
from jose import jwt

# Mock external dependencies before importing
import sys
from unittest.mock import MagicMock

# Mock transformers and other ML libraries
sys.modules['transformers'] = MagicMock()
sys.modules['hopsworks'] = MagicMock()
sys.modules['wandb'] = MagicMock()
sys.modules['psutil'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Mock settings before importing app modules
with patch.dict('os.environ', {
    'SECRET_KEY': 'test-secret-key',
    'MONGODB_URL': 'mongodb://test:27017',
    'DATABASE_NAME': 'test_db'
}):
    from app.api.admin import router as admin_router
    from app.core.config import Settings


# Test app setup
def create_test_app():
    """Create test FastAPI app with admin router."""
    app = FastAPI()
    app.include_router(admin_router)
    return app


@pytest.fixture
def test_app():
    """Create test app fixture."""
    return create_test_app()


@pytest.fixture
def client(test_app):
    """Create test client fixture."""
    return TestClient(test_app)


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    settings = Settings(
        secret_key="test-secret-key",
        algorithm="HS256",
        access_token_expire_minutes=30,
        mongodb_url="mongodb://test",
        database_name="test_db"
    )
    return settings


@pytest.fixture
def admin_user():
    """Sample admin user data for testing."""
    return {
        "_id": ObjectId(),
        "email": "admin@example.com",
        "username": "adminuser",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "role": "admin",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "preferences": {}
    }


@pytest.fixture
def regular_user():
    """Sample regular user data for testing."""
    return {
        "_id": ObjectId(),
        "email": "user@example.com",
        "username": "regularuser",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "role": "user",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "preferences": {}
    }


def create_auth_headers(user_id: str, settings) -> dict:
    """Create authentication headers for testing."""
    token_data = {"sub": user_id}
    token = jwt.encode(token_data, settings.secret_key, algorithm=settings.algorithm)
    return {"Authorization": f"Bearer {token}"}


class TestUserManagementEndpoints:
    """Test admin user management endpoints."""
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.admin.get_database')
    def test_get_users_success(self, mock_admin_db, mock_dep_settings, mock_get_redis, mock_dep_db, client, admin_user, mock_settings):
        """Test successful retrieval of users list."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        # Mock dependency database
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = admin_user
        mock_dep_db.return_value = mock_db_dep
        
        # Mock Redis
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        # Mock admin endpoint database
        mock_db_admin = MagicMock()
        
        # Create mock cursor for users
        sample_users = [
            {
                "_id": ObjectId(),
                "email": "user1@example.com",
                "username": "user1",
                "role": "user",
                "is_active": True,
                "created_at": datetime.utcnow(),
                "preferences": {}
            },
            {
                "_id": ObjectId(),
                "email": "user2@example.com",
                "username": "user2",
                "role": "user",
                "is_active": True,
                "created_at": datetime.utcnow(),
                "preferences": {}
            }
        ]
        
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__.return_value = iter(sample_users)
        mock_db_admin.users.find.return_value = mock_cursor
        mock_admin_db.return_value = mock_db_admin
        
        headers = create_auth_headers(str(admin_user["_id"]), mock_dep_settings)
        response = client.get("/admin/users", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["email"] == "user1@example.com"
        assert data[1]["email"] == "user2@example.com"
        
        # Verify no sensitive data is returned
        for user in data:
            assert "hashed_password" not in user
            assert "id" in user
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    def test_get_users_non_admin_denied(self, mock_dep_settings, mock_get_redis, mock_dep_db, client, regular_user, mock_settings):
        """Test that non-admin users cannot access user list."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = regular_user
        mock_dep_db.return_value = mock_db_dep
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        headers = create_auth_headers(str(regular_user["_id"]), mock_dep_settings)
        response = client.get("/admin/users", headers=headers)
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "Not enough permissions" in response.json()["detail"]
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.admin.get_database')
    def test_get_user_by_id_success(self, mock_admin_db, mock_dep_settings, mock_get_redis, mock_dep_db, client, admin_user, mock_settings):
        """Test successful retrieval of specific user by ID."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = admin_user
        mock_dep_db.return_value = mock_db_dep
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        target_user = {
            "_id": "user123",
            "email": "target@example.com",
            "username": "targetuser",
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "preferences": {}
        }
        
        mock_db_admin = MagicMock()
        mock_db_admin.users.find_one.return_value = target_user
        mock_admin_db.return_value = mock_db_admin
        
        headers = create_auth_headers(str(admin_user["_id"]), mock_dep_settings)
        response = client.get("/admin/users/user123", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == "target@example.com"
        assert data["username"] == "targetuser"
        assert "hashed_password" not in data
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.admin.get_database')
    def test_get_user_by_id_not_found(self, mock_admin_db, mock_dep_settings, mock_get_redis, mock_dep_db, client, admin_user, mock_settings):
        """Test user not found error."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = admin_user
        mock_dep_db.return_value = mock_db_dep
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        mock_db_admin = MagicMock()
        mock_db_admin.users.find_one.return_value = None
        mock_admin_db.return_value = mock_db_admin
        
        headers = create_auth_headers(str(admin_user["_id"]), mock_dep_settings)
        response = client.get("/admin/users/nonexistent", headers=headers)
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "User not found" in response.json()["detail"]
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.admin.get_database')
    def test_deactivate_user_success(self, mock_admin_db, mock_dep_settings, mock_get_redis, mock_dep_db, client, admin_user, mock_settings):
        """Test successful user deactivation."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = admin_user
        mock_dep_db.return_value = mock_db_dep
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        mock_db_admin = MagicMock()
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_db_admin.users.update_one.return_value = mock_result
        mock_admin_db.return_value = mock_db_admin
        
        headers = create_auth_headers(str(admin_user["_id"]), mock_dep_settings)
        response = client.put("/admin/users/user123/deactivate", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "User deactivated successfully" in data["message"]
        
        # Verify database call
        mock_db_admin.users.update_one.assert_called_once()
        call_args = mock_db_admin.users.update_one.call_args
        assert call_args[0][0] == {"_id": "user123"}
        assert call_args[0][1]["$set"]["is_active"] is False
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.admin.get_database')
    def test_activate_user_success(self, mock_admin_db, mock_dep_settings, mock_get_redis, mock_dep_db, client, admin_user, mock_settings):
        """Test successful user activation."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = admin_user
        mock_dep_db.return_value = mock_db_dep
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        mock_db_admin = MagicMock()
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_db_admin.users.update_one.return_value = mock_result
        mock_admin_db.return_value = mock_db_admin
        
        headers = create_auth_headers(str(admin_user["_id"]), mock_dep_settings)
        response = client.put("/admin/users/user123/activate", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "User activated successfully" in data["message"]
        
        # Verify database call
        mock_db_admin.users.update_one.assert_called_once()
        call_args = mock_db_admin.users.update_one.call_args
        assert call_args[0][0] == {"_id": "user123"}
        assert call_args[0][1]["$set"]["is_active"] is True


class TestSystemHealthEndpoint:
    """Test system health monitoring endpoint."""
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.admin.get_database')
    @patch('app.api.admin.get_redis')
    @patch('app.api.admin.mlops_service')
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_get_system_health_success(self, mock_disk, mock_memory, mock_cpu, mock_mlops, mock_admin_redis, mock_admin_db, mock_dep_settings, mock_get_redis, mock_dep_db, client, admin_user, mock_settings):
        """Test successful system health check."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = admin_user
        mock_dep_db.return_value = mock_db_dep
        
        mock_redis_dep = AsyncMock()
        mock_redis_dep.get.return_value = None
        mock_get_redis.return_value = mock_redis_dep
        
        # Mock MongoDB health
        mock_db_admin = MagicMock()
        mock_db_admin.command = AsyncMock()
        mock_db_admin.command.side_effect = [
            "pong",  # ping response
            {  # dbStats response
                "dataSize": 1024 * 1024 * 100,  # 100MB
                "collections": 5,
                "indexes": 10
            }
        ]
        mock_admin_db.return_value = mock_db_admin
        
        # Mock Redis health
        mock_redis_admin = AsyncMock()
        mock_redis_admin.ping.return_value = "PONG"
        mock_redis_admin.info.return_value = {
            "used_memory": 1024 * 1024 * 50,  # 50MB
            "connected_clients": 10,
            "total_commands_processed": 1000
        }
        mock_admin_redis.return_value = mock_redis_admin
        
        # Mock MLOps service
        mock_mlops.get_service_status.return_value = {
            "initialized": True,
            "services": {
                "hopsworks": "healthy",
                "huggingface": "healthy",
                "wandb": "healthy"
            }
        }
        
        # Mock system resources
        mock_cpu.return_value = 45.5
        mock_memory.return_value = MagicMock(percent=60.0, available=8 * 1024**3)  # 8GB available
        mock_disk.return_value = MagicMock(percent=70.0, free=100 * 1024**3)  # 100GB free
        
        headers = create_auth_headers(str(admin_user["_id"]), mock_dep_settings)
        response = client.get("/admin/system-health", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["overall_status"] == "healthy"
        assert "services" in data
        assert "system_resources" in data
        
        # Check MongoDB status
        assert data["services"]["mongodb"]["status"] == "healthy"
        assert data["services"]["mongodb"]["database_size_mb"] == 100.0
        
        # Check Redis status
        assert data["services"]["redis"]["status"] == "healthy"
        assert data["services"]["redis"]["memory_used_mb"] == 50.0
        
        # Check MLOps status
        assert data["services"]["mlops"]["status"] == "healthy"
        assert data["services"]["mlops"]["initialized"] is True
        
        # Check system resources
        assert data["system_resources"]["cpu_usage_percent"] == 45.5
        assert data["system_resources"]["memory_usage_percent"] == 60.0
        assert data["system_resources"]["disk_usage_percent"] == 70.0
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.admin.get_database')
    def test_get_system_health_mongodb_failure(self, mock_admin_db, mock_dep_settings, mock_get_redis, mock_dep_db, client, admin_user, mock_settings):
        """Test system health with MongoDB failure."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = admin_user
        mock_dep_db.return_value = mock_db_dep
        
        mock_redis_dep = AsyncMock()
        mock_redis_dep.get.return_value = None
        mock_get_redis.return_value = mock_redis_dep
        
        # Mock MongoDB failure
        mock_db_admin = MagicMock()
        mock_db_admin.command = AsyncMock(side_effect=Exception("Connection failed"))
        mock_admin_db.return_value = mock_db_admin
        
        headers = create_auth_headers(str(admin_user["_id"]), mock_dep_settings)
        
        with patch('app.api.admin.get_redis') as mock_admin_redis, \
             patch('app.api.admin.mlops_service') as mock_mlops, \
             patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_admin_redis.return_value = None
            mock_mlops.get_service_status.return_value = {"initialized": False}
            mock_cpu.return_value = 30.0
            mock_memory.return_value = MagicMock(percent=50.0, available=8 * 1024**3)
            mock_disk.return_value = MagicMock(percent=60.0, free=100 * 1024**3)
            
            response = client.get("/admin/system-health", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["overall_status"] == "degraded"
        assert data["services"]["mongodb"]["status"] == "unhealthy"
        assert "error" in data["services"]["mongodb"]


class TestAnalyticsEndpoint:
    """Test admin analytics endpoint."""
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.admin.get_database')
    def test_get_analytics_success(self, mock_admin_db, mock_dep_settings, mock_get_redis, mock_dep_db, client, admin_user, mock_settings):
        """Test successful analytics retrieval."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = admin_user
        mock_dep_db.return_value = mock_db_dep
        
        mock_redis_dep = AsyncMock()
        mock_redis_dep.get.return_value = None
        mock_get_redis.return_value = mock_redis_dep
        
        # Mock database analytics queries
        mock_db_admin = MagicMock()
        
        # Mock user counts
        mock_db_admin.users.count_documents.side_effect = [100, 85, 15]  # total, active, new
        
        # Mock daily registrations aggregation
        daily_reg_cursor = AsyncMock()
        daily_reg_cursor.__aiter__.return_value = iter([
            {"_id": "2024-01-01", "count": 5},
            {"_id": "2024-01-02", "count": 10}
        ])
        
        # Mock sentiment analysis counts
        mock_db_admin.sentiment_analyses.count_documents.side_effect = [1000, 150]  # total, recent
        
        # Mock sentiment distribution aggregation
        sentiment_dist_cursor = AsyncMock()
        sentiment_dist_cursor.__aiter__.return_value = iter([
            {"_id": "positive", "count": 80, "avg_confidence": 0.85},
            {"_id": "negative", "count": 40, "avg_confidence": 0.78},
            {"_id": "neutral", "count": 30, "avg_confidence": 0.72}
        ])
        
        # Mock daily analyses aggregation
        daily_analyses_cursor = AsyncMock()
        daily_analyses_cursor.__aiter__.return_value = iter([
            {"_id": "2024-01-01", "count": 50},
            {"_id": "2024-01-02", "count": 75}
        ])
        
        # Mock team sentiment aggregation
        team_sentiment_cursor = AsyncMock()
        team_sentiment_cursor.__aiter__.return_value = iter([
            {
                "_id": "team_1",
                "total_analyses": 100,
                "avg_sentiment_score": 0.75,
                "positive_count": 60,
                "negative_count": 25,
                "neutral_count": 15
            }
        ])
        
        # Set up aggregate method to return appropriate cursors
        def mock_aggregate(pipeline):
            if any("created_at" in str(stage) for stage in pipeline):
                return daily_reg_cursor
            elif any("sentiment" in str(stage) for stage in pipeline):
                if any("team_id" in str(stage) for stage in pipeline):
                    return team_sentiment_cursor
                elif any("avg_confidence" in str(stage) for stage in pipeline):
                    return sentiment_dist_cursor
                else:
                    return daily_analyses_cursor
            return AsyncMock()
        
        mock_db_admin.users.aggregate = mock_aggregate
        mock_db_admin.sentiment_analyses.aggregate = mock_aggregate
        
        # Mock other collections to return empty cursors
        empty_cursor = AsyncMock()
        empty_cursor.__aiter__.return_value = iter([])
        mock_db_admin.performance_metrics.aggregate = lambda x: empty_cursor
        mock_db_admin.api_logs.aggregate = lambda x: empty_cursor
        mock_db_admin.error_logs.aggregate = lambda x: empty_cursor
        
        mock_admin_db.return_value = mock_db_admin
        
        headers = create_auth_headers(str(admin_user["_id"]), mock_dep_settings)
        response = client.get("/admin/analytics?days=7", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["period"] == "7 days"
        assert "users" in data
        assert "sentiment_analyses" in data
        
        # Check user statistics
        assert data["users"]["total"] == 100
        assert data["users"]["active"] == 85
        assert data["users"]["new"] == 15
        
        # Check sentiment analysis statistics
        assert data["sentiment_analyses"]["total"] == 1000
        assert data["sentiment_analyses"]["recent"] == 150


class TestModelRetrainingEndpoint:
    """Test model retraining endpoint."""
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.admin.get_database')
    @patch('app.api.admin.mlops_service')
    def test_retrain_models_success(self, mock_mlops, mock_admin_db, mock_dep_settings, mock_get_redis, mock_dep_db, client, admin_user, mock_settings):
        """Test successful model retraining trigger."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = admin_user
        mock_dep_db.return_value = mock_db_dep
        
        mock_redis_dep = AsyncMock()
        mock_redis_dep.get.return_value = None
        mock_get_redis.return_value = mock_redis_dep
        
        # Mock MLOps service
        mock_mlops.initialized = True
        mock_experiment_run = MagicMock()
        mock_experiment_run.run_id = "run_123"
        mock_experiment_run.experiment_id = "exp_456"
        mock_experiment_run.status.value = "running"
        mock_experiment_run.started_at = datetime.utcnow()
        mock_mlops.trigger_retraining.return_value = mock_experiment_run
        
        # Mock database
        mock_db_admin = MagicMock()
        mock_db_admin.ml_jobs.insert_one = AsyncMock()
        mock_admin_db.return_value = mock_db_admin
        
        headers = create_auth_headers(str(admin_user["_id"]), mock_dep_settings)
        
        retraining_request = {
            "model_name": "sentiment_base",
            "trigger_reason": "manual_admin_trigger",
            "auto_deploy": False
        }
        
        response = client.post("/admin/retrain-models", json=retraining_request, headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "Model retraining job started successfully" in data["message"]
        assert data["job_id"] == "run_123"
        assert data["experiment_id"] == "exp_456"
        assert data["model_name"] == "sentiment_base"
        assert data["status"] == "running"
        
        # Verify MLOps service was called
        mock_mlops.trigger_retraining.assert_called_once()
        
        # Verify job was stored in database
        mock_db_admin.ml_jobs.insert_one.assert_called_once()
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.admin.mlops_service')
    def test_retrain_models_mlops_failure(self, mock_mlops, mock_dep_settings, mock_get_redis, mock_dep_db, client, admin_user, mock_settings):
        """Test model retraining with MLOps service failure."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = admin_user
        mock_dep_db.return_value = mock_db_dep
        
        mock_redis_dep = AsyncMock()
        mock_redis_dep.get.return_value = None
        mock_get_redis.return_value = mock_redis_dep
        
        # Mock MLOps service failure
        mock_mlops.initialized = False
        mock_mlops.initialize = AsyncMock(side_effect=Exception("MLOps initialization failed"))
        
        headers = create_auth_headers(str(admin_user["_id"]), mock_dep_settings)
        response = client.post("/admin/retrain-models", headers=headers)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Failed to trigger model retraining" in response.json()["detail"]


class TestCacheClearEndpoint:
    """Test cache clearing endpoint."""
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.admin.get_redis')
    @patch('app.api.admin.mlops_service')
    def test_clear_cache_redis_success(self, mock_mlops, mock_admin_redis, mock_dep_settings, mock_get_redis, mock_dep_db, client, admin_user, mock_settings):
        """Test successful Redis cache clearing."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = admin_user
        mock_dep_db.return_value = mock_db_dep
        
        mock_redis_dep = AsyncMock()
        mock_redis_dep.get.return_value = None
        mock_get_redis.return_value = mock_redis_dep
        
        # Mock Redis for admin endpoint
        mock_redis_admin = AsyncMock()
        mock_redis_admin.flushdb = AsyncMock()
        mock_admin_redis.return_value = mock_redis_admin
        
        headers = create_auth_headers(str(admin_user["_id"]), mock_dep_settings)
        response = client.delete("/admin/cache/clear?cache_type=redis", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "Cache cleared successfully" in data["message"]
        assert "redis" in data["cleared_caches"]
        
        # Verify Redis flushdb was called
        mock_redis_admin.flushdb.assert_called_once()
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.admin.get_redis')
    def test_clear_cache_no_redis_available(self, mock_admin_redis, mock_dep_settings, mock_get_redis, mock_dep_db, client, admin_user, mock_settings):
        """Test cache clearing when Redis is not available."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = admin_user
        mock_dep_db.return_value = mock_db_dep
        
        mock_redis_dep = AsyncMock()
        mock_redis_dep.get.return_value = None
        mock_get_redis.return_value = mock_redis_dep
        
        # Mock no Redis available
        mock_admin_redis.return_value = None
        
        headers = create_auth_headers(str(admin_user["_id"]), mock_dep_settings)
        response = client.delete("/admin/cache/clear", headers=headers)
        
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "No cache services available" in response.json()["detail"]