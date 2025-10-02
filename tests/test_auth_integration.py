"""
Integration tests for authentication workflows.
Tests complete user authentication flows including login/logout cycles.
"""
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI, status
from datetime import datetime, timedelta
from bson import ObjectId
from jose import jwt

# Mock settings before importing app modules
with patch.dict('os.environ', {
    'SECRET_KEY': 'test-secret-key',
    'MONGODB_URL': 'mongodb://test:27017',
    'DATABASE_NAME': 'test_db'
}):
    from app.api.auth import router as auth_router
    from app.core.config import Settings
    from app.models.user import UserRole


# Test app setup
def create_test_app():
    """Create test FastAPI app with auth router."""
    app = FastAPI()
    app.include_router(auth_router)
    return app


@pytest.fixture
def test_app():
    """Create test app fixture."""
    return create_test_app()


@pytest.fixture
def client(test_app):
    """Create test client fixture."""
    with TestClient(test_app) as client:
        yield client


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    return Settings(
        secret_key="test-secret-key",
        algorithm="HS256",
        access_token_expire_minutes=30,
        mongodb_url="mongodb://test",
        database_name="test_db"
    )


class TestCompleteAuthenticationFlow:
    """Test complete authentication workflows."""
    
    @patch('app.api.auth.get_database')
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.auth.settings')
    def test_complete_registration_login_profile_flow(self, mock_auth_settings, mock_dep_settings, mock_get_redis, mock_dep_db, mock_auth_db, client):
        """Test complete flow: register -> login -> get profile."""
        # Setup mocks
        mock_auth_settings.secret_key = "test-secret-key"
        mock_auth_settings.algorithm = "HS256"
        mock_auth_settings.access_token_expire_minutes = 30
        
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        # Mock database for registration
        mock_db_auth = MagicMock()
        mock_db_auth.users.find_one.return_value = None  # User doesn't exist
        user_id = ObjectId()
        mock_db_auth.users.insert_one.return_value = MagicMock(inserted_id=user_id)
        mock_db_auth.users.update_one = AsyncMock()
        mock_auth_db.return_value = mock_db_auth
        
        # Mock database for dependencies
        created_user = {
            "_id": user_id,
            "email": "testflow@example.com",
            "username": "testflowuser",
            "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "testpassword123"
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "preferences": {}
        }
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = created_user
        mock_dep_db.return_value = mock_db_dep
        
        # Mock Redis
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Token not blacklisted
        mock_get_redis.return_value = mock_redis
        
        # Step 1: Register user
        registration_data = {
            "email": "testflow@example.com",
            "username": "testflowuser",
            "password": "testpassword123"
        }
        
        register_response = client.post("/auth/register", json=registration_data)
        assert register_response.status_code == status.HTTP_200_OK
        
        register_data = register_response.json()
        assert register_data["email"] == "testflow@example.com"
        assert register_data["username"] == "testflowuser"
        
        # Update mock to return user for login
        mock_db_auth.users.find_one.return_value = created_user
        
        # Step 2: Login with registered user
        login_data = {
            "username": "testflow@example.com",
            "password": "testpassword123"
        }
        
        login_response = client.post("/auth/login", data=login_data)
        assert login_response.status_code == status.HTTP_200_OK
        
        login_result = login_response.json()
        assert "access_token" in login_result
        assert login_result["token_type"] == "bearer"
        
        # Step 3: Get profile with token
        headers = {"Authorization": f"Bearer {login_result['access_token']}"}
        profile_response = client.get("/auth/profile", headers=headers)
        
        assert profile_response.status_code == status.HTTP_200_OK
        profile_data = profile_response.json()
        assert profile_data["email"] == "testflow@example.com"
        assert profile_data["username"] == "testflowuser"
        assert profile_data["role"] == "user"
    
    @patch('app.api.auth.get_database')
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.auth.settings')
    @patch('app.api.auth.get_redis')
    def test_login_logout_token_blacklist_flow(self, mock_auth_redis, mock_auth_settings, mock_dep_settings, mock_get_redis, mock_dep_db, mock_auth_db, client):
        """Test login -> logout -> attempt to use blacklisted token."""
        # Setup mocks
        mock_auth_settings.secret_key = "test-secret-key"
        mock_auth_settings.algorithm = "HS256"
        mock_auth_settings.access_token_expire_minutes = 30
        
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        user_id = ObjectId()
        user_data = {
            "_id": user_id,
            "email": "logout@example.com",
            "username": "logoutuser",
            "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "preferences": {}
        }
        
        # Mock database
        mock_db_auth = MagicMock()
        mock_db_auth.users.find_one.return_value = user_data
        mock_db_auth.users.update_one = AsyncMock()
        mock_auth_db.return_value = mock_db_auth
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = user_data
        mock_dep_db.return_value = mock_db_dep
        
        # Mock Redis
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Initially token not blacklisted
        mock_redis.setex = AsyncMock()
        mock_get_redis.return_value = mock_redis
        mock_auth_redis.return_value = mock_redis
        
        # Step 1: Login
        login_data = {
            "username": "logout@example.com",
            "password": "secret"
        }
        
        login_response = client.post("/auth/login", data=login_data)
        assert login_response.status_code == status.HTTP_200_OK
        
        login_result = login_response.json()
        token = login_result["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Step 2: Verify token works
        profile_response = client.get("/auth/profile", headers=headers)
        assert profile_response.status_code == status.HTTP_200_OK
        
        # Step 3: Logout (blacklist token)
        logout_response = client.delete("/auth/logout", headers=headers)
        assert logout_response.status_code == status.HTTP_200_OK
        
        # Verify token was blacklisted
        mock_redis.setex.assert_called_once()
        
        # Step 4: Try to use blacklisted token
        mock_redis.get.return_value = "1"  # Token is now blacklisted
        
        profile_response_after_logout = client.get("/auth/profile", headers=headers)
        assert profile_response_after_logout.status_code == status.HTTP_401_UNAUTHORIZED
    
    @patch('app.api.auth.get_database')
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.auth.settings')
    def test_token_refresh_flow(self, mock_auth_settings, mock_dep_settings, mock_get_redis, mock_dep_db, mock_auth_db, client):
        """Test login -> refresh token -> use new token."""
        # Setup mocks
        mock_auth_settings.secret_key = "test-secret-key"
        mock_auth_settings.algorithm = "HS256"
        mock_auth_settings.access_token_expire_minutes = 30
        
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        user_id = ObjectId()
        user_data = {
            "_id": user_id,
            "email": "refresh@example.com",
            "username": "refreshuser",
            "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "preferences": {}
        }
        
        # Mock database
        mock_db_auth = MagicMock()
        mock_db_auth.users.find_one.return_value = user_data
        mock_db_auth.users.update_one = AsyncMock()
        mock_auth_db.return_value = mock_db_auth
        
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = user_data
        mock_dep_db.return_value = mock_db_dep
        
        # Mock Redis
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Token not blacklisted
        mock_get_redis.return_value = mock_redis
        
        # Step 1: Login
        login_data = {
            "username": "refresh@example.com",
            "password": "secret"
        }
        
        login_response = client.post("/auth/login", data=login_data)
        assert login_response.status_code == status.HTTP_200_OK
        
        login_result = login_response.json()
        original_token = login_result["access_token"]
        headers = {"Authorization": f"Bearer {original_token}"}
        
        # Step 2: Refresh token
        refresh_response = client.post("/auth/refresh", headers=headers)
        assert refresh_response.status_code == status.HTTP_200_OK
        
        refresh_result = refresh_response.json()
        new_token = refresh_result["access_token"]
        
        # Verify new token is different
        assert new_token != original_token
        
        # Step 3: Use new token
        new_headers = {"Authorization": f"Bearer {new_token}"}
        profile_response = client.get("/auth/profile", headers=new_headers)
        
        assert profile_response.status_code == status.HTTP_200_OK
        profile_data = profile_response.json()
        assert profile_data["email"] == "refresh@example.com"


class TestRoleBasedAccessControl:
    """Test role-based access control workflows."""
    
    def create_auth_headers(self, user_id: str, settings) -> dict:
        """Create authentication headers for testing."""
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, settings.secret_key, algorithm=settings.algorithm)
        return {"Authorization": f"Bearer {token}"}
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    def test_regular_user_access_control(self, mock_settings, mock_get_redis, mock_get_db, client):
        """Test that regular users can access user endpoints but not admin endpoints."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        user_id = ObjectId()
        regular_user = {
            "_id": user_id,
            "email": "user@example.com",
            "username": "regularuser",
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "preferences": {}
        }
        
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = regular_user
        mock_get_db.return_value = mock_db
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        headers = self.create_auth_headers(str(user_id), mock_settings)
        
        # Should be able to access profile
        profile_response = client.get("/auth/profile", headers=headers)
        assert profile_response.status_code == status.HTTP_200_OK
        
        # Should be able to refresh token
        refresh_response = client.post("/auth/refresh", headers=headers)
        assert refresh_response.status_code == status.HTTP_200_OK
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    def test_admin_user_access_control(self, mock_settings, mock_get_redis, mock_get_db, client):
        """Test that admin users can access both user and admin endpoints."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        user_id = ObjectId()
        admin_user = {
            "_id": user_id,
            "email": "admin@example.com",
            "username": "adminuser",
            "role": "admin",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "preferences": {}
        }
        
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = admin_user
        mock_get_db.return_value = mock_db
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        headers = self.create_auth_headers(str(user_id), mock_settings)
        
        # Should be able to access profile
        profile_response = client.get("/auth/profile", headers=headers)
        assert profile_response.status_code == status.HTTP_200_OK
        
        # Should be able to refresh token
        refresh_response = client.post("/auth/refresh", headers=headers)
        assert refresh_response.status_code == status.HTTP_200_OK


class TestTokenExpiration:
    """Test token expiration scenarios."""
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    def test_expired_token_rejection(self, mock_settings, mock_get_redis, mock_get_db, client):
        """Test that expired tokens are rejected."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        user_id = str(ObjectId())
        
        # Create expired token
        expired_time = datetime.utcnow() - timedelta(minutes=1)
        token_data = {"sub": user_id, "exp": expired_time}
        expired_token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        
        response = client.get("/auth/profile", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    @patch('app.api.auth.settings')
    def test_token_expiration_time_setting(self, mock_settings):
        """Test that token expiration respects settings."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        mock_settings.access_token_expire_minutes = 60
        
        from app.api.auth import create_access_token
        
        data = {"sub": "user123"}
        token = create_access_token(data)
        
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        exp_time = datetime.fromtimestamp(payload["exp"])
        expected_time = datetime.utcnow() + timedelta(minutes=60)
        
        # Allow 5 second tolerance
        time_diff = abs((exp_time - expected_time).total_seconds())
        assert time_diff < 5


class TestSecurityScenarios:
    """Test various security scenarios."""
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    def test_token_reuse_after_logout(self, mock_settings, mock_get_redis, mock_get_db, client):
        """Test that tokens cannot be reused after logout."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        user_id = str(ObjectId())
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        
        # Mock Redis to simulate blacklisted token
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "1"  # Token is blacklisted
        mock_get_redis.return_value = mock_redis
        
        headers = {"Authorization": f"Bearer {token}"}
        
        response = client.get("/auth/profile", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_malformed_token_rejection(self, client):
        """Test that malformed tokens are rejected."""
        headers = {"Authorization": "Bearer malformed-token"}
        
        response = client.get("/auth/profile", headers=headers)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_missing_authorization_header(self, client):
        """Test that requests without authorization header are rejected."""
        response = client.get("/auth/profile")
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_wrong_token_format(self, client):
        """Test that wrong token format is rejected."""
        headers = {"Authorization": "Basic dGVzdDp0ZXN0"}  # Basic auth instead of Bearer
        
        response = client.get("/auth/profile", headers=headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN