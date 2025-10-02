"""
Integration tests for authentication endpoints.
Tests the complete authentication flow through FastAPI endpoints.
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
    settings = Settings(
        secret_key="test-secret-key",
        algorithm="HS256",
        access_token_expire_minutes=30,
        mongodb_url="mongodb://test",
        database_name="test_db"
    )
    return settings


@pytest.fixture
def sample_user():
    """Sample user data for testing."""
    return {
        "_id": ObjectId(),
        "email": "test@example.com",
        "username": "testuser",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "role": "user",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "preferences": {}
    }


@pytest.fixture
def admin_user():
    """Sample admin user data for testing."""
    return {
        "_id": ObjectId(),
        "email": "admin@example.com",
        "username": "adminuser",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # "secret"
        "role": "admin",
        "is_active": True,
        "created_at": datetime.utcnow(),
        "preferences": {}
    }


class TestRegistrationEndpoint:
    """Test user registration endpoint."""
    
    @patch('app.api.auth.get_database')
    def test_register_success(self, mock_get_db, client):
        """Test successful user registration."""
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = None  # User doesn't exist
        mock_db.users.insert_one.return_value = MagicMock(inserted_id=ObjectId())
        mock_get_db.return_value = mock_db
        
        user_data = {
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "newpassword123"
        }
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["username"] == "newuser"
        assert "password" not in data
        assert "hashed_password" not in data
        
        # Verify database calls
        mock_db.users.find_one.assert_any_call({"email": "newuser@example.com"})
        mock_db.users.find_one.assert_any_call({"username": "newuser"})
        mock_db.users.insert_one.assert_called_once()
    
    @patch('app.api.auth.get_database')
    def test_register_email_already_exists(self, mock_get_db, client, sample_user):
        """Test registration with existing email."""
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = sample_user  # Email exists
        mock_get_db.return_value = mock_db
        
        user_data = {
            "email": "test@example.com",
            "username": "newuser",
            "password": "newpassword123"
        }
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Email already registered" in response.json()["detail"]
    
    @patch('app.api.auth.get_database')
    def test_register_username_already_exists(self, mock_get_db, client, sample_user):
        """Test registration with existing username."""
        mock_db = MagicMock()
        # First call (email check) returns None, second call (username check) returns user
        mock_db.users.find_one.side_effect = [None, sample_user]
        mock_get_db.return_value = mock_db
        
        user_data = {
            "email": "newemail@example.com",
            "username": "testuser",
            "password": "newpassword123"
        }
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Username already taken" in response.json()["detail"]
    
    def test_register_invalid_email(self, client):
        """Test registration with invalid email format."""
        user_data = {
            "email": "invalid-email",
            "username": "newuser",
            "password": "newpassword123"
        }
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_register_short_password(self, client):
        """Test registration with too short password."""
        user_data = {
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "short"
        }
        
        response = client.post("/auth/register", json=user_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestLoginEndpoint:
    """Test user login endpoint."""
    
    @patch('app.api.auth.get_database')
    @patch('app.api.auth.settings')
    def test_login_success(self, mock_settings, mock_get_db, client, sample_user):
        """Test successful user login."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        mock_settings.access_token_expire_minutes = 30
        
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = sample_user
        mock_db.users.update_one = AsyncMock()
        mock_get_db.return_value = mock_db
        
        login_data = {
            "username": "test@example.com",
            "password": "secret"
        }
        
        response = client.post("/auth/login", data=login_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        
        # Verify token contains correct user ID
        token = data["access_token"]
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert payload["sub"] == str(sample_user["_id"])
    
    @patch('app.api.auth.get_database')
    def test_login_user_not_found(self, mock_get_db, client):
        """Test login with non-existent user."""
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = None
        mock_get_db.return_value = mock_db
        
        login_data = {
            "username": "nonexistent@example.com",
            "password": "password"
        }
        
        response = client.post("/auth/login", data=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect email or password" in response.json()["detail"]
    
    @patch('app.api.auth.get_database')
    def test_login_wrong_password(self, mock_get_db, client, sample_user):
        """Test login with wrong password."""
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = sample_user
        mock_get_db.return_value = mock_db
        
        login_data = {
            "username": "test@example.com",
            "password": "wrongpassword"
        }
        
        response = client.post("/auth/login", data=login_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect email or password" in response.json()["detail"]


class TestProfileEndpoint:
    """Test user profile endpoints."""
    
    def create_auth_headers(self, user_id: str, settings) -> dict:
        """Create authentication headers for testing."""
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, settings.secret_key, algorithm=settings.algorithm)
        return {"Authorization": f"Bearer {token}"}
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    def test_get_profile_success(self, mock_settings, mock_get_redis, mock_get_db, client, sample_user):
        """Test successful profile retrieval."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = sample_user
        mock_get_db.return_value = mock_db
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Token not blacklisted
        mock_get_redis.return_value = mock_redis
        
        headers = self.create_auth_headers(str(sample_user["_id"]), mock_settings)
        response = client.get("/auth/profile", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == sample_user["email"]
        assert data["username"] == sample_user["username"]
        assert data["role"] == sample_user["role"]
        assert "hashed_password" not in data
    
    def test_get_profile_no_auth(self, client):
        """Test profile retrieval without authentication."""
        response = client.get("/auth/profile")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    def test_get_profile_invalid_token(self, mock_settings, mock_get_redis, mock_get_db, client):
        """Test profile retrieval with invalid token."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/auth/profile", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestRefreshTokenEndpoint:
    """Test token refresh endpoint."""
    
    def create_auth_headers(self, user_id: str, settings) -> dict:
        """Create authentication headers for testing."""
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, settings.secret_key, algorithm=settings.algorithm)
        return {"Authorization": f"Bearer {token}"}
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.auth.settings')
    def test_refresh_token_success(self, mock_auth_settings, mock_dep_settings, mock_get_redis, mock_get_db, client, sample_user):
        """Test successful token refresh."""
        mock_auth_settings.secret_key = "test-secret-key"
        mock_auth_settings.algorithm = "HS256"
        mock_auth_settings.access_token_expire_minutes = 30
        
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = sample_user
        mock_get_db.return_value = mock_db
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Token not blacklisted
        mock_get_redis.return_value = mock_redis
        
        headers = self.create_auth_headers(str(sample_user["_id"]), mock_dep_settings)
        response = client.post("/auth/refresh", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_refresh_token_no_auth(self, client):
        """Test token refresh without authentication."""
        response = client.post("/auth/refresh")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestLogoutEndpoint:
    """Test user logout endpoint."""
    
    def create_auth_headers(self, user_id: str, settings) -> dict:
        """Create authentication headers for testing."""
        token_data = {"sub": user_id, "exp": datetime.utcnow() + timedelta(minutes=30)}
        token = jwt.encode(token_data, settings.secret_key, algorithm=settings.algorithm)
        return {"Authorization": f"Bearer {token}"}
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.auth.get_redis')
    @patch('app.api.auth.settings')
    def test_logout_success(self, mock_auth_settings, mock_auth_redis, mock_dep_settings, mock_dep_redis, mock_get_db, client, sample_user):
        """Test successful user logout."""
        mock_auth_settings.secret_key = "test-secret-key"
        mock_auth_settings.algorithm = "HS256"
        
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = sample_user
        mock_get_db.return_value = mock_db
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Token not blacklisted
        mock_redis.setex = AsyncMock()
        mock_dep_redis.return_value = mock_redis
        mock_auth_redis.return_value = mock_redis
        
        headers = self.create_auth_headers(str(sample_user["_id"]), mock_dep_settings)
        response = client.delete("/auth/logout", headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "Successfully logged out" in data["message"]
        
        # Verify token was blacklisted
        mock_redis.setex.assert_called_once()
    
    def test_logout_no_auth(self, client):
        """Test logout without authentication."""
        response = client.delete("/auth/logout")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestUpdateProfileEndpoint:
    """Test profile update endpoint."""
    
    def create_auth_headers(self, user_id: str, settings) -> dict:
        """Create authentication headers for testing."""
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, settings.secret_key, algorithm=settings.algorithm)
        return {"Authorization": f"Bearer {token}"}
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.auth.get_database')
    def test_update_profile_success(self, mock_auth_db, mock_dep_settings, mock_get_redis, mock_dep_db, client, sample_user):
        """Test successful profile update."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        # Mock for dependency
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = sample_user
        mock_dep_db.return_value = mock_db_dep
        
        # Mock for endpoint
        mock_db_auth = MagicMock()
        mock_db_auth.users.find_one.side_effect = [None, None]  # No conflicts
        mock_db_auth.users.update_one = AsyncMock()
        updated_user = sample_user.copy()
        updated_user["username"] = "newusername"
        mock_db_auth.users.find_one.return_value = updated_user
        mock_auth_db.return_value = mock_db_auth
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        headers = self.create_auth_headers(str(sample_user["_id"]), mock_dep_settings)
        update_data = {"username": "newusername"}
        
        response = client.put("/auth/profile", json=update_data, headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["username"] == "newusername"
    
    def test_update_profile_no_auth(self, client):
        """Test profile update without authentication."""
        update_data = {"username": "newusername"}
        response = client.put("/auth/profile", json=update_data)
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestChangePasswordEndpoint:
    """Test password change endpoint."""
    
    def create_auth_headers(self, user_id: str, settings) -> dict:
        """Create authentication headers for testing."""
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, settings.secret_key, algorithm=settings.algorithm)
        return {"Authorization": f"Bearer {token}"}
    
    @patch('app.core.dependencies.get_database')
    @patch('app.core.dependencies.get_redis')
    @patch('app.core.dependencies.settings')
    @patch('app.api.auth.get_database')
    def test_change_password_success(self, mock_auth_db, mock_dep_settings, mock_get_redis, mock_dep_db, client, sample_user):
        """Test successful password change."""
        mock_dep_settings.secret_key = "test-secret-key"
        mock_dep_settings.algorithm = "HS256"
        
        # Mock for dependency
        mock_db_dep = MagicMock()
        mock_db_dep.users.find_one.return_value = sample_user
        mock_dep_db.return_value = mock_db_dep
        
        # Mock for endpoint
        mock_db_auth = MagicMock()
        mock_db_auth.users.update_one = AsyncMock()
        mock_auth_db.return_value = mock_db_auth
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_get_redis.return_value = mock_redis
        
        headers = self.create_auth_headers(str(sample_user["_id"]), mock_dep_settings)
        password_data = {
            "currentPassword": "secret",
            "newPassword": "newsecret123"
        }
        
        response = client.put("/auth/change-password", json=password_data, headers=headers)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "Password changed successfully" in data["message"]
    
    def test_change_password_no_auth(self, client):
        """Test password change without authentication."""
        password_data = {
            "currentPassword": "secret",
            "newPassword": "newsecret123"
        }
        response = client.put("/auth/change-password", json=password_data)
        
        assert response.status_code == status.HTTP_403_FORBIDDEN