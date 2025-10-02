"""
Unit tests for authentication system components.
Tests individual functions and methods without external dependencies.
"""
import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
from jose import jwt
from fastapi import HTTPException, status
from bson import ObjectId

# Mock settings before importing app modules
with patch.dict('os.environ', {
    'SECRET_KEY': 'test-secret-key',
    'MONGODB_URL': 'mongodb://test:27017',
    'DATABASE_NAME': 'test_db'
}):
    from app.api.auth import (
        verify_password,
        get_password_hash,
        create_access_token,
        authenticate_user
    )
    from app.core.dependencies import get_current_user, get_current_admin_user
    from app.models.user import UserCreate, UserRole


class TestPasswordHandling:
    """Test password hashing and verification."""
    
    def test_password_hashing(self):
        """Test password hashing creates different hashes for same password."""
        password = "testpassword123"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        
        # Hashes should be different due to salt
        assert hash1 != hash2
        assert len(hash1) > 0
        assert len(hash2) > 0
    
    def test_password_verification_success(self):
        """Test successful password verification."""
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_password_verification_failure(self):
        """Test failed password verification."""
        password = "testpassword123"
        wrong_password = "wrongpassword"
        hashed = get_password_hash(password)
        
        assert verify_password(wrong_password, hashed) is False
    
    def test_empty_password_handling(self):
        """Test handling of empty passwords."""
        # Empty password should still hash (bcrypt handles empty strings)
        # but we can test that it creates a hash
        result = get_password_hash("")
        assert len(result) > 0
        assert result.startswith("$2b$")


class TestJWTTokenHandling:
    """Test JWT token creation and validation."""
    
    @patch('app.api.auth.settings')
    def test_create_access_token_default_expiry(self, mock_settings):
        """Test token creation with default expiry."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        mock_settings.access_token_expire_minutes = 30
        
        data = {"sub": "user123"}
        token = create_access_token(data)
        
        # Decode token to verify contents
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert payload["sub"] == "user123"
        assert "exp" in payload
    
    @patch('app.api.auth.settings')
    def test_create_access_token_custom_expiry(self, mock_settings):
        """Test token creation with custom expiry."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        data = {"sub": "user123"}
        expires_delta = timedelta(minutes=60)
        token = create_access_token(data, expires_delta)
        
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert payload["sub"] == "user123"
        
        # Check that expiry exists
        assert "exp" in payload
        assert isinstance(payload["exp"], (int, float))
    
    @patch('app.api.auth.settings')
    def test_token_with_additional_claims(self, mock_settings):
        """Test token creation with additional claims."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        mock_settings.access_token_expire_minutes = 30
        
        data = {"sub": "user123", "role": "admin", "email": "test@example.com"}
        token = create_access_token(data)
        
        payload = jwt.decode(token, "test-secret-key", algorithms=["HS256"])
        assert payload["sub"] == "user123"
        assert payload["role"] == "admin"
        assert payload["email"] == "test@example.com"


class TestUserAuthentication:
    """Test user authentication logic."""
    
    @pytest.mark.asyncio
    async def test_authenticate_user_success(self):
        """Test successful user authentication."""
        mock_db = MagicMock()
        user_data = {
            "_id": ObjectId(),
            "email": "test@example.com",
            "hashed_password": get_password_hash("testpassword123"),
            "role": "user",
            "is_active": True
        }
        mock_db.users.find_one = AsyncMock(return_value=user_data)
        
        result = await authenticate_user("test@example.com", "testpassword123", mock_db)
        
        assert result == user_data
        mock_db.users.find_one.assert_called_once_with({"email": "test@example.com"})
    
    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self):
        """Test authentication with non-existent user."""
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value=None)
        
        result = await authenticate_user("nonexistent@example.com", "password", mock_db)
        
        assert result is False
        mock_db.users.find_one.assert_called_once_with({"email": "nonexistent@example.com"})
    
    @pytest.mark.asyncio
    async def test_authenticate_user_wrong_password(self):
        """Test authentication with wrong password."""
        mock_db = MagicMock()
        user_data = {
            "_id": ObjectId(),
            "email": "test@example.com",
            "hashed_password": get_password_hash("correctpassword"),
            "role": "user",
            "is_active": True
        }
        mock_db.users.find_one = AsyncMock(return_value=user_data)
        
        result = await authenticate_user("test@example.com", "wrongpassword", mock_db)
        
        assert result is False


class TestCurrentUserDependency:
    """Test get_current_user dependency."""
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_get_current_user_success(self, mock_settings):
        """Test successful current user retrieval."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        # Create mock credentials
        mock_credentials = MagicMock()
        user_id = str(ObjectId())
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        mock_credentials.credentials = token
        
        # Create mock database
        mock_db = MagicMock()
        user_data = {
            "_id": user_id,
            "email": "test@example.com",
            "role": "user",
            "is_active": True
        }
        mock_db.users.find_one = AsyncMock(return_value=user_data)
        
        # Create mock Redis
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Token not blacklisted
        
        result = await get_current_user(mock_credentials, mock_db, mock_redis)
        
        assert result == user_data
        mock_db.users.find_one.assert_called_once_with({"_id": user_id})
        mock_redis.get.assert_called_once_with(f"blacklist:{token}")
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_get_current_user_redis_failure_graceful(self, mock_settings):
        """Test current user retrieval handles Redis failures gracefully."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        mock_credentials = MagicMock()
        user_id = str(ObjectId())
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        mock_credentials.credentials = token
        
        mock_db = MagicMock()
        user_data = {
            "_id": user_id,
            "email": "test@example.com",
            "role": "user",
            "is_active": True
        }
        mock_db.users.find_one = AsyncMock(return_value=user_data)
        
        # Mock Redis to raise an exception (simulating connection failure)
        mock_redis = AsyncMock()
        mock_redis.get.side_effect = Exception("Redis connection failed")
        
        # Should still work despite Redis failure
        result = await get_current_user(mock_credentials, mock_db, mock_redis)
        
        assert result == user_data
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_get_current_user_invalid_token(self, mock_settings):
        """Test current user retrieval with invalid token."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        mock_credentials = MagicMock()
        mock_credentials.credentials = "invalid-token"
        
        mock_db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_credentials, mock_db, mock_redis)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_get_current_user_no_subject(self, mock_settings):
        """Test current user retrieval with token missing subject."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        mock_credentials = MagicMock()
        token_data = {"role": "user"}  # Missing 'sub' claim
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        mock_credentials.credentials = token
        
        mock_db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_credentials, mock_db, mock_redis)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_get_current_user_user_not_found(self, mock_settings):
        """Test current user retrieval when user doesn't exist in database."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        mock_credentials = MagicMock()
        user_id = str(ObjectId())
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        mock_credentials.credentials = token
        
        mock_db = MagicMock()
        mock_db.users.find_one = AsyncMock(return_value=None)  # User not found
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(mock_credentials, mock_db, mock_redis)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED


class TestAdminUserDependency:
    """Test get_current_admin_user dependency."""
    
    @pytest.mark.asyncio
    async def test_get_current_admin_user_success(self):
        """Test successful admin user retrieval."""
        admin_user = {
            "_id": str(ObjectId()),
            "email": "admin@example.com",
            "role": "admin",
            "is_active": True
        }
        
        result = await get_current_admin_user(admin_user)
        
        assert result == admin_user
    
    @pytest.mark.asyncio
    async def test_get_current_admin_user_not_admin(self):
        """Test admin user retrieval with non-admin user."""
        regular_user = {
            "_id": str(ObjectId()),
            "email": "user@example.com",
            "role": "user",
            "is_active": True
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin_user(regular_user)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Not enough permissions" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_get_current_admin_user_missing_role(self):
        """Test admin user retrieval with user missing role."""
        user_without_role = {
            "_id": str(ObjectId()),
            "email": "user@example.com",
            "is_active": True
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin_user(user_without_role)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN


class TestUserModelValidation:
    """Test user model validation."""
    
    def test_user_create_valid(self):
        """Test valid user creation data."""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpassword123",
            "role": UserRole.USER
        }
        
        user = UserCreate(**user_data)
        
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.password == "testpassword123"
        assert user.role == UserRole.USER
    
    def test_user_create_invalid_email(self):
        """Test user creation with invalid email."""
        user_data = {
            "email": "invalid-email",
            "username": "testuser",
            "password": "testpassword123"
        }
        
        with pytest.raises(ValueError):
            UserCreate(**user_data)
    
    def test_user_create_short_username(self):
        """Test user creation with too short username."""
        user_data = {
            "email": "test@example.com",
            "username": "ab",  # Too short
            "password": "testpassword123"
        }
        
        with pytest.raises(ValueError):
            UserCreate(**user_data)
    
    def test_user_create_short_password(self):
        """Test user creation with too short password."""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "short"  # Too short
        }
        
        with pytest.raises(ValueError):
            UserCreate(**user_data)
    
    def test_user_create_default_role(self):
        """Test user creation with default role."""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "testpassword123"
        }
        
        user = UserCreate(**user_data)
        
        assert user.role == UserRole.USER