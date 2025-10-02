"""
Tests for authentication middleware and dependency injection.
Tests the authentication middleware components and dependency functions.
"""
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException, status, WebSocket
from fastapi.security import HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from bson import ObjectId
from jose import jwt

# Mock settings before importing app modules
with patch.dict('os.environ', {
    'SECRET_KEY': 'test-secret-key',
    'MONGODB_URL': 'mongodb://test:27017',
    'DATABASE_NAME': 'test_db'
}):
    from app.core.dependencies import (
        get_current_user,
        get_current_admin_user,
        get_optional_user,
        get_current_user_websocket,
        get_current_user_or_api_key
    )
    from app.core.exceptions import AuthenticationError, AuthorizationError


class TestGetCurrentUserDependency:
    """Test get_current_user dependency function."""
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_valid_token_success(self, mock_settings):
        """Test successful authentication with valid token."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        user_id = str(ObjectId())
        user_data = {
            "_id": user_id,
            "email": "test@example.com",
            "role": "user",
            "is_active": True
        }
        
        # Create valid token
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        
        # Mock credentials
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        # Mock database
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = user_data
        
        # Mock Redis
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Token not blacklisted
        
        result = await get_current_user(credentials, mock_db, mock_redis)
        
        assert result == user_data
        mock_db.users.find_one.assert_called_once_with({"_id": user_id})
        mock_redis.get.assert_called_once_with(f"blacklist:{token}")
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_blacklisted_token_rejection(self, mock_settings):
        """Test rejection of blacklisted token."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        user_id = str(ObjectId())
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        mock_db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "1"  # Token is blacklisted
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials, mock_db, mock_redis)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        mock_redis.get.assert_called_once_with(f"blacklist:{token}")
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_invalid_token_format(self, mock_settings):
        """Test rejection of invalid token format."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")
        
        mock_db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials, mock_db, mock_redis)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_token_missing_subject(self, mock_settings):
        """Test rejection of token missing subject claim."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        # Token without 'sub' claim
        token_data = {"role": "user", "exp": datetime.utcnow() + timedelta(minutes=30)}
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        mock_db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials, mock_db, mock_redis)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_user_not_found_in_database(self, mock_settings):
        """Test rejection when user not found in database."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        user_id = str(ObjectId())
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = None  # User not found
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials, mock_db, mock_redis)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_redis_connection_failure_graceful_handling(self, mock_settings):
        """Test graceful handling of Redis connection failures."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        user_id = str(ObjectId())
        user_data = {
            "_id": user_id,
            "email": "test@example.com",
            "role": "user",
            "is_active": True
        }
        
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = user_data
        
        # Mock Redis to raise exception
        mock_redis = AsyncMock()
        mock_redis.get.side_effect = Exception("Redis connection failed")
        
        # Should still work despite Redis failure
        result = await get_current_user(credentials, mock_db, mock_redis)
        
        assert result == user_data


class TestGetCurrentAdminUserDependency:
    """Test get_current_admin_user dependency function."""
    
    @pytest.mark.asyncio
    async def test_admin_user_success(self):
        """Test successful admin user validation."""
        admin_user = {
            "_id": str(ObjectId()),
            "email": "admin@example.com",
            "role": "admin",
            "is_active": True
        }
        
        result = await get_current_admin_user(admin_user)
        
        assert result == admin_user
    
    @pytest.mark.asyncio
    async def test_non_admin_user_rejection(self):
        """Test rejection of non-admin user."""
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
    async def test_user_without_role_rejection(self):
        """Test rejection of user without role field."""
        user_without_role = {
            "_id": str(ObjectId()),
            "email": "user@example.com",
            "is_active": True
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin_user(user_without_role)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
    
    @pytest.mark.asyncio
    async def test_inactive_admin_user(self):
        """Test that inactive admin users are still validated for role."""
        inactive_admin = {
            "_id": str(ObjectId()),
            "email": "admin@example.com",
            "role": "admin",
            "is_active": False
        }
        
        # Should still pass role check (activity check is separate)
        result = await get_current_admin_user(inactive_admin)
        assert result == inactive_admin


class TestGetOptionalUserDependency:
    """Test get_optional_user dependency function."""
    
    @pytest.mark.asyncio
    async def test_no_credentials_returns_none(self):
        """Test that missing credentials returns None."""
        mock_db = MagicMock()
        
        result = await get_optional_user(None, mock_db)
        
        assert result is None
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.get_current_user')
    async def test_valid_credentials_returns_user(self, mock_get_current_user):
        """Test that valid credentials return user."""
        user_data = {
            "_id": str(ObjectId()),
            "email": "test@example.com",
            "role": "user"
        }
        
        mock_get_current_user.return_value = user_data
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-token")
        mock_db = MagicMock()
        
        result = await get_optional_user(credentials, mock_db)
        
        assert result == user_data
        mock_get_current_user.assert_called_once_with(credentials, mock_db)
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.get_current_user')
    async def test_invalid_credentials_returns_none(self, mock_get_current_user):
        """Test that invalid credentials return None instead of raising exception."""
        mock_get_current_user.side_effect = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
        
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid-token")
        mock_db = MagicMock()
        
        result = await get_optional_user(credentials, mock_db)
        
        assert result is None


class TestGetCurrentUserWebSocketDependency:
    """Test get_current_user_websocket dependency function."""
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_valid_websocket_token(self, mock_settings):
        """Test valid WebSocket token authentication."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        user_id = str(ObjectId())
        user_data = {
            "_id": user_id,
            "email": "test@example.com",
            "role": "user"
        }
        
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        
        mock_websocket = MagicMock(spec=WebSocket)
        mock_db = MagicMock()
        mock_db.users.find_one.return_value = user_data
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None  # Token not blacklisted
        
        result = await get_current_user_websocket(mock_websocket, token, mock_db, mock_redis)
        
        assert result == user_data
        mock_db.users.find_one.assert_called_once_with({"_id": user_id})
    
    @pytest.mark.asyncio
    async def test_websocket_no_token_returns_none(self):
        """Test WebSocket authentication without token returns None."""
        mock_websocket = MagicMock(spec=WebSocket)
        mock_db = MagicMock()
        mock_redis = AsyncMock()
        
        result = await get_current_user_websocket(mock_websocket, None, mock_db, mock_redis)
        
        assert result is None
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_websocket_blacklisted_token_returns_none(self, mock_settings):
        """Test WebSocket authentication with blacklisted token returns None."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        user_id = str(ObjectId())
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        
        mock_websocket = MagicMock(spec=WebSocket)
        mock_db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = "1"  # Token is blacklisted
        
        result = await get_current_user_websocket(mock_websocket, token, mock_db, mock_redis)
        
        assert result is None
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_websocket_invalid_token_returns_none(self, mock_settings):
        """Test WebSocket authentication with invalid token returns None."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        mock_websocket = MagicMock(spec=WebSocket)
        mock_db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        
        result = await get_current_user_websocket(mock_websocket, "invalid-token", mock_db, mock_redis)
        
        assert result is None


class TestAPIKeyAuthentication:
    """Test API key authentication functionality."""
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.get_api_key')
    @patch('app.core.dependencies.get_current_user')
    async def test_user_or_api_key_with_valid_api_key(self, mock_get_current_user, mock_get_api_key):
        """Test authentication with valid API key."""
        from app.core.api_keys import APIKey, APIKeyScope
        
        api_key = APIKey(
            key_id="test-key",
            name="Test Key",
            scopes=[APIKeyScope.READ],
            is_active=True
        )
        
        mock_get_api_key.return_value = api_key
        
        credentials = None
        mock_db = MagicMock()
        mock_redis = AsyncMock()
        
        result = await get_current_user_or_api_key(credentials, api_key, mock_db, mock_redis)
        
        assert result == api_key
        mock_get_current_user.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.get_api_key')
    @patch('app.core.dependencies.get_current_user')
    async def test_user_or_api_key_with_valid_user(self, mock_get_current_user, mock_get_api_key):
        """Test authentication with valid user token."""
        user_data = {
            "_id": str(ObjectId()),
            "email": "test@example.com",
            "role": "user"
        }
        
        mock_get_api_key.return_value = None
        mock_get_current_user.return_value = user_data
        
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid-token")
        mock_db = MagicMock()
        mock_redis = AsyncMock()
        
        result = await get_current_user_or_api_key(credentials, None, mock_db, mock_redis)
        
        assert result == user_data
        mock_get_current_user.assert_called_once_with(credentials, mock_db, mock_redis)
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.get_api_key')
    async def test_user_or_api_key_no_auth(self, mock_get_api_key):
        """Test authentication failure when neither API key nor user token provided."""
        mock_get_api_key.return_value = None
        
        credentials = None
        mock_db = MagicMock()
        mock_redis = AsyncMock()
        
        with pytest.raises(AuthenticationError) as exc_info:
            await get_current_user_or_api_key(credentials, None, mock_db, mock_redis)
        
        assert "Authentication required" in str(exc_info.value)


class TestMiddlewareErrorHandling:
    """Test error handling in authentication middleware."""
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_jwt_decode_error_handling(self, mock_settings):
        """Test handling of JWT decode errors."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        # Create token with wrong secret
        token_data = {"sub": str(ObjectId())}
        token = jwt.encode(token_data, "wrong-secret", algorithm="HS256")
        
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        mock_db = MagicMock()
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(credentials, mock_db, mock_redis)
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Could not validate credentials" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    @patch('app.core.dependencies.settings')
    async def test_database_error_handling(self, mock_settings):
        """Test handling of database errors."""
        mock_settings.secret_key = "test-secret-key"
        mock_settings.algorithm = "HS256"
        
        user_id = str(ObjectId())
        token_data = {"sub": user_id}
        token = jwt.encode(token_data, "test-secret-key", algorithm="HS256")
        
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
        
        # Mock database to raise exception
        mock_db = MagicMock()
        mock_db.users.find_one.side_effect = Exception("Database connection failed")
        
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        
        # Should propagate database error
        with pytest.raises(Exception) as exc_info:
            await get_current_user(credentials, mock_db, mock_redis)
        
        assert "Database connection failed" in str(exc_info.value)