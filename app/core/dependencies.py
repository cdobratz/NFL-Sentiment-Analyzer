from fastapi import Depends, HTTPException, status, WebSocket, Query, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import JWTError, jwt
from typing import Optional, Union
import logging
from .config import settings
from .database import get_database, get_redis
from .api_keys import api_key_manager, APIKey, APIKeyScope
from .exceptions import AuthenticationError, AuthorizationError

logger = logging.getLogger(__name__)

security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db = Depends(get_database),
    redis = Depends(get_redis)
):
    """Get current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token = credentials.credentials
    
    # Check if token is blacklisted
    if redis:
        try:
            is_blacklisted = await redis.get(f"blacklist:{token}")
            if is_blacklisted:
                raise credentials_exception
        except Exception as e:
            logger.warning(f"Failed to check token blacklist: {e}")
    
    try:
        payload = jwt.decode(
            token, 
            settings.secret_key, 
            algorithms=[settings.algorithm]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Get user from database
    user = await db.users.find_one({"_id": user_id})
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_admin_user(current_user: dict = Depends(get_current_user)):
    """Get current user and verify admin role"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db = Depends(get_database)
):
    """Get current user if authenticated, otherwise return None"""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


async def get_current_user_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    db = Depends(get_database),
    redis = Depends(get_redis)
):
    """Get current user for WebSocket connections"""
    if not token:
        return None
    
    # Check if token is blacklisted
    if redis:
        try:
            is_blacklisted = await redis.get(f"blacklist:{token}")
            if is_blacklisted:
                return None
        except Exception as e:
            logger.warning(f"Failed to check token blacklist: {e}")
    
    try:
        payload = jwt.decode(
            token, 
            settings.secret_key, 
            algorithms=[settings.algorithm]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
    except JWTError:
        return None
    
    # Get user from database
    user = await db.users.find_one({"_id": user_id})
    return user

async def get_api_key(api_key: Optional[str] = Depends(api_key_header)) -> Optional[APIKey]:
    """Validate API key and return APIKey object if valid"""
    if not api_key:
        return None
    
    validated_key = await api_key_manager.validate_api_key(api_key)
    return validated_key


async def require_api_key(api_key: APIKey = Depends(get_api_key)) -> APIKey:
    """Require valid API key for access"""
    if not api_key:
        raise AuthenticationError("Valid API key required")
    return api_key


def require_api_scope(required_scope: APIKeyScope):
    """Dependency factory to require specific API key scope"""
    async def check_scope(api_key: APIKey = Depends(require_api_key)) -> APIKey:
        if not api_key_manager.has_scope(api_key, required_scope):
            raise AuthorizationError(f"API key missing required scope: {required_scope.value}")
        return api_key
    return check_scope


async def get_current_user_or_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    api_key: Optional[APIKey] = Depends(get_api_key),
    db = Depends(get_database),
    redis = Depends(get_redis)
) -> Union[dict, APIKey]:
    """Get current user via JWT token or API key"""
    
    # Try API key first
    if api_key:
        return api_key
    
    # Fall back to JWT authentication
    if not credentials:
        raise AuthenticationError("Authentication required (JWT token or API key)")
    
    return await get_current_user(credentials, db, redis)


def require_user_or_api_scope(required_scope: APIKeyScope):
    """Require either authenticated user or API key with specific scope"""
    async def check_auth(
        auth: Union[dict, APIKey] = Depends(get_current_user_or_api_key)
    ) -> Union[dict, APIKey]:
        
        # If it's an API key, check scope
        if isinstance(auth, APIKey):
            if not api_key_manager.has_scope(auth, required_scope):
                raise AuthorizationError(f"API key missing required scope: {required_scope.value}")
        
        # If it's a user, they have access (additional role checks can be added here)
        return auth
    
    return check_auth