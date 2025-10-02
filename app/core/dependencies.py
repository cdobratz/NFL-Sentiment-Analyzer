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
    db=Depends(get_database),
    redis=Depends(get_redis),
):
    """
    Authenticate the request's Bearer JWT (including a Redis blacklist check) and return the corresponding user document.
    
    Returns:
        The user document retrieved from the database.
    
    Raises:
        HTTPException: 401 Unauthorized when the token is missing, invalid, blacklisted, or the user does not exist.
    """
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
            token, settings.secret_key, algorithms=[settings.algorithm]
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
    """
    Ensure the current authenticated user has the "admin" role.
    
    Returns:
        dict: The authenticated user's document.
    
    Raises:
        HTTPException: If the user's role is not "admin" (403 Forbidden).
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )
    return current_user


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db=Depends(get_database),
):
    """
    Return the authenticated user dict when valid credentials are provided, otherwise None.
    
    Returns:
        Optional[dict]: The user document from the database if authentication succeeds, `None` otherwise.
    """
    if not credentials:
        return None

    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


async def get_current_user_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    db=Depends(get_database),
    redis=Depends(get_redis),
):
    """
    Resolve the authenticated user for a WebSocket connection using a JWT provided in the connection's query parameters.
    
    Checks the token against a Redis-backed blacklist and decodes the JWT to obtain the user identifier, then returns the matching user document from the database.
    
    Parameters:
        token (Optional[str]): JWT provided as the `token` query parameter used to authenticate the WebSocket connection.
    
    Returns:
        The user document from the database if the token is valid and the user exists, `None` otherwise.
    """
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
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
    except JWTError:
        return None

    # Get user from database
    user = await db.users.find_one({"_id": user_id})
    return user


async def get_api_key(
    api_key: Optional[str] = Depends(api_key_header),
) -> Optional[APIKey]:
    """
    Validate the provided API key header and return the corresponding APIKey when valid.
    
    Returns:
        APIKey or None: The validated APIKey object if the header contains a valid key, `None` if no key was provided or validation failed.
    """
    if not api_key:
        return None

    validated_key = await api_key_manager.validate_api_key(api_key)
    return validated_key


async def require_api_key(api_key: APIKey = Depends(get_api_key)) -> APIKey:
    """
    Enforces presence of a valid API key for an endpoint.
    
    Returns:
        APIKey: The validated API key.
    
    Raises:
        AuthenticationError: If no valid API key is provided.
    """
    if not api_key:
        raise AuthenticationError("Valid API key required")
    return api_key


def require_api_scope(required_scope: APIKeyScope):
    """
    Create a dependency that enforces an API key has the specified scope.
    
    Parameters:
        required_scope (APIKeyScope): The scope that an API key must include to be accepted.
    
    Returns:
        Callable: A dependency function that validates the current API key includes `required_scope` and returns the `APIKey` when validation succeeds. Raises `AuthorizationError` if the API key is missing the required scope.
    """

    async def check_scope(api_key: APIKey = Depends(require_api_key)) -> APIKey:
        """
        Enforces that the provided API key includes the required scope.
        
        Raises AuthorizationError if the API key does not have the required scope.
        
        Parameters:
            api_key (APIKey): The API key to validate (injected dependency).
        
        Returns:
            APIKey: The validated API key when it contains the required scope.
        """
        if not api_key_manager.has_scope(api_key, required_scope):
            raise AuthorizationError(
                f"API key missing required scope: {required_scope.value}"
            )
        return api_key

    return check_scope


async def get_current_user_or_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    api_key: Optional[APIKey] = Depends(get_api_key),
    db=Depends(get_database),
    redis=Depends(get_redis),
) -> Union[dict, APIKey]:
    """
    Authenticate the request by returning an API key or a user object obtained from a JWT.
    
    Prefers a valid `APIKey` when present; if no API key is provided, requires JWT credentials and returns the associated user document. Raises `AuthenticationError` if neither an API key nor JWT credentials are present.
    
    Returns:
        `APIKey` when an API key is present, otherwise the user document (`dict`) resolved from the JWT.
    """

    # Try API key first
    if api_key:
        return api_key

    # Fall back to JWT authentication
    if not credentials:
        raise AuthenticationError("Authentication required (JWT token or API key)")

    return await get_current_user(credentials, db, redis)


def require_user_or_api_scope(required_scope: APIKeyScope):
    """
    Create a dependency that allows access when either an authenticated user or an API key with a specific scope is present.
    
    Parameters:
        required_scope (APIKeyScope): The API key scope required when an API key is used for authentication.
    
    Returns:
        Callable: A dependency function that accepts either a user dict or an APIKey and returns the authenticated object when authorization succeeds.
    
    Raises:
        AuthorizationError: If an APIKey is provided but does not include the required scope.
    """

    async def check_auth(
        auth: Union[dict, APIKey] = Depends(get_current_user_or_api_key)
    ) -> Union[dict, APIKey]:

        # If it's an API key, check scope
        """
        Ensure the authenticated principal has the required API key scope when applicable.
        
        Parameters:
            auth (Union[dict, APIKey]): Authenticated principal, either a user dictionary or an `APIKey` instance.
        
        Returns:
            Union[dict, APIKey]: The original `auth` object when authorization checks pass.
        
        Raises:
            AuthorizationError: If `auth` is an `APIKey` that does not include the required scope.
        """
        if isinstance(auth, APIKey):
            if not api_key_manager.has_scope(auth, required_scope):
                raise AuthorizationError(
                    f"API key missing required scope: {required_scope.value}"
                )

        # If it's a user, they have access (additional role checks can be added here)
        return auth

    return check_auth
