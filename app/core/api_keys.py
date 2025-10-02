"""
API key management system for third-party integrations.
"""

import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from enum import Enum
import logging

from .database import db_manager
from .exceptions import AuthenticationError, AuthorizationError, ValidationError

logger = logging.getLogger(__name__)


class APIKeyScope(Enum):
    """API key permission scopes"""

    READ_SENTIMENT = "read:sentiment"
    WRITE_SENTIMENT = "write:sentiment"
    READ_DATA = "read:data"
    WRITE_DATA = "write:data"
    READ_ANALYTICS = "read:analytics"
    ADMIN = "admin"


class APIKeyStatus(Enum):
    """API key status"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"


class APIKey(BaseModel):
    """API key model"""

    id: str
    name: str
    key_hash: str
    scopes: List[APIKeyScope]
    status: APIKeyStatus
    created_at: datetime
    expires_at: Optional[datetime]
    last_used_at: Optional[datetime]
    usage_count: int = 0
    rate_limit: int = 1000  # requests per hour
    created_by: str  # user ID
    metadata: Dict[str, Any] = {}


class APIKeyManager:
    """Manages API keys for third-party access"""

    def __init__(self):
        """
        Initialize the APIKeyManager.

        Sets the `collection_name` attribute to "api_keys", the database collection used for storing API key records.
        """
        self.collection_name = "api_keys"

    def _hash_key(self, key: str) -> str:
        """
        Hash an API key for secure storage.

        Returns:
            Hexadecimal SHA-256 hash of the provided key.
        """
        return hashlib.sha256(key.encode()).hexdigest()

    def _generate_key(self) -> str:
        """
        Generate a new API key string with a fixed prefix and a URL-safe random token.

        Returns:
            api_key (str): An API key string beginning with "nfl_" followed by a URL-safe random token.
        """
        return f"nfl_{secrets.token_urlsafe(32)}"

    async def create_api_key(
        self,
        name: str,
        scopes: List[APIKeyScope],
        created_by: str,
        expires_in_days: Optional[int] = None,
        rate_limit: int = 1000,
        metadata: Dict[str, Any] = None,
    ) -> tuple[str, APIKey]:
        """
        Generate, persist, and return a new API key record along with its plaintext key.

        Parameters:
            name (str): Human-readable name for the API key.
            scopes (List[APIKeyScope]): Permission scopes granted to the key.
            created_by (str): Identifier of the user that creates the key.
            expires_in_days (Optional[int]): Number of days until the key expires; if omitted the key does not expire.
            rate_limit (int): Allowed requests per hour for the key.
            metadata (Dict[str, Any]): Optional additional data to store with the key.

        Returns:
            tuple[str, APIKey]: A tuple containing the generated plaintext API key and the persisted APIKey model.

        Raises:
            ValidationError: If key generation or persistence fails.
        """
        try:
            # Generate key
            key = self._generate_key()
            key_hash = self._hash_key(key)

            # Calculate expiration
            expires_at = None
            if expires_in_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

            # Create API key object
            api_key = APIKey(
                id=secrets.token_urlsafe(16),
                name=name,
                key_hash=key_hash,
                scopes=scopes,
                status=APIKeyStatus.ACTIVE,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                last_used_at=None,
                usage_count=0,
                rate_limit=rate_limit,
                created_by=created_by,
                metadata=metadata or {},
            )

            # Store in database
            db = db_manager.get_database()
            collection = db[self.collection_name]

            await collection.insert_one(api_key.dict())

            logger.info(
                f"Created API key: {name}",
                extra={
                    "api_key_id": api_key.id,
                    "created_by": created_by,
                    "scopes": [s.value for s in scopes],
                },
            )

            return key, api_key

        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise ValidationError(f"Failed to create API key: {str(e)}")

    async def validate_api_key(self, key: str) -> Optional[APIKey]:
        """
        Validate an API key and return its APIKey record when it is active and not expired.

        Checks the provided key against stored keys; if a matching key is found and its status is ACTIVE and its expires_at (if set) is in the future, updates the key's last_used_at and usage_count and returns the corresponding APIKey. If the key is expired, updates the stored status to EXPIRED and returns None. Returns None when the key is not found, not active, expired, or if an error occurs.

        Returns:
            APIKey or None: `APIKey` instance when the key is valid, `None` otherwise.
        """
        try:
            key_hash = self._hash_key(key)

            # Find key in database
            db = db_manager.get_database()
            collection = db[self.collection_name]

            key_doc = await collection.find_one({"key_hash": key_hash})
            if not key_doc:
                return None

            api_key = APIKey(**key_doc)

            # Check status
            if api_key.status != APIKeyStatus.ACTIVE:
                return None

            # Check expiration
            if api_key.expires_at and api_key.expires_at < datetime.utcnow():
                # Mark as expired
                await collection.update_one(
                    {"_id": key_doc["_id"]},
                    {"$set": {"status": APIKeyStatus.EXPIRED.value}},
                )
                return None

            # Update usage
            await collection.update_one(
                {"_id": key_doc["_id"]},
                {
                    "$set": {"last_used_at": datetime.utcnow()},
                    "$inc": {"usage_count": 1},
                },
            )

            return api_key

        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return None

    async def revoke_api_key(self, key_id: str, revoked_by: str) -> bool:
        """
        Mark an API key as revoked in the persistent store.

        Parameters:
            key_id (str): Identifier of the API key to revoke.
            revoked_by (str): User ID of the actor performing the revocation (used for logging).

        Returns:
            `True` if the key's status was updated to revoked, `False` otherwise.
        """
        try:
            db = db_manager.get_database()
            collection = db[self.collection_name]

            result = await collection.update_one(
                {"id": key_id}, {"$set": {"status": APIKeyStatus.REVOKED.value}}
            )

            if result.modified_count > 0:
                logger.info(
                    f"Revoked API key: {key_id}", extra={"revoked_by": revoked_by}
                )
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            return False

    async def list_api_keys(self, created_by: Optional[str] = None) -> List[APIKey]:
        """
        List API keys, optionally filtered by the creator.

        Parameters:
            created_by (Optional[str]): If provided, only return keys created by this user ID; if omitted, return all keys.

        Returns:
            List[APIKey]: APIKey instances ordered by creation time (newest first).
        """
        try:
            db = db_manager.get_database()
            collection = db[self.collection_name]

            query = {}
            if created_by:
                query["created_by"] = created_by

            cursor = collection.find(query).sort("created_at", -1)
            keys = []

            async for doc in cursor:
                keys.append(APIKey(**doc))

            return keys

        except Exception as e:
            logger.error(f"Failed to list API keys: {e}")
            return []

    async def get_api_key_usage(self, key_id: str) -> Dict[str, Any]:
        """
        Return usage statistics for the API key identified by `key_id`.

        Parameters:
            key_id (str): The unique identifier of the API key to retrieve usage for.

        Returns:
            Dict[str, Any]: A dictionary containing usage metrics:
                - `total_requests` (int): Total number of requests made with the key.
                - `days_active` (int): Number of days since the key was created (minimum 1).
                - `avg_daily_requests` (float): Average daily requests, rounded to two decimal places.
                - `last_used` (Optional[str]): ISO 8601 timestamp of the last use, or `None` if never used.
                - `rate_limit` (int): Configured requests-per-hour rate limit for the key.
                - `status` (str): Current key status value (e.g., "ACTIVE", "REVOKED").
        """
        try:
            db = db_manager.get_database()
            collection = db[self.collection_name]

            key_doc = await collection.find_one({"id": key_id})
            if not key_doc:
                return {}

            api_key = APIKey(**key_doc)

            # Calculate usage statistics
            now = datetime.utcnow()
            days_active = (now - api_key.created_at).days or 1
            avg_daily_usage = api_key.usage_count / days_active

            return {
                "total_requests": api_key.usage_count,
                "days_active": days_active,
                "avg_daily_requests": round(avg_daily_usage, 2),
                "last_used": (
                    api_key.last_used_at.isoformat() if api_key.last_used_at else None
                ),
                "rate_limit": api_key.rate_limit,
                "status": api_key.status.value,
            }

        except Exception as e:
            logger.error(f"Failed to get API key usage: {e}")
            return {}

    def has_scope(self, api_key: APIKey, required_scope: APIKeyScope) -> bool:
        """
        Determine whether the given API key grants the required permission scope; `ADMIN` counts as granting all scopes.

        Parameters:
            api_key (APIKey): The API key record to check.
            required_scope (APIKeyScope): The permission scope required.

        Returns:
            bool: `True` if `required_scope` is present in `api_key.scopes` or `APIKeyScope.ADMIN` is present, `False` otherwise.
        """
        return required_scope in api_key.scopes or APIKeyScope.ADMIN in api_key.scopes


# Global API key manager instance
api_key_manager = APIKeyManager()
