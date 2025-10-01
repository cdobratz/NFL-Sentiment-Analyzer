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
        self.collection_name = "api_keys"
    
    def _hash_key(self, key: str) -> str:
        """Hash an API key for secure storage"""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _generate_key(self) -> str:
        """Generate a new API key"""
        return f"nfl_{secrets.token_urlsafe(32)}"
    
    async def create_api_key(
        self,
        name: str,
        scopes: List[APIKeyScope],
        created_by: str,
        expires_in_days: Optional[int] = None,
        rate_limit: int = 1000,
        metadata: Dict[str, Any] = None
    ) -> tuple[str, APIKey]:
        """Create a new API key"""
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
                metadata=metadata or {}
            )
            
            # Store in database
            db = db_manager.get_database()
            collection = db[self.collection_name]
            
            await collection.insert_one(api_key.dict())
            
            logger.info(f"Created API key: {name}", extra={
                "api_key_id": api_key.id,
                "created_by": created_by,
                "scopes": [s.value for s in scopes]
            })
            
            return key, api_key
            
        except Exception as e:
            logger.error(f"Failed to create API key: {e}")
            raise ValidationError(f"Failed to create API key: {str(e)}")
    
    async def validate_api_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key and return the key object if valid"""
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
                    {"$set": {"status": APIKeyStatus.EXPIRED.value}}
                )
                return None
            
            # Update usage
            await collection.update_one(
                {"_id": key_doc["_id"]},
                {
                    "$set": {"last_used_at": datetime.utcnow()},
                    "$inc": {"usage_count": 1}
                }
            )
            
            return api_key
            
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return None
    
    async def revoke_api_key(self, key_id: str, revoked_by: str) -> bool:
        """Revoke an API key"""
        try:
            db = db_manager.get_database()
            collection = db[self.collection_name]
            
            result = await collection.update_one(
                {"id": key_id},
                {"$set": {"status": APIKeyStatus.REVOKED.value}}
            )
            
            if result.modified_count > 0:
                logger.info(f"Revoked API key: {key_id}", extra={
                    "revoked_by": revoked_by
                })
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to revoke API key: {e}")
            return False
    
    async def list_api_keys(self, created_by: Optional[str] = None) -> List[APIKey]:
        """List API keys, optionally filtered by creator"""
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
        """Get usage statistics for an API key"""
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
                "last_used": api_key.last_used_at.isoformat() if api_key.last_used_at else None,
                "rate_limit": api_key.rate_limit,
                "status": api_key.status.value
            }
            
        except Exception as e:
            logger.error(f"Failed to get API key usage: {e}")
            return {}
    
    def has_scope(self, api_key: APIKey, required_scope: APIKeyScope) -> bool:
        """Check if API key has required scope"""
        return required_scope in api_key.scopes or APIKeyScope.ADMIN in api_key.scopes


# Global API key manager instance
api_key_manager = APIKeyManager()