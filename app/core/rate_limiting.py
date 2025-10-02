"""
Advanced rate limiting system with user-based quotas and Redis backend.
"""

import time
import asyncio
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging

from .database import db_manager
from .config import settings
from .exceptions import RateLimitError
from .api_keys import APIKey

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    """Types of rate limits"""

    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"


class RateLimiter:
    """Advanced rate limiter with Redis backend and user-specific quotas"""

    def __init__(self):
        self.default_limits = {
            RateLimitType.PER_MINUTE: settings.rate_limit_requests,
            RateLimitType.PER_HOUR: settings.rate_limit_requests * 60,
            RateLimitType.PER_DAY: settings.rate_limit_requests * 60 * 24,
        }

        # User role based limits
        self.role_limits = {
            "user": {
                RateLimitType.PER_MINUTE: 100,
                RateLimitType.PER_HOUR: 5000,
                RateLimitType.PER_DAY: 50000,
            },
            "admin": {
                RateLimitType.PER_MINUTE: 1000,
                RateLimitType.PER_HOUR: 50000,
                RateLimitType.PER_DAY: 500000,
            },
        }

    def _get_window_seconds(self, limit_type: RateLimitType) -> int:
        """Get window duration in seconds for rate limit type"""
        if limit_type == RateLimitType.PER_MINUTE:
            return 60
        elif limit_type == RateLimitType.PER_HOUR:
            return 3600
        elif limit_type == RateLimitType.PER_DAY:
            return 86400
        return 60

    def _get_redis_key(
        self, identifier: str, limit_type: RateLimitType, endpoint: str = None
    ) -> str:
        """Generate Redis key for rate limiting"""
        key_parts = ["rate_limit", identifier, limit_type.value]
        if endpoint:
            key_parts.append(endpoint)
        return ":".join(key_parts)

    async def _get_current_count(self, redis_key: str) -> int:
        """Get current request count from Redis"""
        try:
            redis = db_manager.get_redis()
            count = await redis.get(redis_key)
            return int(count) if count else 0
        except Exception as e:
            logger.error(f"Failed to get rate limit count: {e}")
            return 0

    async def _increment_count(self, redis_key: str, window_seconds: int) -> int:
        """Increment request count in Redis with expiration"""
        try:
            redis = db_manager.get_redis()

            # Use pipeline for atomic operations
            pipe = redis.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, window_seconds)
            results = await pipe.execute()

            return results[0]
        except Exception as e:
            logger.error(f"Failed to increment rate limit count: {e}")
            return 1

    def _get_user_limits(self, user: dict) -> Dict[RateLimitType, int]:
        """Get rate limits for a user based on their role"""
        role = user.get("role", "user")
        return self.role_limits.get(role, self.role_limits["user"])

    def _get_api_key_limits(self, api_key: APIKey) -> Dict[RateLimitType, int]:
        """Get rate limits for an API key"""
        # API keys have custom rate limits
        rate_limit_per_hour = api_key.rate_limit

        return {
            RateLimitType.PER_MINUTE: min(rate_limit_per_hour // 60, 100),
            RateLimitType.PER_HOUR: rate_limit_per_hour,
            RateLimitType.PER_DAY: rate_limit_per_hour * 24,
        }

    async def check_rate_limit(
        self,
        identifier: str,
        limit_type: RateLimitType = RateLimitType.PER_MINUTE,
        endpoint: str = None,
        user: dict = None,
        api_key: APIKey = None,
    ) -> Tuple[bool, Dict[str, int]]:
        """
        Check if request is within rate limits.

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        try:
            # Determine limits based on user/API key
            if api_key:
                limits = self._get_api_key_limits(api_key)
                identifier = f"api_key:{api_key.id}"
            elif user:
                limits = self._get_user_limits(user)
                identifier = f"user:{user['_id']}"
            else:
                limits = self.default_limits
                identifier = f"ip:{identifier}"

            limit = limits[limit_type]
            window_seconds = self._get_window_seconds(limit_type)
            redis_key = self._get_redis_key(identifier, limit_type, endpoint)

            # Get current count
            current_count = await self._get_current_count(redis_key)

            # Check if limit exceeded
            if current_count >= limit:
                # Calculate reset time
                redis = db_manager.get_redis()
                ttl = await redis.ttl(redis_key)
                reset_time = int(time.time()) + (ttl if ttl > 0 else window_seconds)

                return False, {
                    "limit": limit,
                    "remaining": 0,
                    "reset": reset_time,
                    "retry_after": ttl if ttl > 0 else window_seconds,
                }

            # Increment count
            new_count = await self._increment_count(redis_key, window_seconds)
            remaining = max(0, limit - new_count)

            # Calculate reset time
            reset_time = int(time.time()) + window_seconds

            return True, {
                "limit": limit,
                "remaining": remaining,
                "reset": reset_time,
                "retry_after": 0,
            }

        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if rate limiting fails
            return True, {
                "limit": 100,
                "remaining": 99,
                "reset": int(time.time()) + 60,
                "retry_after": 0,
            }

    async def check_multiple_limits(
        self,
        identifier: str,
        endpoint: str = None,
        user: dict = None,
        api_key: APIKey = None,
    ) -> Tuple[bool, Dict[str, Dict[str, int]]]:
        """Check multiple rate limit types simultaneously"""

        results = {}
        overall_allowed = True

        # Check all limit types
        for limit_type in [
            RateLimitType.PER_MINUTE,
            RateLimitType.PER_HOUR,
            RateLimitType.PER_DAY,
        ]:
            allowed, info = await self.check_rate_limit(
                identifier, limit_type, endpoint, user, api_key
            )

            results[limit_type.value] = info

            if not allowed:
                overall_allowed = False
                # Don't break - we want to get info for all limits

        return overall_allowed, results

    async def get_rate_limit_status(
        self, identifier: str, user: dict = None, api_key: APIKey = None
    ) -> Dict[str, Dict[str, int]]:
        """Get current rate limit status without incrementing counters"""

        try:
            # Determine limits and identifier
            if api_key:
                limits = self._get_api_key_limits(api_key)
                identifier = f"api_key:{api_key.id}"
            elif user:
                limits = self._get_user_limits(user)
                identifier = f"user:{user['_id']}"
            else:
                limits = self.default_limits
                identifier = f"ip:{identifier}"

            results = {}

            for limit_type, limit in limits.items():
                window_seconds = self._get_window_seconds(limit_type)
                redis_key = self._get_redis_key(identifier, limit_type)

                current_count = await self._get_current_count(redis_key)
                remaining = max(0, limit - current_count)

                # Get TTL for reset time
                redis = db_manager.get_redis()
                ttl = await redis.ttl(redis_key)
                reset_time = int(time.time()) + (ttl if ttl > 0 else window_seconds)

                results[limit_type.value] = {
                    "limit": limit,
                    "remaining": remaining,
                    "reset": reset_time,
                    "current": current_count,
                }

            return results

        except Exception as e:
            logger.error(f"Failed to get rate limit status: {e}")
            return {}


# Global rate limiter instance
rate_limiter = RateLimiter()
