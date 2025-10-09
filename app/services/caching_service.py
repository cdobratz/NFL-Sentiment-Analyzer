"""
Redis caching service for frequently accessed sentiment data.
Implements intelligent caching strategies for NFL sentiment analysis.
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import redis.asyncio as redis
import logging
from app.core.database import get_redis
from app.models.sentiment import (
    TeamSentiment,
    PlayerSentiment,
    GameSentiment,
    SentimentTrend,
    SentimentResult,
)

logger = logging.getLogger(__name__)


class CachingService:
    """Redis-based caching service for sentiment data"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize the caching service with an optional Redis client and default TTL values.

        Parameters:
            redis_client (Optional[redis.Redis]): An existing Redis client to use for operations. If omitted or None, the service will obtain a client lazily when needed.

        Attributes set:
            redis_client: Stored Redis client or None.
            default_ttl: Default time-to-live in seconds (300).
            long_ttl: TTL for aggregated data in seconds (3600).
            short_ttl: TTL for real-time data in seconds (60).
        """
        self.redis_client = redis_client
        self.default_ttl = 300  # 5 minutes default TTL
        self.long_ttl = 3600  # 1 hour for aggregated data
        self.short_ttl = 60  # 1 minute for real-time data

    async def get_redis_client(self) -> Optional[redis.Redis]:
        """
        Get the Redis client, initializing and caching it if not already set.

        @returns Redis client instance, or `None` if a client could not be obtained.
        """
        if not self.redis_client:
            self.redis_client = get_redis()
        return self.redis_client

    def _serialize_data(self, data: Any) -> str:
        """
        Serialize an object for storage in Redis using safe JSON serialization.

        Parameters:
            data: The object to serialize. Pydantic-like models, dicts, and lists are encoded as JSON.

        Returns:
            str: A JSON string for all supported data types.
        """
        if hasattr(data, "dict"):
            # Pydantic model
            return json.dumps(data.dict())
        elif isinstance(data, (dict, list)):
            return json.dumps(data, default=str)
        else:
            # Convert other objects to JSON-serializable format
            try:
                return json.dumps(data, default=str)
            except (TypeError, ValueError):
                # For non-serializable objects, convert to string representation
                return json.dumps(str(data))

    def _deserialize_data(self, data: str, data_type: str = "json") -> Any:
        """
        Deserialize a Redis-stored string into its original Python object.

        Parameters:
            data (str): String retrieved from Redis. Expected to be JSON text when `data_type` is "json",
                or a hex-encoded pickle representation for other `data_type` values.
            data_type (str): Format of `data`. Use "json" to parse JSON; any other value treats `data`
                as a hex-encoded pickle.

        Returns:
            Any: The deserialized Python object, or `None` if deserialization fails.
        """
        try:
            if data_type == "json":
                return json.loads(data)
            else:
                # Safe JSON deserialization instead of pickle
                try:
                    # Decode hex string to bytes, then to UTF-8 text
                    decoded_bytes = bytes.fromhex(data)
                    decoded_text = decoded_bytes.decode('utf-8')
                    return json.loads(decoded_text)
                except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as decode_error:
                    logger.error(f"Error decoding cached data: {decode_error}")
                    return None
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            return None

    async def set_cache(
        self, key: str, data: Any, ttl: Optional[int] = None, data_type: str = "json"
    ) -> bool:
        """
        Store a value in Redis under the given key with an optional time-to-live (TTL).

        Parameters:
                ttl (int, optional): Time-to-live in seconds for the cached key; if omitted, the service's default_ttl is used.

        Returns:
                True if the value was stored successfully, False if the Redis client is unavailable or an error occurred.
        """
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return False

            serialized_data = self._serialize_data(data)
            ttl = ttl or self.default_ttl

            await redis_client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Error setting cache for key {key}: {e}")
            return False

    async def get_cache(self, key: str, data_type: str = "json") -> Optional[Any]:
        """
        Retrieve and deserialize a cached value by key.

        Parameters:
            data_type (str): Format used when deserializing cached data; `"json"` to JSON-decode, any other value to unpickle from a hex-encoded pickle.

        Returns:
            The deserialized cached value if the key exists, `None` otherwise.
        """
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return None

            data = await redis_client.get(key)
            if data:
                # Decode bytes to string if needed
                if isinstance(data, bytes):
                    data = data.decode('utf-8')
                return self._deserialize_data(data, data_type)
            return None
        except Exception as e:
            logger.error(f"Error getting cache for key {key}: {e}")
            return None

    async def delete_cache(self, key: str) -> bool:
        """
        Remove a specific key from the Redis cache.

        Parameters:
            key (str): The cache key to remove.

        Returns:
            `true` if the delete was performed (Redis client was available and the command was issued), `false` if Redis was unavailable or an error occurred.
        """
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return False

            await redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting cache for key {key}: {e}")
            return False

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all Redis keys that match the given pattern.

        Parameters:
            pattern (str): Glob-style pattern used to match Redis keys (for example, "team_sentiment:*").

        Returns:
            int: Number of keys deleted.
        """
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return 0

            # Use SCAN instead of KEYS to avoid blocking
            keys = []
            async for key in redis_client.scan_iter(match=pattern):
                keys.append(key)
            if keys:
                return await redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error deleting cache pattern {pattern}: {e}")
            return 0

    async def exists(self, key: str) -> bool:
        """
        Determine whether a cache entry exists for the given key.

        @returns `true` if a cache entry with the key exists, `false` otherwise.
        """
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return False

            return bool(await redis_client.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache existence for key {key}: {e}")
            return False

    async def get_ttl(self, key: str) -> int:
        """
        Retrieve the remaining time-to-live (TTL) in seconds for a Redis key.

        Returns:
            Remaining TTL in seconds. Redis may return:
              - a non-negative integer for seconds remaining,
              - -1 if the key exists but has no expiration (or if Redis is unavailable or an error occurs in this method),
              - -2 if the key does not exist.
        """
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return -1

            return await redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return -1

    # Sentiment-specific caching methods

    async def cache_team_sentiment(
        self, team_id: str, sentiment: TeamSentiment, ttl: Optional[int] = None
    ) -> bool:
        """
        Cache sentiment data for a team under a dedicated Redis key.

        Parameters:
            team_id (str): Identifier of the team used to construct the cache key.
            sentiment (TeamSentiment): Team sentiment object to store; will be serialized for Redis.
            ttl (Optional[int]): Time-to-live in seconds for the cached entry; if omitted, the service's long_ttl is used.

        Returns:
            true if the value was stored successfully, false otherwise.
        """
        key = f"team_sentiment:{team_id}"
        return await self.set_cache(key, sentiment, ttl or self.long_ttl)

    async def get_team_sentiment(self, team_id: str) -> Optional[Dict]:
        """
        Retrieve cached sentiment data for a team.

        Returns:
            dict: Cached team sentiment data if present, `None` otherwise.
        """
        key = f"team_sentiment:{team_id}"
        return await self.get_cache(key)

    async def cache_player_sentiment(
        self, player_id: str, sentiment: PlayerSentiment, ttl: Optional[int] = None
    ) -> bool:
        """
        Cache sentiment for a player in Redis under the key "player_sentiment:{player_id}".

        Parameters:
                player_id (str): Player identifier used to form the cache key.
                sentiment (PlayerSentiment): Sentiment object to store.
                ttl (Optional[int]): Time-to-live in seconds; if omitted, uses the service's long_ttl.

        Returns:
                bool: `true` if the value was stored successfully, `false` otherwise.
        """
        key = f"player_sentiment:{player_id}"
        return await self.set_cache(key, sentiment, ttl or self.long_ttl)

    async def get_player_sentiment(self, player_id: str) -> Optional[Dict]:
        """
        Retrieve cached sentiment for a player.

        Returns:
            dict: Sentiment data for the player if present, `None` otherwise.
        """
        key = f"player_sentiment:{player_id}"
        return await self.get_cache(key)

    async def cache_game_sentiment(
        self, game_id: str, sentiment: GameSentiment, ttl: Optional[int] = None
    ) -> bool:
        """
        Cache sentiment for a specific game in Redis under key "game_sentiment:{game_id}".

        Parameters:
            game_id (str): Identifier of the game.
            sentiment (GameSentiment): Sentiment payload to store; will be serialized for caching.
            ttl (Optional[int]): Time-to-live in seconds; uses the service's long_ttl when omitted.

        Returns:
            bool: `True` if the value was stored successfully, `False` otherwise.
        """
        key = f"game_sentiment:{game_id}"
        return await self.set_cache(key, sentiment, ttl or self.long_ttl)

    async def get_game_sentiment(self, game_id: str) -> Optional[Dict]:
        """
        Retrieve cached sentiment data for a game.

        Returns:
            Cached game sentiment dictionary if present, `None` otherwise.
        """
        key = f"game_sentiment:{game_id}"
        return await self.get_cache(key)

    async def cache_sentiment_trends(
        self,
        entity_type: str,
        entity_id: str,
        trends: List[SentimentTrend],
        period: str = "24h",
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache sentiment trend objects for a specific entity and time period.

        Parameters:
                entity_type (str): Entity category (e.g., "team", "player", "game").
                entity_id (str): Unique identifier of the entity.
                trends (List[SentimentTrend]): Ordered list of sentiment trend entries to cache.
                period (str): Time window label for the trends (default: "24h").
                ttl (Optional[int]): Cache time-to-live in seconds; uses the service default when omitted.

        Returns:
                `true` if the cache was set successfully, `false` otherwise.
        """
        key = f"sentiment_trends:{entity_type}:{entity_id}:{period}"
        return await self.set_cache(key, trends, ttl or self.default_ttl)

    async def get_sentiment_trends(
        self, entity_type: str, entity_id: str, period: str = "24h"
    ) -> Optional[List]:
        """
        Retrieve cached sentiment trends for a specific entity and period.

        Parameters:
            entity_type (str): Entity category (e.g., "team", "player", "game").
            entity_id (str): Identifier of the entity.
            period (str): Time range identifier for the trends (defaults to "24h").

        Returns:
            A list of sentiment trend entries for the entity and period, or `None` if no cached value exists.
        """
        key = f"sentiment_trends:{entity_type}:{entity_id}:{period}"
        return await self.get_cache(key)

    async def cache_analytics_data(
        self, query_hash: str, data: Dict, ttl: Optional[int] = None
    ) -> bool:
        """
        Cache analytics query results under a key derived from the query hash.

        Parameters:
            query_hash (str): Hash identifying the analytics query.
            data (Dict): Analytics result data to store in cache.
            ttl (Optional[int]): Time-to-live in seconds; when omitted, uses the service's `long_ttl`.

        Returns:
            `true` if the data was stored successfully, `false` otherwise.
        """
        key = f"analytics:{query_hash}"
        return await self.set_cache(key, data, ttl or self.long_ttl)

    async def get_analytics_data(self, query_hash: str) -> Optional[Dict]:
        """
        Retrieve cached analytics results for a given query hash.

        Parameters:
                query_hash (str): Unique hash identifying the analytics query.

        Returns:
                analytics (Optional[Dict]): The cached analytics result dictionary if present, `None` if not found.
        """
        key = f"analytics:{query_hash}"
        return await self.get_cache(key)

    async def cache_leaderboard(
        self, leaderboard_type: str, data: List[Dict], ttl: Optional[int] = None
    ) -> bool:
        """
        Cache leaderboard entries for a specific leaderboard type.

        Parameters:
            leaderboard_type (str): Identifier for the leaderboard (e.g., "top_players", "most_improved").
            data (List[Dict]): List of leaderboard entries to store; each entry is a dictionary of leaderboard fields.
            ttl (Optional[int]): Time-to-live in seconds to override the default cache duration.

        Returns:
            bool: `true` if the data was successfully stored in cache, `false` otherwise.
        """
        key = f"leaderboard:{leaderboard_type}"
        return await self.set_cache(key, data, ttl or self.default_ttl)

    async def get_leaderboard(self, leaderboard_type: str) -> Optional[List]:
        """
        Retrieve cached leaderboard for the given leaderboard type.

        Parameters:
            leaderboard_type (str): Identifier for the leaderboard (for example, "top_players" or "team_rankings").

        Returns:
            Optional[List]: The cached leaderboard as a list of entries, or `None` if no cached value exists.
        """
        key = f"leaderboard:{leaderboard_type}"
        return await self.get_cache(key)

    async def invalidate_entity_cache(self, entity_type: str, entity_id: str):
        """
        Invalidate cached entries related to a specific entity by deleting matching Redis keys.

        Parameters:
            entity_type (str): The type of entity (e.g., "team", "player", "game") used in cache key prefixes.
            entity_id (str): The identifier of the entity whose related cache entries should be removed.

        Description:
            Deletes keys matching patterns for the entity, including:
              - "<entity_type>_sentiment:<entity_id>"
              - "sentiment_trends:<entity_type>:<entity_id>:*"
              - "analytics:*<entity_id>*"
        """
        patterns = [
            f"{entity_type}_sentiment:{entity_id}",
            f"sentiment_trends:{entity_type}:{entity_id}:*",
            f"analytics:*{entity_id}*",
        ]

        for pattern in patterns:
            await self.delete_pattern(pattern)

    async def invalidate_all_sentiment_cache(self):
        """
        Invalidate all sentiment-related cache entries.

        Deletes Redis keys that match patterns for team, player, and game sentiments, sentiment trends, analytics results, and leaderboards.
        """
        patterns = [
            "team_sentiment:*",
            "player_sentiment:*",
            "game_sentiment:*",
            "sentiment_trends:*",
            "analytics:*",
            "leaderboard:*",
        ]

        for pattern in patterns:
            await self.delete_pattern(pattern)

    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Collect Redis cache and keyspace statistics relevant to sentiment caching.

        Returns:
            stats (Dict[str, Any]): Dictionary containing:
                - connected_clients (int): Number of connected Redis clients.
                - used_memory (str): Human-readable memory usage (e.g., "10MB").
                - total_keys (int): Total keys reported for the default DB (db0).
                - key_counts (Dict[str, int]): Counts of keys matching sentiment-related patterns:
                    "team_sentiment:*", "player_sentiment:*", "game_sentiment:*",
                    "sentiment_trends:*", "analytics:*", and "leaderboard:*".
                - hit_rate (float): Ratio of keyspace hits to total keyspace lookups (hits / (hits + misses)).
            Returns an empty dict on error or if Redis is unavailable.
        """
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return {}

            info = await redis_client.info()

            # Count keys by pattern
            key_counts = {}
            patterns = [
                "team_sentiment:*",
                "player_sentiment:*",
                "game_sentiment:*",
                "sentiment_trends:*",
                "analytics:*",
                "leaderboard:*",
            ]

            for pattern in patterns:
                # Use SCAN instead of KEYS to avoid blocking
                keys = []
                async for key in redis_client.scan_iter(match=pattern):
                    keys.append(key)
                key_counts[pattern] = len(keys)

            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "total_keys": info.get("db0", {}).get("keys", 0),
                "key_counts": key_counts,
                "hit_rate": info.get("keyspace_hits", 0)
                / max(info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1),
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


# Global caching service instance
caching_service = CachingService()


async def get_caching_service() -> CachingService:
    """
    Provide the shared CachingService singleton for dependency injection.

    @returns The module-level CachingService instance used by the application.
    """
    return caching_service
