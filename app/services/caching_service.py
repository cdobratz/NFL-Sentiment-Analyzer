"""
Redis caching service for frequently accessed sentiment data.
Implements intelligent caching strategies for NFL sentiment analysis.
"""

import json
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import redis
import logging
from app.core.database import get_redis
from app.models.sentiment import (
    TeamSentiment, PlayerSentiment, GameSentiment,
    SentimentTrend, SentimentResult
)

logger = logging.getLogger(__name__)


class CachingService:
    """Redis-based caching service for sentiment data"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client
        self.default_ttl = 300  # 5 minutes default TTL
        self.long_ttl = 3600    # 1 hour for aggregated data
        self.short_ttl = 60     # 1 minute for real-time data
    
    async def get_redis_client(self) -> Optional[redis.Redis]:
        """Get Redis client instance"""
        if not self.redis_client:
            self.redis_client = await get_redis()
        return self.redis_client
    
    def _serialize_data(self, data: Any) -> str:
        """Serialize data for Redis storage"""
        if hasattr(data, 'dict'):
            # Pydantic model
            return json.dumps(data.dict())
        elif isinstance(data, (dict, list)):
            return json.dumps(data, default=str)
        else:
            # Use pickle for complex objects
            return pickle.dumps(data).hex()
    
    def _deserialize_data(self, data: str, data_type: str = "json") -> Any:
        """Deserialize data from Redis"""
        try:
            if data_type == "json":
                return json.loads(data)
            else:
                return pickle.loads(bytes.fromhex(data))
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            return None
    
    async def set_cache(
        self, 
        key: str, 
        data: Any, 
        ttl: Optional[int] = None,
        data_type: str = "json"
    ) -> bool:
        """Set cache with optional TTL"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return False
            
            serialized_data = self._serialize_data(data)
            ttl = ttl or self.default_ttl
            
            redis_client.setex(key, ttl, serialized_data)
            return True
        except Exception as e:
            logger.error(f"Error setting cache for key {key}: {e}")
            return False
    
    async def get_cache(self, key: str, data_type: str = "json") -> Optional[Any]:
        """Get data from cache"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return None
            
            data = redis_client.get(key)
            if data:
                return self._deserialize_data(data, data_type)
            return None
        except Exception as e:
            logger.error(f"Error getting cache for key {key}: {e}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Delete cache entry"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return False
            
            redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting cache for key {key}: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return 0
            
            keys = redis_client.keys(pattern)
            if keys:
                return redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error deleting cache pattern {pattern}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return False
            
            return bool(redis_client.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache existence for key {key}: {e}")
            return False
    
    async def get_ttl(self, key: str) -> int:
        """Get TTL for a key"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return -1
            
            return redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return -1
    
    # Sentiment-specific caching methods
    
    async def cache_team_sentiment(
        self, 
        team_id: str, 
        sentiment: TeamSentiment,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache team sentiment data"""
        key = f"team_sentiment:{team_id}"
        return await self.set_cache(key, sentiment, ttl or self.long_ttl)
    
    async def get_team_sentiment(self, team_id: str) -> Optional[Dict]:
        """Get cached team sentiment"""
        key = f"team_sentiment:{team_id}"
        return await self.get_cache(key)
    
    async def cache_player_sentiment(
        self, 
        player_id: str, 
        sentiment: PlayerSentiment,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache player sentiment data"""
        key = f"player_sentiment:{player_id}"
        return await self.set_cache(key, sentiment, ttl or self.long_ttl)
    
    async def get_player_sentiment(self, player_id: str) -> Optional[Dict]:
        """Get cached player sentiment"""
        key = f"player_sentiment:{player_id}"
        return await self.get_cache(key)
    
    async def cache_game_sentiment(
        self, 
        game_id: str, 
        sentiment: GameSentiment,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache game sentiment data"""
        key = f"game_sentiment:{game_id}"
        return await self.set_cache(key, sentiment, ttl or self.long_ttl)
    
    async def get_game_sentiment(self, game_id: str) -> Optional[Dict]:
        """Get cached game sentiment"""
        key = f"game_sentiment:{game_id}"
        return await self.get_cache(key)
    
    async def cache_sentiment_trends(
        self, 
        entity_type: str, 
        entity_id: str, 
        trends: List[SentimentTrend],
        period: str = "24h",
        ttl: Optional[int] = None
    ) -> bool:
        """Cache sentiment trends"""
        key = f"sentiment_trends:{entity_type}:{entity_id}:{period}"
        return await self.set_cache(key, trends, ttl or self.default_ttl)
    
    async def get_sentiment_trends(
        self, 
        entity_type: str, 
        entity_id: str, 
        period: str = "24h"
    ) -> Optional[List]:
        """Get cached sentiment trends"""
        key = f"sentiment_trends:{entity_type}:{entity_id}:{period}"
        return await self.get_cache(key)
    
    async def cache_analytics_data(
        self, 
        query_hash: str, 
        data: Dict,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache analytics query results"""
        key = f"analytics:{query_hash}"
        return await self.set_cache(key, data, ttl or self.long_ttl)
    
    async def get_analytics_data(self, query_hash: str) -> Optional[Dict]:
        """Get cached analytics data"""
        key = f"analytics:{query_hash}"
        return await self.get_cache(key)
    
    async def cache_leaderboard(
        self, 
        leaderboard_type: str, 
        data: List[Dict],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache leaderboard data"""
        key = f"leaderboard:{leaderboard_type}"
        return await self.set_cache(key, data, ttl or self.default_ttl)
    
    async def get_leaderboard(self, leaderboard_type: str) -> Optional[List]:
        """Get cached leaderboard"""
        key = f"leaderboard:{leaderboard_type}"
        return await self.get_cache(key)
    
    async def invalidate_entity_cache(self, entity_type: str, entity_id: str):
        """Invalidate all cache entries for an entity"""
        patterns = [
            f"{entity_type}_sentiment:{entity_id}",
            f"sentiment_trends:{entity_type}:{entity_id}:*",
            f"analytics:*{entity_id}*"
        ]
        
        for pattern in patterns:
            await self.delete_pattern(pattern)
    
    async def invalidate_all_sentiment_cache(self):
        """Invalidate all sentiment-related cache"""
        patterns = [
            "team_sentiment:*",
            "player_sentiment:*", 
            "game_sentiment:*",
            "sentiment_trends:*",
            "analytics:*",
            "leaderboard:*"
        ]
        
        for pattern in patterns:
            await self.delete_pattern(pattern)
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            redis_client = await self.get_redis_client()
            if not redis_client:
                return {}
            
            info = redis_client.info()
            
            # Count keys by pattern
            key_counts = {}
            patterns = [
                "team_sentiment:*",
                "player_sentiment:*",
                "game_sentiment:*", 
                "sentiment_trends:*",
                "analytics:*",
                "leaderboard:*"
            ]
            
            for pattern in patterns:
                keys = redis_client.keys(pattern)
                key_counts[pattern] = len(keys)
            
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "total_keys": info.get("db0", {}).get("keys", 0),
                "key_counts": key_counts,
                "hit_rate": info.get("keyspace_hits", 0) / max(
                    info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0), 1
                )
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {}


# Global caching service instance
caching_service = CachingService()


async def get_caching_service() -> CachingService:
    """Dependency to get caching service"""
    return caching_service