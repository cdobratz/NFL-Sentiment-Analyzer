import pymongo
import redis
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import logging
from .config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        self.mongodb_client: Optional[AsyncIOMotorClient] = None
        self.redis_client: Optional[redis.Redis] = None
        self.db = None
    
    async def connect_mongodb(self):
        """Connect to MongoDB Atlas"""
        try:
            self.mongodb_client = AsyncIOMotorClient(settings.mongodb_url)
            # Verify connection
            await self.mongodb_client.admin.command('ping')
            self.db = self.mongodb_client[settings.database_name]
            logger.info("Successfully connected to MongoDB Atlas!")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB Atlas: {e}")
            raise
    
    async def connect_redis(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(settings.redis_url, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis!")
        except Exception as e:
            logger.error(f"Error connecting to Redis: {e}")
            # Redis is optional, so we don't raise here
            self.redis_client = None
    
    async def disconnect(self):
        """Disconnect from databases"""
        if self.mongodb_client:
            self.mongodb_client.close()
        if self.redis_client:
            self.redis_client.close()
    
    def get_database(self):
        """Get MongoDB database instance"""
        return self.db
    
    def get_redis(self):
        """Get Redis client instance"""
        return self.redis_client


# Global database manager instance
db_manager = DatabaseManager()


async def get_database():
    """Dependency to get database instance"""
    return db_manager.get_database()


async def get_redis():
    """Dependency to get Redis instance"""
    return db_manager.get_redis()