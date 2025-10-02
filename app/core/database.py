import pymongo
import redis
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
import logging
from .config import settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self):
        """
        Initialize a DatabaseManager with all client and database slots unset.

        Attributes:
            mongodb_client (Optional[AsyncIOMotorClient]): MongoDB async client instance or None until connected.
            redis_client (Optional[redis.Redis]): Redis client instance or None until connected.
            db: MongoDB database handle or None until a database is selected after connection.
        """
        self.mongodb_client: Optional[AsyncIOMotorClient] = None
        self.redis_client: Optional[redis.Redis] = None
        self.db = None

    async def connect_mongodb(self):
        """
        Establishes and verifies a connection to MongoDB Atlas and stores the client and database on the instance.

        Sets self.mongodb_client to an AsyncIOMotorClient connected using settings.mongodb_url and sets self.db to the database named by settings.database_name.

        Raises:
            Exception: If the connection or ping verification fails, the original exception is re-raised.
        """
        try:
            self.mongodb_client = AsyncIOMotorClient(settings.mongodb_url)
            # Verify connection
            await self.mongodb_client.admin.command("ping")
            self.db = self.mongodb_client[settings.database_name]
            logger.info("Successfully connected to MongoDB Atlas!")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB Atlas: {e}")
            raise

    async def connect_redis(self):
        """
        Establishes a Redis client from settings.redis_url and assigns it to self.redis_client.

        On success, self.redis_client is a connected redis client. On failure, the error is logged, self.redis_client is set to None, and the exception is not raised.
        """
        try:
            self.redis_client = redis.from_url(
                settings.redis_url, decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info("Successfully connected to Redis!")
        except Exception as e:
            logger.error(f"Error connecting to Redis: {e}")
            # Redis is optional, so we don't raise here
            self.redis_client = None

    async def disconnect(self):
        """
        Close any active MongoDB and Redis client connections managed by this instance.

        If a client is not set, it is ignored. This method does not return a value.
        """
        if self.mongodb_client:
            self.mongodb_client.close()
        if self.redis_client:
            self.redis_client.close()

    def get_database(self):
        """
        Return the configured MongoDB database handle.

        Returns:
            Optional[pymongo.database.Database]: The MongoDB database instance, or `None` if no connection has been established.
        """
        return self.db

    def get_redis(self):
        """
        Return the configured Redis client used by this manager.

        Returns:
            redis.Redis | None: The Redis client instance if connected, otherwise `None`.
        """
        return self.redis_client


# Global database manager instance
db_manager = DatabaseManager()


async def get_database():
    """
    Provide the current MongoDB database instance for dependency injection.

    Returns:
        The active MongoDB database handle, or `None` if no connection has been established.
    """
    return db_manager.get_database()


async def get_redis():
    """
    Retrieve the shared Redis client used by the application.

    Returns:
        redis_client (redis.Redis | None): The Redis client instance or `None` if Redis is unavailable.
    """
    return db_manager.get_redis()
