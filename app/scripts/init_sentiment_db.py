"""
Database initialization script for sentiment analysis collections and indexes.
"""

import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient

from ..core.config import settings
from ..core.database_indexes import create_all_indexes

logger = logging.getLogger(__name__)


async def initialize_sentiment_database():
    """Initialize database with proper collections and indexes for sentiment analysis"""
    
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient(settings.MONGODB_URL)
        db = client[settings.DATABASE_NAME]
        
        print("🗄️  Initializing NFL Sentiment Analysis Database")
        print("=" * 50)
        
        # Create indexes
        print("\n1. Creating database indexes...")
        index_results = await create_all_indexes(db)
        
        for collection, indexes in index_results.items():
            print(f"  ✅ {collection}: {len(indexes)} indexes created")
        
        # Verify collections exist
        print("\n2. Verifying collections...")
        collections = await db.list_collection_names()
        
        required_collections = [
            "sentiment_analysis",
            "team_sentiment", 
            "player_sentiment",
            "game_sentiment",
            "teams",
            "players", 
            "games"
        ]
        
        for collection in required_collections:
            if collection in collections:
                count = await db[collection].count_documents({})
                print(f"  ✅ {collection}: {count} documents")
            else:
                print(f"  ⚠️  {collection}: Collection will be created on first insert")
        
        # Test database connection
        print("\n3. Testing database connection...")
        server_info = await client.server_info()
        print(f"  ✅ Connected to MongoDB {server_info['version']}")
        
        print("\n" + "=" * 50)
        print("✅ Database initialization completed successfully!")
        
        # Close connection
        client.close()
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        print(f"❌ Database initialization failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run the initialization
    asyncio.run(initialize_sentiment_database())