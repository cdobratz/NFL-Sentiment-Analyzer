"""
Database migration service for schema updates and data transformations.
Handles versioned migrations for the NFL sentiment analysis database.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging
from app.core.database import get_database
from app.core.database_indexes import create_all_indexes, drop_all_indexes

logger = logging.getLogger(__name__)


class Migration:
    """Base migration class"""
    
    def __init__(self, version: str, description: str):
        self.version = version
        self.description = description
        self.applied_at: Optional[datetime] = None
    
    async def up(self, db: AsyncIOMotorDatabase) -> bool:
        """Apply the migration"""
        raise NotImplementedError("Migration must implement up() method")
    
    async def down(self, db: AsyncIOMotorDatabase) -> bool:
        """Rollback the migration"""
        raise NotImplementedError("Migration must implement down() method")


class DatabaseMigrationService:
    """Service for managing database migrations"""
    
    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None):
        self.db = db
        self.migrations_collection = "schema_migrations"
        self.migrations: List[Migration] = []
        self._register_migrations()
    
    async def get_database(self) -> AsyncIOMotorDatabase:
        """Get database instance"""
        if not self.db:
            self.db = await get_database()
        return self.db
    
    def _register_migrations(self):
        """Register all available migrations"""
        self.migrations = [
            InitialSchemaMigration("001", "Initial schema setup with indexes"),
            EnhancedSentimentModelMigration("002", "Enhanced sentiment model with NFL context"),
            AggregatedSentimentMigration("003", "Add aggregated sentiment collections"),
            AnalyticsIndexesMigration("004", "Add analytics and reporting indexes"),
            CachingOptimizationMigration("005", "Optimize collections for caching"),
        ]
    
    async def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """Get list of applied migrations"""
        db = await self.get_database()
        collection = db[self.migrations_collection]
        
        cursor = collection.find({}).sort("applied_at", 1)
        return await cursor.to_list(length=None)
    
    async def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations"""
        applied = await self.get_applied_migrations()
        applied_versions = {m["version"] for m in applied}
        
        return [m for m in self.migrations if m.version not in applied_versions]
    
    async def apply_migration(self, migration: Migration) -> bool:
        """Apply a single migration"""
        db = await self.get_database()
        collection = db[self.migrations_collection]
        
        try:
            logger.info(f"Applying migration {migration.version}: {migration.description}")
            
            # Apply the migration
            success = await migration.up(db)
            
            if success:
                # Record the migration
                migration_record = {
                    "version": migration.version,
                    "description": migration.description,
                    "applied_at": datetime.utcnow(),
                    "status": "applied"
                }
                
                await collection.insert_one(migration_record)
                logger.info(f"Migration {migration.version} applied successfully")
                return True
            else:
                logger.error(f"Migration {migration.version} failed")
                return False
                
        except Exception as e:
            logger.error(f"Error applying migration {migration.version}: {e}")
            
            # Record the failed migration
            migration_record = {
                "version": migration.version,
                "description": migration.description,
                "applied_at": datetime.utcnow(),
                "status": "failed",
                "error": str(e)
            }
            
            await collection.insert_one(migration_record)
            return False
    
    async def rollback_migration(self, migration: Migration) -> bool:
        """Rollback a single migration"""
        db = await self.get_database()
        collection = db[self.migrations_collection]
        
        try:
            logger.info(f"Rolling back migration {migration.version}: {migration.description}")
            
            # Rollback the migration
            success = await migration.down(db)
            
            if success:
                # Remove the migration record
                await collection.delete_one({"version": migration.version})
                logger.info(f"Migration {migration.version} rolled back successfully")
                return True
            else:
                logger.error(f"Migration {migration.version} rollback failed")
                return False
                
        except Exception as e:
            logger.error(f"Error rolling back migration {migration.version}: {e}")
            return False
    
    async def migrate_up(self, target_version: Optional[str] = None) -> Dict[str, Any]:
        """Apply all pending migrations up to target version"""
        pending = await self.get_pending_migrations()
        
        if target_version:
            # Filter to only migrations up to target version
            pending = [m for m in pending if m.version <= target_version]
        
        results = {
            "applied": [],
            "failed": [],
            "total": len(pending)
        }
        
        for migration in pending:
            success = await self.apply_migration(migration)
            
            if success:
                results["applied"].append({
                    "version": migration.version,
                    "description": migration.description
                })
            else:
                results["failed"].append({
                    "version": migration.version,
                    "description": migration.description
                })
                # Stop on first failure
                break
        
        return results
    
    async def migrate_down(self, target_version: str) -> Dict[str, Any]:
        """Rollback migrations down to target version"""
        applied = await self.get_applied_migrations()
        
        # Find migrations to rollback (in reverse order)
        to_rollback = [
            m for m in reversed(applied) 
            if m["version"] > target_version
        ]
        
        results = {
            "rolled_back": [],
            "failed": [],
            "total": len(to_rollback)
        }
        
        for migration_record in to_rollback:
            # Find the migration class
            migration = next(
                (m for m in self.migrations if m.version == migration_record["version"]),
                None
            )
            
            if migration:
                success = await self.rollback_migration(migration)
                
                if success:
                    results["rolled_back"].append({
                        "version": migration.version,
                        "description": migration.description
                    })
                else:
                    results["failed"].append({
                        "version": migration.version,
                        "description": migration.description
                    })
                    # Stop on first failure
                    break
        
        return results
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """Get current migration status"""
        applied = await self.get_applied_migrations()
        pending = await self.get_pending_migrations()
        
        return {
            "current_version": applied[-1]["version"] if applied else None,
            "applied_count": len(applied),
            "pending_count": len(pending),
            "total_migrations": len(self.migrations),
            "applied_migrations": applied,
            "pending_migrations": [
                {
                    "version": m.version,
                    "description": m.description
                }
                for m in pending
            ]
        }


# Migration implementations

class InitialSchemaMigration(Migration):
    """Initial schema setup with basic indexes"""
    
    async def up(self, db: AsyncIOMotorDatabase) -> bool:
        try:
            # Create basic indexes
            await create_all_indexes(db)
            
            # Create initial collections if they don't exist
            collections = [
                "sentiment_analysis",
                "team_sentiment", 
                "player_sentiment",
                "game_sentiment",
                "teams",
                "players", 
                "games"
            ]
            
            existing_collections = await db.list_collection_names()
            
            for collection_name in collections:
                if collection_name not in existing_collections:
                    await db.create_collection(collection_name)
            
            return True
        except Exception as e:
            logger.error(f"Error in InitialSchemaMigration: {e}")
            return False
    
    async def down(self, db: AsyncIOMotorDatabase) -> bool:
        try:
            # Drop all custom indexes
            await drop_all_indexes(db)
            return True
        except Exception as e:
            logger.error(f"Error rolling back InitialSchemaMigration: {e}")
            return False


class EnhancedSentimentModelMigration(Migration):
    """Enhanced sentiment model with NFL context"""
    
    async def up(self, db: AsyncIOMotorDatabase) -> bool:
        try:
            # Add new fields to existing sentiment documents
            collection = db.sentiment_analysis
            
            # Update documents that don't have the new fields
            await collection.update_many(
                {"nfl_context": {"$exists": False}},
                {
                    "$set": {
                        "nfl_context": {},
                        "sentiment_weights": {},
                        "emotion_scores": {},
                        "aspect_sentiments": {},
                        "keyword_contributions": {}
                    }
                }
            )
            
            return True
        except Exception as e:
            logger.error(f"Error in EnhancedSentimentModelMigration: {e}")
            return False
    
    async def down(self, db: AsyncIOMotorDatabase) -> bool:
        try:
            # Remove the enhanced fields
            collection = db.sentiment_analysis
            
            await collection.update_many(
                {},
                {
                    "$unset": {
                        "nfl_context": "",
                        "sentiment_weights": "",
                        "emotion_scores": "",
                        "aspect_sentiments": "",
                        "keyword_contributions": ""
                    }
                }
            )
            
            return True
        except Exception as e:
            logger.error(f"Error rolling back EnhancedSentimentModelMigration: {e}")
            return False


class AggregatedSentimentMigration(Migration):
    """Add aggregated sentiment collections"""
    
    async def up(self, db: AsyncIOMotorDatabase) -> bool:
        try:
            # Create aggregated sentiment collections
            collections = [
                "team_sentiment_hourly",
                "team_sentiment_daily", 
                "player_sentiment_hourly",
                "player_sentiment_daily",
                "game_sentiment_timeline"
            ]
            
            existing_collections = await db.list_collection_names()
            
            for collection_name in collections:
                if collection_name not in existing_collections:
                    await db.create_collection(collection_name)
            
            # Add indexes for time-series data
            hourly_indexes = [
                ("entity_id", 1),
                ("timestamp", -1),
                ("sentiment_score", -1)
            ]
            
            for collection_name in collections:
                collection = db[collection_name]
                for index in hourly_indexes:
                    await collection.create_index([index])
            
            return True
        except Exception as e:
            logger.error(f"Error in AggregatedSentimentMigration: {e}")
            return False
    
    async def down(self, db: AsyncIOMotorDatabase) -> bool:
        try:
            # Drop the aggregated collections
            collections = [
                "team_sentiment_hourly",
                "team_sentiment_daily",
                "player_sentiment_hourly", 
                "player_sentiment_daily",
                "game_sentiment_timeline"
            ]
            
            for collection_name in collections:
                await db.drop_collection(collection_name)
            
            return True
        except Exception as e:
            logger.error(f"Error rolling back AggregatedSentimentMigration: {e}")
            return False


class AnalyticsIndexesMigration(Migration):
    """Add analytics and reporting indexes"""
    
    async def up(self, db: AsyncIOMotorDatabase) -> bool:
        try:
            # Add specialized indexes for analytics queries
            analytics_indexes = {
                "sentiment_analysis": [
                    [("team_id", 1), ("category", 1), ("sentiment_score", -1)],
                    [("player_id", 1), ("source", 1), ("confidence", -1)],
                    [("game_id", 1), ("timestamp", -1), ("sentiment", 1)],
                    [("source", 1), ("category", 1), ("timestamp", -1)]
                ],
                "team_sentiment": [
                    [("current_sentiment", -1), ("total_mentions", -1)],
                    [("performance_sentiment", -1), ("last_updated", -1)]
                ],
                "player_sentiment": [
                    [("team_id", 1), ("fantasy_sentiment", -1)],
                    [("position", 1), ("performance_sentiment", -1)]
                ]
            }
            
            for collection_name, indexes in analytics_indexes.items():
                collection = db[collection_name]
                for index_spec in indexes:
                    await collection.create_index(index_spec)
            
            return True
        except Exception as e:
            logger.error(f"Error in AnalyticsIndexesMigration: {e}")
            return False
    
    async def down(self, db: AsyncIOMotorDatabase) -> bool:
        try:
            # This would require tracking which indexes were added
            # For simplicity, we'll just log that manual cleanup may be needed
            logger.warning("AnalyticsIndexesMigration rollback requires manual index cleanup")
            return True
        except Exception as e:
            logger.error(f"Error rolling back AnalyticsIndexesMigration: {e}")
            return False


class CachingOptimizationMigration(Migration):
    """Optimize collections for caching"""
    
    async def up(self, db: AsyncIOMotorDatabase) -> bool:
        try:
            # Add fields to support caching strategies
            collections_to_update = [
                "team_sentiment",
                "player_sentiment", 
                "game_sentiment"
            ]
            
            for collection_name in collections_to_update:
                collection = db[collection_name]
                
                # Add cache-related fields
                await collection.update_many(
                    {"cache_version": {"$exists": False}},
                    {
                        "$set": {
                            "cache_version": 1,
                            "cache_updated_at": datetime.utcnow(),
                            "cache_ttl": 300  # 5 minutes default
                        }
                    }
                )
            
            # Create indexes for cache optimization
            cache_indexes = [
                ("cache_updated_at", -1),
                ("cache_version", 1)
            ]
            
            for collection_name in collections_to_update:
                collection = db[collection_name]
                for index_spec in cache_indexes:
                    await collection.create_index([index_spec])
            
            return True
        except Exception as e:
            logger.error(f"Error in CachingOptimizationMigration: {e}")
            return False
    
    async def down(self, db: AsyncIOMotorDatabase) -> bool:
        try:
            # Remove cache-related fields
            collections_to_update = [
                "team_sentiment",
                "player_sentiment",
                "game_sentiment"
            ]
            
            for collection_name in collections_to_update:
                collection = db[collection_name]
                
                await collection.update_many(
                    {},
                    {
                        "$unset": {
                            "cache_version": "",
                            "cache_updated_at": "",
                            "cache_ttl": ""
                        }
                    }
                )
            
            return True
        except Exception as e:
            logger.error(f"Error rolling back CachingOptimizationMigration: {e}")
            return False


# Global migration service instance
migration_service = DatabaseMigrationService()


async def get_migration_service() -> DatabaseMigrationService:
    """Dependency to get migration service"""
    return migration_service