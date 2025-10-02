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
        """
        Initialize the migration with its semantic version identifier and human-readable description.
        
        Parameters:
            version (str): Unique version identifier for the migration (e.g., "001", "2025-01-15-001").
            description (str): Short human-readable summary of what the migration does.
        
        Notes:
            The `applied_at` attribute is initialized to `None` and will be set to a `datetime` when the migration is applied.
        """
        self.version = version
        self.description = description
        self.applied_at: Optional[datetime] = None

    async def up(self, db: AsyncIOMotorDatabase) -> bool:
        """
        Apply this migration's changes to the given database.
        
        Parameters:
            db (AsyncIOMotorDatabase): The database instance the migration should modify.
        
        Returns:
            bool: `True` if the migration was applied successfully, `False` otherwise.
        
        Raises:
            NotImplementedError: If the base class method is not overridden by a subclass.
        """
        raise NotImplementedError("Migration must implement up() method")

    async def down(self, db: AsyncIOMotorDatabase) -> bool:
        """
        Perform the rollback for this migration.
        
        Returns:
            bool: `True` if the rollback succeeded, `False` otherwise.
        
        Raises:
            NotImplementedError: Always raised by the base implementation; subclasses must override this method.
        """
        raise NotImplementedError("Migration must implement down() method")


class DatabaseMigrationService:
    """Service for managing database migrations"""

    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None):
        """
        Initialize the DatabaseMigrationService and register built-in migrations.
        
        Parameters:
            db (Optional[AsyncIOMotorDatabase]): An existing async MongoDB database instance to use; if omitted, the service will obtain the database lazily.
        """
        self.db = db
        self.migrations_collection = "schema_migrations"
        self.migrations: List[Migration] = []
        self._register_migrations()

    async def get_database(self) -> AsyncIOMotorDatabase:
        """
        Return the cached database instance, creating and caching it on first access.
        
        Returns:
            AsyncIOMotorDatabase: The connected database instance.
        """
        if not self.db:
            self.db = await get_database()
        return self.db

    def _register_migrations(self):
        """
        Populate the service with the ordered list of built-in migration instances.
        
        This assigns an ordered list of Migration objects to self.migrations representing the predefined migration sequence (versions "001" through "005").
        """
        self.migrations = [
            InitialSchemaMigration("001", "Initial schema setup with indexes"),
            EnhancedSentimentModelMigration(
                "002", "Enhanced sentiment model with NFL context"
            ),
            AggregatedSentimentMigration("003", "Add aggregated sentiment collections"),
            AnalyticsIndexesMigration("004", "Add analytics and reporting indexes"),
            CachingOptimizationMigration("005", "Optimize collections for caching"),
        ]

    async def get_applied_migrations(self) -> List[Dict[str, Any]]:
        """
        Return applied migration documents ordered by their application time.
        
        Returns:
            List[Dict[str, Any]]: Applied migration documents sorted by `applied_at` in ascending order (oldest first).
        """
        db = await self.get_database()
        collection = db[self.migrations_collection]

        cursor = collection.find({}).sort("applied_at", 1)
        return await cursor.to_list(length=None)

    async def get_pending_migrations(self) -> List[Migration]:
        """
        Return registered migrations not yet recorded as applied.
        
        Returns:
            List[Migration]: Migration instances whose `version` is not present in the applied migrations collection.
        """
        applied = await self.get_applied_migrations()
        applied_versions = {m["version"] for m in applied}

        return [m for m in self.migrations if m.version not in applied_versions]

    async def apply_migration(self, migration: Migration) -> bool:
        """
        Apply a single registered migration and record its outcome in the migrations collection.
        
        Parameters:
            migration (Migration): The migration instance to apply.
        
        Returns:
            bool: `True` if the migration was applied successfully and recorded, `False` otherwise.
        """
        db = await self.get_database()
        collection = db[self.migrations_collection]

        try:
            logger.info(
                f"Applying migration {migration.version}: {migration.description}"
            )

            # Apply the migration
            success = await migration.up(db)

            if success:
                # Record the migration
                migration_record = {
                    "version": migration.version,
                    "description": migration.description,
                    "applied_at": datetime.utcnow(),
                    "status": "applied",
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
                "error": str(e),
            }

            await collection.insert_one(migration_record)
            return False

    async def rollback_migration(self, migration: Migration) -> bool:
        """
        Rollback the provided migration and remove its applied record if successful.
        
        Returns:
            bool: `True` if the migration was rolled back and its record was removed from the migrations collection, `False` otherwise.
        """
        db = await self.get_database()
        collection = db[self.migrations_collection]

        try:
            logger.info(
                f"Rolling back migration {migration.version}: {migration.description}"
            )

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
        """
        Apply pending migrations in sequence, optionally stopping at a target version.
        
        Parameters:
            target_version (Optional[str]): If provided, only migrations with version less than or equal
                to this value will be applied.
        
        Returns:
            result (Dict[str, Any]): Summary of the migration run containing:
                - `applied`: list of applied migrations as dicts with `version` and `description`.
                - `failed`: list of migrations that failed (stopped on first failure) with `version` and `description`.
                - `total`: total number of migrations attempted.
        """
        pending = await self.get_pending_migrations()

        if target_version:
            # Filter to only migrations up to target version
            pending = [m for m in pending if m.version <= target_version]

        results = {"applied": [], "failed": [], "total": len(pending)}

        for migration in pending:
            success = await self.apply_migration(migration)

            if success:
                results["applied"].append(
                    {"version": migration.version, "description": migration.description}
                )
            else:
                results["failed"].append(
                    {"version": migration.version, "description": migration.description}
                )
                # Stop on first failure
                break

        return results

    async def migrate_down(self, target_version: str) -> Dict[str, Any]:
        """
        Rollback applied migrations with versions greater than the given target version.
        
        Migrations are processed in reverse-applied order and the operation stops on the first failure.
        
        Parameters:
            target_version (str): Version threshold; any applied migration with a version greater than this value will be rolled back.
        
        Returns:
            dict: A summary with keys:
                - "rolled_back" (List[Dict[str, str]]): Successfully rolled back migrations as objects with "version" and "description".
                - "failed" (List[Dict[str, str]]): Migrations that failed to roll back (contains the failing migration and any recorded before stopping).
                - "total" (int): Number of migrations considered for rollback.
        """
        applied = await self.get_applied_migrations()

        # Find migrations to rollback (in reverse order)
        to_rollback = [m for m in reversed(applied) if m["version"] > target_version]

        results = {"rolled_back": [], "failed": [], "total": len(to_rollback)}

        for migration_record in to_rollback:
            # Find the migration class
            migration = next(
                (
                    m
                    for m in self.migrations
                    if m.version == migration_record["version"]
                ),
                None,
            )

            if migration:
                success = await self.rollback_migration(migration)

                if success:
                    results["rolled_back"].append(
                        {
                            "version": migration.version,
                            "description": migration.description,
                        }
                    )
                else:
                    results["failed"].append(
                        {
                            "version": migration.version,
                            "description": migration.description,
                        }
                    )
                    # Stop on first failure
                    break

        return results

    async def get_migration_status(self) -> Dict[str, Any]:
        """
        Return the current migration status for the migration service.
        
        Returns:
            status (Dict[str, Any]): A dictionary with migration metadata:
                - current_version: The version string of the last applied migration, or None if none applied.
                - applied_count: Number of applied migrations.
                - pending_count: Number of pending (not yet applied) migrations.
                - total_migrations: Total number of registered migrations.
                - applied_migrations: List of applied migration documents as stored in the migrations collection (sorted by applied_at ascending).
                - pending_migrations: List of dicts for each pending migration with keys `version` and `description`.
        """
        applied = await self.get_applied_migrations()
        pending = await self.get_pending_migrations()

        return {
            "current_version": applied[-1]["version"] if applied else None,
            "applied_count": len(applied),
            "pending_count": len(pending),
            "total_migrations": len(self.migrations),
            "applied_migrations": applied,
            "pending_migrations": [
                {"version": m.version, "description": m.description} for m in pending
            ],
        }


# Migration implementations


class InitialSchemaMigration(Migration):
    """Initial schema setup with basic indexes"""

    async def up(self, db: AsyncIOMotorDatabase) -> bool:
        """
        Create initial collections and base indexes required by the application.
        
        Parameters:
            db (AsyncIOMotorDatabase): Database instance to modify.
        
        Returns:
            bool: `True` if the migration completed successfully and the required collections and indexes were ensured, `False` if an error occurred.
        """
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
                "games",
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
        """
        Rolls back the initial schema by dropping all custom indexes.
        
        Returns:
            True if indexes were dropped successfully, False otherwise.
        """
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
        """
        Add enhanced sentiment model fields to sentiment_analysis documents that are missing them.
        
        Sets `nfl_context`, `sentiment_weights`, `emotion_scores`, `aspect_sentiments`, and `keyword_contributions` to empty objects on any document where `nfl_context` does not exist.
        
        Returns:
            `true` if the migration succeeded, `false` otherwise.
        """
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
                        "keyword_contributions": {},
                    }
                },
            )

            return True
        except Exception as e:
            logger.error(f"Error in EnhancedSentimentModelMigration: {e}")
            return False

    async def down(self, db: AsyncIOMotorDatabase) -> bool:
        """
        Removes enhanced sentiment model fields from all documents in the sentiment_analysis collection.
        
        Returns:
            True if the fields were removed without error, False otherwise.
        """
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
                        "keyword_contributions": "",
                    }
                },
            )

            return True
        except Exception as e:
            logger.error(f"Error rolling back EnhancedSentimentModelMigration: {e}")
            return False


class AggregatedSentimentMigration(Migration):
    """Add aggregated sentiment collections"""

    async def up(self, db: AsyncIOMotorDatabase) -> bool:
        """
        Create aggregated sentiment collections and ensure time-series indexes exist.
        
        Creates the hourly/daily aggregated collections and adds indexes for entity_id, timestamp, and sentiment_score if they are not already present.
        
        Returns:
            bool: `True` if all collections and indexes were created or already present, `False` on error.
        """
        try:
            # Create aggregated sentiment collections
            collections = [
                "team_sentiment_hourly",
                "team_sentiment_daily",
                "player_sentiment_hourly",
                "player_sentiment_daily",
                "game_sentiment_timeline",
            ]

            existing_collections = await db.list_collection_names()

            for collection_name in collections:
                if collection_name not in existing_collections:
                    await db.create_collection(collection_name)

            # Add indexes for time-series data
            hourly_indexes = [
                ("entity_id", 1),
                ("timestamp", -1),
                ("sentiment_score", -1),
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
        """
        Remove aggregated sentiment collections created by this migration.
        
        Returns:
            True if all target collections were dropped successfully, False otherwise.
        """
        try:
            # Drop the aggregated collections
            collections = [
                "team_sentiment_hourly",
                "team_sentiment_daily",
                "player_sentiment_hourly",
                "player_sentiment_daily",
                "game_sentiment_timeline",
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
        """
        Create analytics-oriented indexes on the `sentiment_analysis`, `team_sentiment`, and `player_sentiment` collections.
        
        Parameters:
            db (AsyncIOMotorDatabase): MongoDB database instance to apply index changes to.
        
        Returns:
            bool: `True` if all indexes were created successfully, `False` otherwise.
        """
        try:
            # Add specialized indexes for analytics queries
            analytics_indexes = {
                "sentiment_analysis": [
                    [("team_id", 1), ("category", 1), ("sentiment_score", -1)],
                    [("player_id", 1), ("source", 1), ("confidence", -1)],
                    [("game_id", 1), ("timestamp", -1), ("sentiment", 1)],
                    [("source", 1), ("category", 1), ("timestamp", -1)],
                ],
                "team_sentiment": [
                    [("current_sentiment", -1), ("total_mentions", -1)],
                    [("performance_sentiment", -1), ("last_updated", -1)],
                ],
                "player_sentiment": [
                    [("team_id", 1), ("fantasy_sentiment", -1)],
                    [("position", 1), ("performance_sentiment", -1)],
                ],
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
        """
        Indicate that analytics-oriented indexes are not removed automatically and that manual index cleanup may be required.
        
        This method does not perform any automatic index removal; it logs a warning advising manual cleanup. It returns success unless an unexpected error occurs during execution.
        
        Returns:
            bool: `True` if the rollback is treated as successful (no automated index removal performed), `False` if an error occurred.
        """
        try:
            # This would require tracking which indexes were added
            # For simplicity, we'll just log that manual cleanup may be needed
            logger.warning(
                "AnalyticsIndexesMigration rollback requires manual index cleanup"
            )
            return True
        except Exception as e:
            logger.error(f"Error rolling back AnalyticsIndexesMigration: {e}")
            return False


class CachingOptimizationMigration(Migration):
    """Optimize collections for caching"""

    async def up(self, db: AsyncIOMotorDatabase) -> bool:
        """
        Adds cache-related fields and indexes to sentiment collections to enable caching.
        
        This migration updates the team_sentiment, player_sentiment, and game_sentiment collections by adding the fields `cache_version` (set to 1), `cache_updated_at` (set to current UTC timestamp), and `cache_ttl` (set to 300 seconds) where they are missing, and creates indexes on `cache_updated_at` (descending) and `cache_version` (ascending).
        
        Returns:
            True if the migration completed successfully, False otherwise.
        """
        try:
            # Add fields to support caching strategies
            collections_to_update = [
                "team_sentiment",
                "player_sentiment",
                "game_sentiment",
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
                            "cache_ttl": 300,  # 5 minutes default
                        }
                    },
                )

            # Create indexes for cache optimization
            cache_indexes = [("cache_updated_at", -1), ("cache_version", 1)]

            for collection_name in collections_to_update:
                collection = db[collection_name]
                for index_spec in cache_indexes:
                    await collection.create_index([index_spec])

            return True
        except Exception as e:
            logger.error(f"Error in CachingOptimizationMigration: {e}")
            return False

    async def down(self, db: AsyncIOMotorDatabase) -> bool:
        """
        Remove cache-related fields from team, player, and game sentiment collections.
        
        This migration rollback unsets `cache_version`, `cache_updated_at`, and `cache_ttl` from the following collections: `team_sentiment`, `player_sentiment`, and `game_sentiment`.
        
        Returns:
            True if all targeted documents were updated successfully, False otherwise.
        """
        try:
            # Remove cache-related fields
            collections_to_update = [
                "team_sentiment",
                "player_sentiment",
                "game_sentiment",
            ]

            for collection_name in collections_to_update:
                collection = db[collection_name]

                await collection.update_many(
                    {},
                    {
                        "$unset": {
                            "cache_version": "",
                            "cache_updated_at": "",
                            "cache_ttl": "",
                        }
                    },
                )

            return True
        except Exception as e:
            logger.error(f"Error rolling back CachingOptimizationMigration: {e}")
            return False


# Global migration service instance
migration_service = DatabaseMigrationService()


async def get_migration_service() -> DatabaseMigrationService:
    """
    Provide the module-level DatabaseMigrationService singleton.
    
    Returns:
        DatabaseMigrationService: The singleton service instance used to manage and run database schema migrations.
    """
    return migration_service
