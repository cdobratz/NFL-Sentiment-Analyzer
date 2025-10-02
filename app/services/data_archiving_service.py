"""
Data archiving service for historical sentiment data management.
Implements intelligent archiving strategies to maintain performance while preserving historical data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging
from app.core.database import get_database
from app.models.sentiment import SentimentAnalysisInDB

logger = logging.getLogger(__name__)


class DataArchivingService:
    """Service for managing historical sentiment data archiving"""

    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None):
        self.db = db
        # Archiving configuration
        self.archive_after_days = 90  # Archive data older than 90 days
        self.delete_after_days = 365  # Delete archived data after 1 year
        self.batch_size = 1000  # Process in batches

        # Collection names
        self.active_collection = "sentiment_analysis"
        self.archive_collection = "sentiment_analysis_archive"
        self.deleted_collection = "sentiment_analysis_deleted"

    async def get_database(self) -> AsyncIOMotorDatabase:
        """Get database instance"""
        if not self.db:
            self.db = await get_database()
        return self.db

    async def archive_old_sentiment_data(self) -> Dict[str, int]:
        """Archive sentiment data older than configured threshold"""
        db = await self.get_database()

        # Calculate cutoff date
        cutoff_date = datetime.utcnow() - timedelta(days=self.archive_after_days)

        # Find documents to archive
        query = {"timestamp": {"$lt": cutoff_date}}

        active_collection = db[self.active_collection]
        archive_collection = db[self.archive_collection]

        archived_count = 0
        deleted_count = 0

        try:
            # Process in batches
            cursor = active_collection.find(query).batch_size(self.batch_size)

            batch = []
            async for document in cursor:
                batch.append(document)

                if len(batch) >= self.batch_size:
                    # Insert batch into archive
                    if batch:
                        await archive_collection.insert_many(batch)
                        archived_count += len(batch)

                        # Delete from active collection
                        ids_to_delete = [doc["_id"] for doc in batch]
                        result = await active_collection.delete_many(
                            {"_id": {"$in": ids_to_delete}}
                        )
                        deleted_count += result.deleted_count

                        batch = []
                        logger.info(f"Archived batch of {len(batch)} documents")

            # Process remaining documents
            if batch:
                await archive_collection.insert_many(batch)
                archived_count += len(batch)

                ids_to_delete = [doc["_id"] for doc in batch]
                result = await active_collection.delete_many(
                    {"_id": {"$in": ids_to_delete}}
                )
                deleted_count += result.deleted_count

            logger.info(
                f"Archiving completed: {archived_count} documents archived, {deleted_count} deleted from active"
            )

        except Exception as e:
            logger.error(f"Error during archiving: {e}")
            raise

        return {
            "archived_count": archived_count,
            "deleted_from_active": deleted_count,
            "cutoff_date": cutoff_date.isoformat(),
        }

    async def delete_old_archived_data(self) -> Dict[str, int]:
        """Delete archived data older than configured threshold"""
        db = await self.get_database()

        # Calculate cutoff date for deletion
        cutoff_date = datetime.utcnow() - timedelta(days=self.delete_after_days)

        archive_collection = db[self.archive_collection]
        deleted_collection = db[self.deleted_collection]

        # Find documents to delete
        query = {"timestamp": {"$lt": cutoff_date}}

        deleted_count = 0
        moved_to_deleted = 0

        try:
            # First, move to deleted collection for audit trail
            cursor = archive_collection.find(query).batch_size(self.batch_size)

            batch = []
            async for document in cursor:
                # Add deletion metadata
                document["deleted_at"] = datetime.utcnow()
                document["deletion_reason"] = "automatic_cleanup"
                batch.append(document)

                if len(batch) >= self.batch_size:
                    # Insert into deleted collection
                    if batch:
                        await deleted_collection.insert_many(batch)
                        moved_to_deleted += len(batch)

                        # Delete from archive
                        ids_to_delete = [doc["_id"] for doc in batch]
                        result = await archive_collection.delete_many(
                            {"_id": {"$in": ids_to_delete}}
                        )
                        deleted_count += result.deleted_count

                        batch = []
                        logger.info(f"Deleted batch of {len(batch)} archived documents")

            # Process remaining documents
            if batch:
                await deleted_collection.insert_many(batch)
                moved_to_deleted += len(batch)

                ids_to_delete = [doc["_id"] for doc in batch]
                result = await archive_collection.delete_many(
                    {"_id": {"$in": ids_to_delete}}
                )
                deleted_count += result.deleted_count

            logger.info(
                f"Deletion completed: {deleted_count} documents deleted from archive"
            )

        except Exception as e:
            logger.error(f"Error during deletion: {e}")
            raise

        return {
            "deleted_count": deleted_count,
            "moved_to_deleted": moved_to_deleted,
            "cutoff_date": cutoff_date.isoformat(),
        }

    async def get_archiving_stats(self) -> Dict[str, Any]:
        """Get statistics about data archiving"""
        db = await self.get_database()

        active_collection = db[self.active_collection]
        archive_collection = db[self.archive_collection]
        deleted_collection = db[self.deleted_collection]

        # Get collection sizes
        active_count = await active_collection.count_documents({})
        archive_count = await archive_collection.count_documents({})
        deleted_count = await deleted_collection.count_documents({})

        # Get date ranges
        active_date_range = await self._get_date_range(active_collection)
        archive_date_range = await self._get_date_range(archive_collection)

        # Calculate next archiving date
        cutoff_date = datetime.utcnow() - timedelta(days=self.archive_after_days)
        eligible_for_archiving = await active_collection.count_documents(
            {"timestamp": {"$lt": cutoff_date}}
        )

        # Calculate next deletion date
        deletion_cutoff = datetime.utcnow() - timedelta(days=self.delete_after_days)
        eligible_for_deletion = await archive_collection.count_documents(
            {"timestamp": {"$lt": deletion_cutoff}}
        )

        return {
            "active_documents": active_count,
            "archived_documents": archive_count,
            "deleted_documents": deleted_count,
            "total_documents": active_count + archive_count + deleted_count,
            "active_date_range": active_date_range,
            "archive_date_range": archive_date_range,
            "eligible_for_archiving": eligible_for_archiving,
            "eligible_for_deletion": eligible_for_deletion,
            "archive_threshold_days": self.archive_after_days,
            "deletion_threshold_days": self.delete_after_days,
            "next_archive_cutoff": cutoff_date.isoformat(),
            "next_deletion_cutoff": deletion_cutoff.isoformat(),
        }

    async def _get_date_range(self, collection) -> Dict[str, Optional[str]]:
        """Get the date range for a collection"""
        try:
            # Get oldest document
            oldest = await collection.find_one({}, sort=[("timestamp", 1)])

            # Get newest document
            newest = await collection.find_one({}, sort=[("timestamp", -1)])

            return {
                "oldest": oldest["timestamp"].isoformat() if oldest else None,
                "newest": newest["timestamp"].isoformat() if newest else None,
            }
        except Exception as e:
            logger.error(f"Error getting date range: {e}")
            return {"oldest": None, "newest": None}

    async def restore_from_archive(
        self, start_date: datetime, end_date: datetime
    ) -> Dict[str, int]:
        """Restore archived data back to active collection for a date range"""
        db = await self.get_database()

        active_collection = db[self.active_collection]
        archive_collection = db[self.archive_collection]

        # Find documents to restore
        query = {"timestamp": {"$gte": start_date, "$lte": end_date}}

        restored_count = 0

        try:
            cursor = archive_collection.find(query).batch_size(self.batch_size)

            batch = []
            async for document in cursor:
                batch.append(document)

                if len(batch) >= self.batch_size:
                    # Insert back into active collection
                    if batch:
                        await active_collection.insert_many(batch)
                        restored_count += len(batch)

                        # Remove from archive
                        ids_to_remove = [doc["_id"] for doc in batch]
                        await archive_collection.delete_many(
                            {"_id": {"$in": ids_to_remove}}
                        )

                        batch = []
                        logger.info(f"Restored batch of {len(batch)} documents")

            # Process remaining documents
            if batch:
                await active_collection.insert_many(batch)
                restored_count += len(batch)

                ids_to_remove = [doc["_id"] for doc in batch]
                await archive_collection.delete_many({"_id": {"$in": ids_to_remove}})

            logger.info(f"Restoration completed: {restored_count} documents restored")

        except Exception as e:
            logger.error(f"Error during restoration: {e}")
            raise

        return {
            "restored_count": restored_count,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
        }

    async def cleanup_deleted_data(self, older_than_days: int = 30) -> int:
        """Permanently delete data from deleted collection"""
        db = await self.get_database()
        deleted_collection = db[self.deleted_collection]

        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

        result = await deleted_collection.delete_many(
            {"deleted_at": {"$lt": cutoff_date}}
        )

        logger.info(
            f"Permanently deleted {result.deleted_count} documents from deleted collection"
        )

        return result.deleted_count

    async def run_maintenance(self) -> Dict[str, Any]:
        """Run complete maintenance cycle"""
        logger.info("Starting data archiving maintenance cycle")

        results = {}

        try:
            # Archive old data
            archive_result = await self.archive_old_sentiment_data()
            results["archiving"] = archive_result

            # Delete old archived data
            deletion_result = await self.delete_old_archived_data()
            results["deletion"] = deletion_result

            # Cleanup permanently deleted data
            cleanup_count = await self.cleanup_deleted_data()
            results["cleanup"] = {"permanently_deleted": cleanup_count}

            # Get final stats
            stats = await self.get_archiving_stats()
            results["final_stats"] = stats

            logger.info("Data archiving maintenance cycle completed successfully")

        except Exception as e:
            logger.error(f"Error during maintenance cycle: {e}")
            results["error"] = str(e)
            raise

        return results


# Global archiving service instance
archiving_service = DataArchivingService()


async def get_archiving_service() -> DataArchivingService:
    """Dependency to get archiving service"""
    return archiving_service
