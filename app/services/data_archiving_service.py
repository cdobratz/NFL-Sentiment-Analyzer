"""
Data archiving service for historical sentiment data management.
Implements intelligent archiving strategies to maintain performance while preserving historical data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging
from pymongo.errors import BulkWriteError
from app.core.database import get_database
from app.models.sentiment import SentimentAnalysisInDB

logger = logging.getLogger(__name__)


class DataArchivingService:
    """Service for managing historical sentiment data archiving"""

    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None):
        """
        Initialize the DataArchivingService with an optional database handle and default archiving configuration.

        Parameters:
            db (Optional[AsyncIOMotorDatabase]): Optional database instance to use for operations. If not provided, the service will acquire a database connection when needed.

        Detailed behavior:
            Sets default thresholds and batch settings:
              - archive_after_days = 90 (archive active documents older than 90 days)
              - delete_after_days = 365 (delete archived documents older than 365 days)
              - batch_size = 1000 (process documents in batches)

            Establishes collection names used by the service:
              - active_collection = "sentiment_analysis"
              - archive_collection = "sentiment_analysis_archive"
              - deleted_collection = "sentiment_analysis_deleted"
        """
        self.db = db
        # Archiving configuration
        self.archive_after_days = 90  # Archive data older than 90 days
        self.delete_after_days = 365  # Delete archived data after 1 year
        self.batch_size = 1000  # Process in batches

        # Collection names
        self.active_collection = "sentiment_analyses"
        self.archive_collection = "sentiment_analyses_archive"
        self.deleted_collection = "sentiment_analyses_deleted"

    async def get_database(self) -> AsyncIOMotorDatabase:
        """
        Return the service's MongoDB database instance and cache it on the service for reuse.

        If the service has no cached database handle, obtain one and store it on self.db before returning.

        Returns:
            db (AsyncIOMotorDatabase): The MongoDB database instance used by the service.
        """
        if not self.db:
            db_result = await get_database()
            if db_result is None:
                raise RuntimeError("Database connection not established")
            self.db = db_result
        return self.db

    async def archive_old_sentiment_data(self) -> Dict[str, int]:
        """
        Archive sentiment documents from the active collection that are older than the configured threshold.

        Returns:
            result (dict): A dictionary with the following keys:
                - archived_count (int): Number of documents successfully inserted into the archive collection.
                - deleted_from_active (int): Number of documents removed from the active collection after archiving.
                - cutoff_date (str): ISO 8601 string of the cutoff timestamp used to select documents for archiving.
        """
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
                        try:
                            result = await archive_collection.insert_many(
                                batch, ordered=False
                            )
                            archived_count += len(result.inserted_ids)
                        except BulkWriteError as e:
                            # Handle duplicate key errors for idempotency
                            n_inserted = e.details.get("nInserted", 0)
                            archived_count += n_inserted
                            logger.info(
                                f"Inserted {n_inserted} documents, skipped {len(batch) - n_inserted} duplicates in batch"
                            )

                        # Delete from active collection
                        ids_to_delete = [doc["_id"] for doc in batch]
                        result = await active_collection.delete_many(
                            {"_id": {"$in": ids_to_delete}}
                        )
                        deleted_count += result.deleted_count

                        batch_count = len(batch)
                        batch = []
                        logger.info(f"Archived batch of {batch_count} documents")

            # Process remaining documents
            if batch:
                try:
                    result = await archive_collection.insert_many(batch, ordered=False)
                    archived_count += len(result.inserted_ids)
                except BulkWriteError as e:
                    # Handle duplicate key errors for idempotency
                    n_inserted = e.details.get("nInserted", 0)
                    archived_count += n_inserted
                    logger.info(
                        f"Inserted {n_inserted} documents, skipped {len(batch) - n_inserted} duplicates in final batch"
                    )

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
        """
        Remove archived documents older than the configured deletion threshold and record them in the deleted collection.

        This moves archived documents with a timestamp earlier than the retention cutoff into the deleted collection with deletion metadata, then removes them from the archive collection. Processing is performed in batches to limit resource usage.

        Returns:
            result (Dict[str, int | str]): Summary of the deletion run containing:
                - deleted_count (int): Number of documents removed from the archive collection.
                - moved_to_deleted (int): Number of documents inserted into the deleted collection.
                - cutoff_date (str): ISO-formatted cutoff timestamp used to identify deletable documents.
        """
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
                        try:
                            result = await deleted_collection.insert_many(
                                batch, ordered=False
                            )
                            moved_to_deleted += len(result.inserted_ids)
                        except BulkWriteError as e:
                            # Handle duplicate key errors for idempotency
                            n_inserted = e.details.get("nInserted", 0)
                            moved_to_deleted += n_inserted
                            logger.info(
                                f"Inserted {n_inserted} documents, skipped {len(batch) - n_inserted} duplicates in deleted batch"
                            )

                        # Delete from archive
                        ids_to_delete = [doc["_id"] for doc in batch]
                        result = await archive_collection.delete_many(
                            {"_id": {"$in": ids_to_delete}}
                        )
                        deleted_count += result.deleted_count

                        batch_count = len(batch)
                        batch = []
                        logger.info(
                            f"Deleted batch of {batch_count} archived documents"
                        )

            # Process remaining documents
            if batch:
                try:
                    result = await deleted_collection.insert_many(batch, ordered=False)
                    moved_to_deleted += len(result.inserted_ids)
                except BulkWriteError as e:
                    # Handle duplicate key errors for idempotency
                    n_inserted = e.details.get("nInserted", 0)
                    moved_to_deleted += n_inserted
                    logger.info(
                        f"Inserted {n_inserted} documents, skipped {len(batch) - n_inserted} duplicates in final deleted batch"
                    )

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
        """
        Collects statistics and threshold information for archiving and deletion across the active, archive, and deleted sentiment collections.

        Returns:
            stats (Dict[str, Any]): A dictionary containing:
                - active_documents (int): Number of documents in the active collection.
                - archived_documents (int): Number of documents in the archive collection.
                - deleted_documents (int): Number of documents in the deleted collection.
                - total_documents (int): Sum of active, archived, and deleted document counts.
                - active_date_range (Dict[str, Optional[str]]): Oldest and newest `timestamp` in the active collection as ISO strings or None.
                - archive_date_range (Dict[str, Optional[str]]): Oldest and newest `timestamp` in the archive collection as ISO strings or None.
                - eligible_for_archiving (int): Count of active documents older than the archive threshold.
                - eligible_for_deletion (int): Count of archived documents older than the deletion threshold.
                - archive_threshold_days (int): Configured number of days after which active documents are eligible for archiving.
                - deletion_threshold_days (int): Configured number of days after which archived documents are eligible for deletion.
                - next_archive_cutoff (str): ISO-formatted cutoff timestamp used to determine archival eligibility.
                - next_deletion_cutoff (str): ISO-formatted cutoff timestamp used to determine deletion eligibility.
        """
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
        """
        Return the oldest and newest `timestamp` values present in the given collection as ISO-formatted strings.

        Parameters:
            collection (Collection): MongoDB collection-like object to inspect; must contain documents with a `timestamp` field.

        Returns:
            Dict[str, Optional[str]]: A mapping with keys `"oldest"` and `"newest"` whose values are the ISO-formatted timestamp strings for the oldest and newest documents respectively, or `None` when the collection has no documents or the value is unavailable.
        """
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
        """
        Restore archived documents whose `timestamp` falls between `start_date` and `end_date` (inclusive) back into the active collection and remove them from the archive.

        Parameters:
            start_date (datetime): Lower bound of the inclusive timestamp range for restoration.
            end_date (datetime): Upper bound of the inclusive timestamp range for restoration.

        Returns:
            dict: {
                "restored_count": int,                # number of documents restored
                "start_date": str,                    # ISO 8601 string of `start_date`
                "end_date": str                       # ISO 8601 string of `end_date`
            }

        Raises:
            Exception: Propagates any exception raised during database operations.
        """
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
                        try:
                            result = await active_collection.insert_many(
                                batch, ordered=False
                            )
                            restored_count += len(result.inserted_ids)
                        except BulkWriteError as e:
                            # Handle duplicate key errors for idempotency
                            n_inserted = e.details.get("nInserted", 0)
                            restored_count += n_inserted
                            logger.info(
                                f"Restored {n_inserted} documents, skipped {len(batch) - n_inserted} duplicates in batch"
                            )

                        # Remove from archive
                        ids_to_remove = [doc["_id"] for doc in batch]
                        await archive_collection.delete_many(
                            {"_id": {"$in": ids_to_remove}}
                        )

                        batch_count = len(batch)
                        batch = []
                        logger.info(f"Restored batch of {batch_count} documents")

            # Process remaining documents
            if batch:
                try:
                    result = await active_collection.insert_many(batch, ordered=False)
                    restored_count += len(result.inserted_ids)
                except BulkWriteError as e:
                    # Handle duplicate key errors for idempotency
                    n_inserted = e.details.get("nInserted", 0)
                    restored_count += n_inserted
                    logger.info(
                        f"Restored {n_inserted} documents, skipped {len(batch) - n_inserted} duplicates in final batch"
                    )

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
        """
        Permanently remove documents from the deleted collection that are older than the given number of days.

        Parameters:
            older_than_days (int): Age threshold in days; documents with `deleted_at` earlier than (now - older_than_days) will be removed.

        Returns:
            int: The number of documents permanently deleted.
        """
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
        """
        Run a full data archiving and cleanup maintenance cycle.

        Performs the following operations in sequence: archive old active sentiment documents, delete old archived documents (moving them to a deleted collection), permanently clean up aged deleted records, and then gather final archiving statistics. Records each step's results in the returned mapping and propagates exceptions after recording an error entry.

        Returns:
            results (dict): Mapping of step names to their outcomes. Expected keys:
                - "archiving": dict with keys like `archived_count`, `deleted_from_active`, and `cutoff_date`.
                - "deletion": dict with keys like `deleted_count`, `moved_to_deleted`, and `cutoff_date`.
                - "cleanup": dict containing `permanently_deleted` (int).
                - "final_stats": dict of aggregated archiving statistics.
                - "error" (optional): string error message present if the cycle failed.
        """
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
    """
    Retrieve the module-level archiving service singleton.

    Returns:
        DataArchivingService: The shared DataArchivingService instance used by the application.
    """
    return archiving_service
