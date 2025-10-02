#!/usr/bin/env python3
"""
Database maintenance CLI script for NFL Sentiment Analyzer.
Provides commands for migrations, archiving, and cache management.
"""

import asyncio
import argparse
import sys
from datetime import datetime
from typing import Optional

# Add the app directory to the path
sys.path.append(".")

from app.core.database import db_manager
from app.services.database_migration_service import get_migration_service
from app.services.data_archiving_service import get_archiving_service
from app.services.caching_service import get_caching_service
from app.core.database_indexes import create_all_indexes


async def run_migrations(target_version: Optional[str] = None):
    """
    Run pending database migrations up to an optional target version.
    
    Connects to the database, reports current migration status, applies pending migrations (stopping at `target_version` if provided), and prints summaries of applied and failed migrations.
    
    Parameters:
        target_version (Optional[str]): If provided, migrate up only until this migration version; if omitted, apply all pending migrations.
    """
    print("🔄 Running database migrations...")

    try:
        await db_manager.connect_mongodb()
        migration_service = await get_migration_service()

        # Get current status
        status = await migration_service.get_migration_status()
        print(f"Current version: {status['current_version']}")
        print(f"Pending migrations: {status['pending_count']}")

        if status["pending_count"] == 0:
            print("✅ No pending migrations")
            return

        # Run migrations
        result = await migration_service.migrate_up(target_version)

        print(f"✅ Applied {len(result['applied'])} migrations")
        for migration in result["applied"]:
            print(f"  - {migration['version']}: {migration['description']}")

        if result["failed"]:
            print(f"❌ Failed migrations: {len(result['failed'])}")
            for migration in result["failed"]:
                print(f"  - {migration['version']}: {migration['description']}")

    except Exception as e:
        print(f"❌ Error running migrations: {e}")
        sys.exit(1)

    finally:
        await db_manager.disconnect()


async def rollback_migrations(target_version: str):
    """
    Roll back database migrations to a specified target version.
    
    Parameters:
    	target_version (str): The migration version to roll back to. The function will connect to the database, attempt to migrate down to this version, and print applied and failed rollback details.
    
    Notes:
    	This function connects to the MongoDB instance and ensures disconnection on completion. On error it prints an error message and exits the process with status code 1.
    """
    print(f"🔄 Rolling back migrations to version {target_version}...")

    try:
        await db_manager.connect_mongodb()
        migration_service = await get_migration_service()

        result = await migration_service.migrate_down(target_version)

        print(f"✅ Rolled back {len(result['rolled_back'])} migrations")
        for migration in result["rolled_back"]:
            print(f"  - {migration['version']}: {migration['description']}")

        if result["failed"]:
            print(f"❌ Failed rollbacks: {len(result['failed'])}")
            for migration in result["failed"]:
                print(f"  - {migration['version']}: {migration['description']}")

    except Exception as e:
        print(f"❌ Error rolling back migrations: {e}")
        sys.exit(1)

    finally:
        await db_manager.disconnect()


async def migration_status():
    """
    Display current migration state and recent migration activity.
    
    Connects to the migration service and prints the current migration version, counts for applied,
    pending, and total migrations. If there are pending migrations, lists each pending migration's
    version and description. If there are applied migrations, shows up to the last five with their
    version, description, and applied timestamp. On error, prints an error message and exits the
    process with status code 1.
    """
    try:
        await db_manager.connect_mongodb()
        migration_service = await get_migration_service()

        status = await migration_service.get_migration_status()

        print("📊 Migration Status")
        print(f"Current version: {status['current_version']}")
        print(f"Applied migrations: {status['applied_count']}")
        print(f"Pending migrations: {status['pending_count']}")
        print(f"Total migrations: {status['total_migrations']}")

        if status["pending_migrations"]:
            print("\n🔄 Pending migrations:")
            for migration in status["pending_migrations"]:
                print(f"  - {migration['version']}: {migration['description']}")

        if status["applied_migrations"]:
            print("\n✅ Applied migrations:")
            for migration in status["applied_migrations"][-5:]:  # Show last 5
                print(
                    f"  - {migration['version']}: {migration['description']} ({migration['applied_at']})"
                )

    except Exception as e:
        print(f"❌ Error getting migration status: {e}")
        sys.exit(1)

    finally:
        await db_manager.disconnect()


async def create_indexes():
    """
    Create all configured database indexes and print a per-collection summary.
    
    Connects to the database, executes index creation, prints the number of indexes created for each collection, and ensures the database connection is closed. On error, prints an error message and exits with status 1.
    """
    print("🔄 Creating database indexes...")

    try:
        await db_manager.connect_mongodb()
        db = db_manager.get_database()

        results = await create_all_indexes(db)

        print("✅ Indexes created successfully")
        for collection, indexes in results.items():
            print(f"  - {collection}: {len(indexes)} indexes")

    except Exception as e:
        print(f"❌ Error creating indexes: {e}")
        sys.exit(1)

    finally:
        await db_manager.disconnect()


async def archive_data():
    """
    Archive old sentiment data from the active database into the archival store.
    
    Connects to MongoDB, retrieves pre-archive statistics, runs the archiving process for old sentiment records, and prints a summary including counts archived, deleted from the active collection, and the cutoff date. On error, prints the error and exits the process with status code 1.
    """
    print("🔄 Running data archiving...")

    try:
        await db_manager.connect_mongodb()
        archiving_service = await get_archiving_service()

        # Get stats before archiving
        stats_before = await archiving_service.get_archiving_stats()
        print(f"Active documents before: {stats_before['active_documents']}")
        print(f"Eligible for archiving: {stats_before['eligible_for_archiving']}")

        # Run archiving
        result = await archiving_service.archive_old_sentiment_data()

        print(f"✅ Archived {result['archived_count']} documents")
        print(f"Deleted from active: {result['deleted_from_active']}")
        print(f"Cutoff date: {result['cutoff_date']}")

    except Exception as e:
        print(f"❌ Error running archiving: {e}")
        sys.exit(1)

    finally:
        await db_manager.disconnect()


async def archive_maintenance():
    """
    Run the full archiving maintenance workflow.
    
    Executes the archiving service's maintenance tasks (archiving, deletion, cleanup) and prints summarized result counts. Opens and closes the database connection as needed and exits the process with status 1 on error.
    """
    print("🔄 Running complete archiving maintenance...")

    try:
        await db_manager.connect_mongodb()
        archiving_service = await get_archiving_service()

        result = await archiving_service.run_maintenance()

        print("✅ Maintenance completed")

        if "archiving" in result:
            arch = result["archiving"]
            print(f"  Archived: {arch['archived_count']} documents")

        if "deletion" in result:
            del_result = result["deletion"]
            print(f"  Deleted from archive: {del_result['deleted_count']} documents")

        if "cleanup" in result:
            cleanup = result["cleanup"]
            print(f"  Permanently deleted: {cleanup['permanently_deleted']} documents")

        if "final_stats" in result:
            stats = result["final_stats"]
            print(f"  Final active documents: {stats['active_documents']}")
            print(f"  Final archived documents: {stats['archived_documents']}")

    except Exception as e:
        print(f"❌ Error running maintenance: {e}")
        sys.exit(1)

    finally:
        await db_manager.disconnect()


async def archive_stats():
    """
    Display current archiving statistics including counts, eligibility, thresholds, and date ranges.
    
    Prints total counts for active, archived, deleted, and overall documents; numbers eligible for archiving and deletion; configured archive and deletion thresholds (in days); and, when available, the oldest-to-newest date ranges for active and archived data.
    """
    try:
        await db_manager.connect_mongodb()
        archiving_service = await get_archiving_service()

        stats = await archiving_service.get_archiving_stats()

        print("📊 Archiving Statistics")
        print(f"Active documents: {stats['active_documents']:,}")
        print(f"Archived documents: {stats['archived_documents']:,}")
        print(f"Deleted documents: {stats['deleted_documents']:,}")
        print(f"Total documents: {stats['total_documents']:,}")
        print(f"Eligible for archiving: {stats['eligible_for_archiving']:,}")
        print(f"Eligible for deletion: {stats['eligible_for_deletion']:,}")
        print(f"Archive threshold: {stats['archive_threshold_days']} days")
        print(f"Deletion threshold: {stats['deletion_threshold_days']} days")

        if stats["active_date_range"]["oldest"]:
            print(
                f"Active data range: {stats['active_date_range']['oldest']} to {stats['active_date_range']['newest']}"
            )

        if stats["archive_date_range"]["oldest"]:
            print(
                f"Archive data range: {stats['archive_date_range']['oldest']} to {stats['archive_date_range']['newest']}"
            )

    except Exception as e:
        print(f"❌ Error getting archiving stats: {e}")
        sys.exit(1)

    finally:
        await db_manager.disconnect()


async def cache_stats():
    """
    Display cache statistics for the Redis-backed caching layer.
    
    Connects to Redis, retrieves statistics from the caching service, and prints a summary including connected clients, memory usage, total keys, hit rate, and per-pattern key counts when available.
    """
    try:
        await db_manager.connect_redis()
        caching_service = await get_caching_service()

        stats = await caching_service.get_cache_stats()

        print("📊 Cache Statistics")
        print(f"Connected clients: {stats.get('connected_clients', 0)}")
        print(f"Used memory: {stats.get('used_memory', '0B')}")
        print(f"Total keys: {stats.get('total_keys', 0)}")
        print(f"Hit rate: {stats.get('hit_rate', 0):.2%}")

        if "key_counts" in stats:
            print("\nKey counts by pattern:")
            for pattern, count in stats["key_counts"].items():
                print(f"  {pattern}: {count}")

    except Exception as e:
        print(f"❌ Error getting cache stats: {e}")
        sys.exit(1)

    finally:
        await db_manager.disconnect()


async def clear_cache(pattern: Optional[str] = None):
    """
    Clear cached sentiment entries either by key pattern or entirely.
    
    If `pattern` is provided, deletes cache keys matching that pattern and reports the number removed.
    If `pattern` is omitted, invalidates the entire sentiment cache.
    
    Parameters:
        pattern (Optional[str]): Key pattern to match cache entries to delete (e.g., "sentiment:*"). Omit to clear all sentiment cache.
    """
    try:
        await db_manager.connect_redis()
        caching_service = await get_caching_service()

        if pattern:
            deleted_count = await caching_service.delete_pattern(pattern)
            print(
                f"✅ Cleared {deleted_count} cache entries matching pattern: {pattern}"
            )
        else:
            await caching_service.invalidate_all_sentiment_cache()
            print("✅ Cleared all sentiment cache")

    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        sys.exit(1)

    finally:
        await db_manager.disconnect()


def main():
    """
    Entry point for the database maintenance command-line interface that parses user arguments and dispatches to the corresponding maintenance subcommands.
    
    Parses subcommands for migrations, index creation, archiving, and cache operations, then invokes the matching async handler (e.g., migrate, rollback, migration-status, create-indexes, archive, archive-maintenance, archive-stats, cache-stats, cache-clear). If no subcommand or an unknown subcommand is provided, prints help.
    """
    parser = argparse.ArgumentParser(
        description="NFL Sentiment Analyzer Database Maintenance"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Migration commands
    migrate_parser = subparsers.add_parser("migrate", help="Run database migrations")
    migrate_parser.add_argument("--target", help="Target migration version")

    rollback_parser = subparsers.add_parser("rollback", help="Rollback migrations")
    rollback_parser.add_argument("target", help="Target migration version")

    subparsers.add_parser("migration-status", help="Show migration status")
    subparsers.add_parser("create-indexes", help="Create database indexes")

    # Archiving commands
    subparsers.add_parser("archive", help="Run data archiving")
    subparsers.add_parser(
        "archive-maintenance", help="Run complete archiving maintenance"
    )
    subparsers.add_parser("archive-stats", help="Show archiving statistics")

    # Cache commands
    subparsers.add_parser("cache-stats", help="Show cache statistics")
    cache_clear_parser = subparsers.add_parser("cache-clear", help="Clear cache")
    cache_clear_parser.add_argument("--pattern", help="Cache key pattern to clear")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Run the appropriate command
    if args.command == "migrate":
        asyncio.run(run_migrations(args.target))
    elif args.command == "rollback":
        asyncio.run(rollback_migrations(args.target))
    elif args.command == "migration-status":
        asyncio.run(migration_status())
    elif args.command == "create-indexes":
        asyncio.run(create_indexes())
    elif args.command == "archive":
        asyncio.run(archive_data())
    elif args.command == "archive-maintenance":
        asyncio.run(archive_maintenance())
    elif args.command == "archive-stats":
        asyncio.run(archive_stats())
    elif args.command == "cache-stats":
        asyncio.run(cache_stats())
    elif args.command == "cache-clear":
        asyncio.run(clear_cache(args.pattern))
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()


if __name__ == "__main__":
    main()
