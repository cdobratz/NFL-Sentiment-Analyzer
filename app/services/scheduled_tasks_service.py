"""
Scheduled tasks service for automated maintenance operations.
Handles periodic archiving, cache cleanup, and system maintenance.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.services.data_archiving_service import get_archiving_service
from app.services.caching_service import get_caching_service
from app.services.analytics_service import get_analytics_service

logger = logging.getLogger(__name__)


class ScheduledTasksService:
    """Service for managing scheduled maintenance tasks"""

    def __init__(self):
        """
        Initialize the ScheduledTasksService instance.
        
        Sets up an AsyncIOScheduler for scheduling tasks, an `is_running` flag initialized to False, and a `task_history` dictionary for storing per-task execution metadata (last run time, status, duration, result or error).
        """
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        self.task_history: Dict[str, Dict[str, Any]] = {}

    async def start(self):
        """
        Start the scheduled tasks service and register its maintenance jobs.
        
        If the service is already running, the call returns immediately without making changes.
        This schedules configured tasks, starts the scheduler, and marks the service as running.
        """
        if self.is_running:
            logger.warning("Scheduled tasks service is already running")
            return

        logger.info("Starting scheduled tasks service...")

        # Schedule tasks
        await self._schedule_tasks()

        # Start the scheduler
        self.scheduler.start()
        self.is_running = True

        logger.info("Scheduled tasks service started successfully")

    async def stop(self):
        """Stop the scheduled tasks service"""
        if not self.is_running:
            return

        logger.info("Stopping scheduled tasks service...")

        self.scheduler.shutdown(wait=True)
        self.is_running = False

        logger.info("Scheduled tasks service stopped")

    async def _schedule_tasks(self):
        """
        Register all maintenance jobs with the scheduler.
        
        Adds scheduled jobs for data archiving, weekly maintenance, periodic cache cleanup, analytics cache refresh, and system health checks, each configured with single-instance execution and coalescing.
        """

        # Daily data archiving at 2 AM
        self.scheduler.add_job(
            self._run_data_archiving,
            CronTrigger(hour=2, minute=0),
            id="daily_archiving",
            name="Daily Data Archiving",
            max_instances=1,
            coalesce=True,
        )

        # Weekly full maintenance on Sundays at 3 AM
        self.scheduler.add_job(
            self._run_full_maintenance,
            CronTrigger(day_of_week=6, hour=3, minute=0),  # Sunday = 6
            id="weekly_maintenance",
            name="Weekly Full Maintenance",
            max_instances=1,
            coalesce=True,
        )

        # Cache cleanup every 4 hours
        self.scheduler.add_job(
            self._run_cache_cleanup,
            IntervalTrigger(hours=4),
            id="cache_cleanup",
            name="Cache Cleanup",
            max_instances=1,
            coalesce=True,
        )

        # Analytics cache refresh every hour
        self.scheduler.add_job(
            self._refresh_analytics_cache,
            IntervalTrigger(hours=1),
            id="analytics_refresh",
            name="Analytics Cache Refresh",
            max_instances=1,
            coalesce=True,
        )

        # System health check every 30 minutes
        self.scheduler.add_job(
            self._system_health_check,
            IntervalTrigger(minutes=30),
            id="health_check",
            name="System Health Check",
            max_instances=1,
            coalesce=True,
        )

        logger.info("Scheduled tasks configured:")
        for job in self.scheduler.get_jobs():
            logger.info(f"  - {job.name} ({job.id}): {job.trigger}")

    async def _run_data_archiving(self):
        """
        Archive old sentiment data using the archiving service and record execution metadata.
        
        Invokes the archiving service to archive old sentiment data and stores a per-run entry in self.task_history containing:
        - "last_run": ISO-formatted start time
        - "status": "success" or "failed"
        - "duration_seconds": run duration in seconds
        - "result": service result on success
        - "error": error message on failure
        """
        task_name = "data_archiving"
        start_time = datetime.utcnow()

        logger.info("Starting scheduled data archiving...")

        try:
            archiving_service = await get_archiving_service()
            result = await archiving_service.archive_old_sentiment_data()

            # Record success
            self.task_history[task_name] = {
                "last_run": start_time.isoformat(),
                "status": "success",
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "result": result,
            }

            logger.info(
                f"Data archiving completed: {result['archived_count']} documents archived"
            )

        except Exception as e:
            # Record failure
            self.task_history[task_name] = {
                "last_run": start_time.isoformat(),
                "status": "failed",
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "error": str(e),
            }

            logger.error(f"Data archiving failed: {e}")

    async def _run_full_maintenance(self):
        """Run weekly full maintenance"""
        task_name = "full_maintenance"
        start_time = datetime.utcnow()

        logger.info("Starting scheduled full maintenance...")

        try:
            archiving_service = await get_archiving_service()
            result = await archiving_service.run_maintenance()

            # Record success
            self.task_history[task_name] = {
                "last_run": start_time.isoformat(),
                "status": "success",
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "result": result,
            }

            logger.info("Full maintenance completed successfully")

        except Exception as e:
            # Record failure
            self.task_history[task_name] = {
                "last_run": start_time.isoformat(),
                "status": "failed",
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "error": str(e),
            }

            logger.error(f"Full maintenance failed: {e}")

    async def _run_cache_cleanup(self):
        """
        Perform cache maintenance by removing expired analytics and old sentiment trend entries, capturing before/after cache statistics, and recording the outcome.
        
        This updates the service's task_history under "cache_cleanup" with last_run, status ("success" or "failed"), duration_seconds, and either a result object containing counts for deleted keys and key totals before/after or an error string. Logs a summary on success or an error on failure.
        """
        task_name = "cache_cleanup"
        start_time = datetime.utcnow()

        logger.info("Starting scheduled cache cleanup...")

        try:
            caching_service = await get_caching_service()

            # Get stats before cleanup
            stats_before = await caching_service.get_cache_stats()

            # Clean up expired analytics cache
            deleted_analytics = await caching_service.delete_pattern("analytics:*")

            # Clean up old trend cache
            deleted_trends = await caching_service.delete_pattern("sentiment_trends:*")

            # Get stats after cleanup
            stats_after = await caching_service.get_cache_stats()

            result = {
                "deleted_analytics": deleted_analytics,
                "deleted_trends": deleted_trends,
                "keys_before": stats_before.get("total_keys", 0),
                "keys_after": stats_after.get("total_keys", 0),
            }

            # Record success
            self.task_history[task_name] = {
                "last_run": start_time.isoformat(),
                "status": "success",
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "result": result,
            }

            logger.info(
                f"Cache cleanup completed: {deleted_analytics + deleted_trends} keys deleted"
            )

        except Exception as e:
            # Record failure
            self.task_history[task_name] = {
                "last_run": start_time.isoformat(),
                "status": "failed",
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "error": str(e),
            }

            logger.error(f"Cache cleanup failed: {e}")

    async def _refresh_analytics_cache(self):
        """
        Refresh the analytics cache by updating popular sentiment leaderboards.
        
        Attempts to refresh leaderboards for "most_positive", "most_negative", and "most_mentioned" and counts how many refreshed successfully. Updates the service's task_history entry "analytics_refresh" with `last_run`, `status` ("success" or "failed"), `duration_seconds`, and either a `result` object containing `refreshed_leaderboards` or an `error` string. Per-leaderboard failures are logged and do not stop other leaderboard refreshes.
        """
        task_name = "analytics_refresh"
        start_time = datetime.utcnow()

        logger.info("Starting analytics cache refresh...")

        try:
            analytics_service = await get_analytics_service()

            # Refresh popular leaderboards
            leaderboard_types = ["most_positive", "most_negative", "most_mentioned"]
            refreshed_count = 0

            for lb_type in leaderboard_types:
                try:
                    await analytics_service.get_sentiment_leaderboards(
                        leaderboard_type=lb_type,
                        entity_type="team",
                        limit=10,
                        time_period="24h",
                    )
                    refreshed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to refresh {lb_type} leaderboard: {e}")

            # Record success
            self.task_history[task_name] = {
                "last_run": start_time.isoformat(),
                "status": "success",
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "result": {"refreshed_leaderboards": refreshed_count},
            }

            logger.info(
                f"Analytics cache refresh completed: {refreshed_count} leaderboards refreshed"
            )

        except Exception as e:
            # Record failure
            self.task_history[task_name] = {
                "last_run": start_time.isoformat(),
                "status": "failed",
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "error": str(e),
            }

            logger.error(f"Analytics cache refresh failed: {e}")

    async def _system_health_check(self):
        """
        Perform a system health check and record the outcome in the service's task history.
        
        Queries the caching and archiving services to determine `cache_healthy` and `archive_healthy`, derives `overall_healthy`, and writes an entry into `self.task_history["health_check"]` containing `last_run` (ISO timestamp), `status` (`"success"` or `"failed"`), `duration_seconds`, and either `result` (the health booleans) or `error` (the exception message). Logs a warning when the overall health is not healthy and logs an error on exceptions.
        """
        task_name = "health_check"
        start_time = datetime.utcnow()

        try:
            # Check cache service
            caching_service = await get_caching_service()
            cache_stats = await caching_service.get_cache_stats()
            cache_healthy = "connected_clients" in cache_stats

            # Check archiving service
            archiving_service = await get_archiving_service()
            archive_stats = await archiving_service.get_archiving_stats()
            archive_healthy = "active_documents" in archive_stats

            health_status = {
                "cache_healthy": cache_healthy,
                "archive_healthy": archive_healthy,
                "overall_healthy": cache_healthy and archive_healthy,
            }

            # Record result
            self.task_history[task_name] = {
                "last_run": start_time.isoformat(),
                "status": "success",
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "result": health_status,
            }

            if not health_status["overall_healthy"]:
                logger.warning(f"System health check detected issues: {health_status}")

        except Exception as e:
            # Record failure
            self.task_history[task_name] = {
                "last_run": start_time.isoformat(),
                "status": "failed",
                "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
                "error": str(e),
            }

            logger.error(f"System health check failed: {e}")

    def get_task_status(self) -> Dict[str, Any]:
        """
        Provides a status summary of the scheduler and its scheduled jobs.
        
        Returns:
            status (Dict[str, Any]): Summary object containing:
                - scheduler_running (bool): Whether the scheduler is currently running.
                - total_jobs (int): Number of scheduled jobs.
                - jobs (List[Dict[str, Any]]): List of job summaries. Each job summary contains:
                    - id (str): Job identifier.
                    - name (str): Job name.
                    - next_run (str | None): ISO 8601 timestamp of the next run, or `None` if not scheduled.
                    - trigger (str): String representation of the job trigger.
                    - last_execution (Dict[str, Any], optional): Latest execution metadata from task_history, if available.
        """
        jobs_info = []

        for job in self.scheduler.get_jobs():
            job_info = {
                "id": job.id,
                "name": job.name,
                "next_run": (
                    job.next_run_time.isoformat() if job.next_run_time else None
                ),
                "trigger": str(job.trigger),
            }

            # Add history if available
            if job.id in self.task_history:
                job_info["last_execution"] = self.task_history[job.id]

            jobs_info.append(job_info)

        return {
            "scheduler_running": self.is_running,
            "total_jobs": len(jobs_info),
            "jobs": jobs_info,
        }

    async def run_task_manually(self, task_id: str) -> Dict[str, Any]:
        """
        Run a mapped maintenance task immediately by its task identifier.
        
        Parameters:
            task_id (str): Identifier of the task to run. Valid values:
                "daily_archiving", "weekly_maintenance", "cache_cleanup",
                "analytics_refresh", "health_check".
        
        Returns:
            result (Dict[str, Any]): A status dictionary with:
                - "status": "success" if the task completed, "failed" otherwise.
                - "message": human-readable summary or error message.
        
        Raises:
            ValueError: If `task_id` is not one of the valid task identifiers.
        """
        task_functions = {
            "daily_archiving": self._run_data_archiving,
            "weekly_maintenance": self._run_full_maintenance,
            "cache_cleanup": self._run_cache_cleanup,
            "analytics_refresh": self._refresh_analytics_cache,
            "health_check": self._system_health_check,
        }

        if task_id not in task_functions:
            raise ValueError(f"Unknown task ID: {task_id}")

        logger.info(f"Manually running task: {task_id}")

        try:
            await task_functions[task_id]()
            return {
                "status": "success",
                "message": f"Task {task_id} completed successfully",
            }
        except Exception as e:
            logger.error(f"Manual task execution failed: {e}")
            return {"status": "failed", "message": str(e)}


# Global scheduled tasks service instance
scheduled_tasks_service = ScheduledTasksService()


async def get_scheduled_tasks_service() -> ScheduledTasksService:
    """
    Get the global scheduled tasks service instance used by the application.
    
    Returns:
        The global ScheduledTasksService instance.
    """
    return scheduled_tasks_service
