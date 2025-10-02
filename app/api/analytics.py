"""
Analytics API endpoints for advanced sentiment analysis reporting.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
import io
from app.services.analytics_service import get_analytics_service, AnalyticsService
from app.services.caching_service import get_caching_service, CachingService
from app.services.data_archiving_service import (
    get_archiving_service,
    DataArchivingService,
)
from app.services.database_migration_service import (
    get_migration_service,
    DatabaseMigrationService,
)
from app.models.sentiment import DataSource, SentimentCategory
from app.core.dependencies import get_current_user
from app.core.exceptions import APIError

# User type is dict from get_current_user dependency

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/metrics")
async def get_sentiment_metrics(
    entity_type: str = Query(..., description="Entity type: team, player, or game"),
    entity_id: Optional[str] = Query(None, description="Specific entity ID"),
    start_date: Optional[datetime] = Query(None, description="Start date for analysis"),
    end_date: Optional[datetime] = Query(None, description="End date for analysis"),
    sources: Optional[List[DataSource]] = Query(
        None, description="Data sources to include"
    ),
    categories: Optional[List[SentimentCategory]] = Query(
        None, description="Sentiment categories to include"
    ),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    """
    Retrieve aggregated sentiment metrics for the specified entity and optional filters.
    
    Parameters:
        entity_type (str): Entity type — "team", "player", or "game".
        entity_id (Optional[str]): Specific entity ID to filter metrics.
        start_date (Optional[datetime]): Start datetime for the analysis range (inclusive).
        end_date (Optional[datetime]): End datetime for the analysis range (inclusive).
        sources (Optional[List[DataSource]]): Data sources to include.
        categories (Optional[List[SentimentCategory]]): Sentiment categories to include.
    
    Returns:
        dict: Aggregated sentiment metrics keyed by metric name, with associated values and metadata.
    """
    try:
        metrics = await analytics_service.get_aggregated_sentiment_metrics(
            entity_type=entity_type,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date,
            sources=sources,
            categories=categories,
        )
        return metrics
    except HTTPException:
        # Re-raise FastAPI HTTPExceptions as-is
        raise
    except APIError as e:
        # Convert service APIErrors to HTTPExceptions with proper status codes
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        # Convert unexpected exceptions to 500 with generic message
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while retrieving metrics"
        )


@router.get("/trends/{entity_type}/{entity_id}")
async def get_sentiment_trends(
    entity_type: str,
    entity_id: str,
    period: str = Query("24h", description="Time period: 1h, 24h, 7d, 30d"),
    interval: str = Query("hour", description="Interval: minute, hour, day"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    """
    Retrieve time-series sentiment trends for a specific entity.
    
    Parameters:
        entity_type (str): Type of the entity (e.g., "team", "player").
        entity_id (str): Identifier of the entity to fetch trends for.
        period (str): Time window for the trends (e.g., "1h", "24h", "7d", "30d").
        interval (str): Aggregation interval for the time series ("minute", "hour", "day").
    
    Returns:
        trends_response (dict): Dictionary with key "trends" mapping to a list of trend dictionaries.
    """
    try:
        trends = await analytics_service.get_sentiment_trends(
            entity_type=entity_type,
            entity_id=entity_id,
            period=period,
            interval=interval,
        )
        return {"trends": [trend.dict() for trend in trends]}
    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while retrieving trends"
        )


@router.get("/leaderboards/{leaderboard_type}")
async def get_sentiment_leaderboards(
    leaderboard_type: str,
    entity_type: str = Query("team", description="Entity type: team or player"),
    limit: int = Query(10, description="Number of results to return"),
    time_period: str = Query("24h", description="Time period: 1h, 24h, 7d"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    """
    Return sentiment leaderboards for a specified leaderboard type and entity scope.
    
    Retrieves a ranked leaderboard of entities based on the chosen sentiment metric.
    
    Parameters:
        leaderboard_type (str): Leaderboard to retrieve. Allowed values: "most_positive", "most_negative", "most_volatile", "most_mentioned".
        entity_type (str): Entity scope for the leaderboard (e.g., "team" or "player").
        limit (int): Maximum number of results to return.
        time_period (str): Time window to consider (e.g., "1h", "24h", "7d").
    
    Returns:
        dict: A dictionary with a "leaderboard" key containing a list of leaderboard entries (ranked entities with associated sentiment metrics).
    
    Raises:
        HTTPException: With status 400 if `leaderboard_type` is invalid; with status 500 if an error occurs while retrieving the leaderboard.
    """
    valid_types = ["most_positive", "most_negative", "most_volatile", "most_mentioned"]
    if leaderboard_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid leaderboard type. Must be one of: {valid_types}",
        )

    try:
        leaderboard = await analytics_service.get_sentiment_leaderboards(
            leaderboard_type=leaderboard_type,
            entity_type=entity_type,
            limit=limit,
            time_period=time_period,
        )
        return {"leaderboard": leaderboard}
    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while retrieving leaderboard"
        )


@router.get("/comparison/{entity_type}/{entity_id}")
async def get_historical_comparison(
    entity_type: str,
    entity_id: str,
    periods: Optional[List[str]] = Query(
        ["7d", "30d", "90d"], description="Comparison periods"
    ),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    """
    Retrieve historical sentiment comparisons for an entity across the given periods.
    
    Parameters:
    	periods (List[str], optional): List of period identifiers to compare (e.g., "7d", "30d", "90d"). Defaults to ["7d", "30d", "90d"].
    
    Returns:
    	comparison (dict): Mapping from period string to comparison data for that period (aggregated sentiment metrics and related summary).
    """
    try:
        comparison = await analytics_service.get_historical_comparison(
            entity_type=entity_type, entity_id=entity_id, comparison_periods=periods
        )
        return comparison
    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while retrieving comparison"
        )


@router.get("/insights/{entity_type}/{entity_id}")
async def get_sentiment_insights(
    entity_type: str,
    entity_id: str,
    analysis_period: str = Query("7d", description="Analysis period: 1d, 7d, 30d"),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    """
    Return advanced sentiment insights and actionable recommendations for the specified entity.
    
    Parameters:
        analysis_period (str): Analysis window to evaluate (e.g., "1d", "7d", "30d").
    
    Returns:
        dict: A dictionary containing analysis results and recommended actions for the entity.
    """
    try:
        insights = await analytics_service.get_sentiment_insights(
            entity_type=entity_type,
            entity_id=entity_id,
            analysis_period=analysis_period,
        )
        return insights
    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while generating insights"
        )


@router.get("/export/{data_type}")
async def export_analytics_data(
    data_type: str,
    export_format: str = Query("json", description="Export format: json or csv"),
    entity_type: Optional[str] = Query(None),
    entity_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    """
    Export analytics data in the requested format and return it as a downloadable stream.
    
    Parameters:
        data_type (str): Type of data to export. Must be one of: "metrics", "trends", "leaderboard".
        export_format (str): Output format. Must be "json" or "csv".
        entity_type (Optional[str]): Optional entity type filter; required when `data_type` is "trends".
        entity_id (Optional[str]): Optional entity id filter; required when `data_type` is "trends".
        start_date (Optional[datetime]): Optional start of the date range to export.
        end_date (Optional[datetime]): Optional end of the date range to export.
    
    Returns:
        StreamingResponse: A response streaming the exported file with an appropriate Content-Type
        and a Content-Disposition attachment filename (JSON or CSV).
    """
    valid_data_types = ["metrics", "trends", "leaderboard"]
    if data_type not in valid_data_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data type. Must be one of: {valid_data_types}",
        )

    valid_formats = ["json", "csv"]
    if export_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid export format. Must be one of: {valid_formats}",
        )

    try:
        # Build kwargs for the export function
        export_kwargs = {}
        if entity_type:
            export_kwargs["entity_type"] = entity_type
        if entity_id:
            export_kwargs["entity_id"] = entity_id
        if start_date:
            export_kwargs["start_date"] = start_date
        if end_date:
            export_kwargs["end_date"] = end_date

        # For trends, we need specific parameters
        if data_type == "trends" and not entity_type:
            raise HTTPException(
                status_code=400, detail="entity_type is required for trends export"
            )
        if data_type == "trends" and not entity_id:
            raise HTTPException(
                status_code=400, detail="entity_id is required for trends export"
            )

        exported_data = await analytics_service.export_analytics_data(
            export_format=export_format, data_type=data_type, **export_kwargs
        )

        # Set appropriate content type and filename
        if export_format == "csv":
            media_type = "text/csv"
            filename = (
                f"{data_type}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            )
        else:
            media_type = "application/json"
            filename = (
                f"{data_type}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        # Return as streaming response
        return StreamingResponse(
            io.StringIO(exported_data),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred while exporting data")


# Cache management endpoints (admin only)


@router.get("/cache/stats")
async def get_cache_stats(
    current_user: dict = Depends(get_current_user),
    caching_service: CachingService = Depends(get_caching_service),
):
    """
    Retrieve cache statistics for the application (admin only).
    
    Requires the current user to have the "admin" role.
    
    Returns:
        dict: A mapping of cache statistics.
    
    Raises:
        HTTPException: 403 if the current user is not an admin.
        HTTPException: 500 if an error occurs while retrieving cache statistics.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        stats = await caching_service.get_cache_stats()
        return stats
    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while retrieving cache stats"
        )


@router.delete("/cache/invalidate")
async def invalidate_cache(
    pattern: Optional[str] = Query(None, description="Cache key pattern to invalidate"),
    entity_type: Optional[str] = Query(None, description="Entity type to invalidate"),
    entity_id: Optional[str] = Query(None, description="Entity ID to invalidate"),
    current_user: dict = Depends(get_current_user),
    caching_service: CachingService = Depends(get_caching_service),
):
    """
    Invalidate cached analytics entries by pattern, by specific entity, or all sentiment cache (admin only).
    
    Parameters:
        pattern (str | None): Cache key pattern to delete; when provided, deletes matching keys and returns the number removed.
        entity_type (str | None): Entity type to invalidate (requires `entity_id`); when provided with `entity_id`, invalidates that entity's cache.
        entity_id (str | None): Identifier of the entity to invalidate (requires `entity_type`).
    
    Returns:
        dict: A message describing what was invalidated (e.g., number of entries deleted, specific entity invalidated, or all sentiment cache invalidated).
    
    Raises:
        HTTPException: 403 if the current user is not an admin; 500 for internal errors encountered while invalidating cache.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        if pattern:
            deleted_count = await caching_service.delete_pattern(pattern)
            return {
                "message": f"Invalidated {deleted_count} cache entries matching pattern: {pattern}"
            }

        elif entity_type and entity_id:
            await caching_service.invalidate_entity_cache(entity_type, entity_id)
            return {"message": f"Invalidated cache for {entity_type}: {entity_id}"}

        else:
            await caching_service.invalidate_all_sentiment_cache()
            return {"message": "Invalidated all sentiment cache"}

    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while invalidating cache"
        )


# Data archiving endpoints (admin only)


@router.get("/archive/stats")
async def get_archiving_stats(
    current_user: dict = Depends(get_current_user),
    archiving_service: DataArchivingService = Depends(get_archiving_service),
):
    """
    Retrieve data archiving statistics (admin only).
    
    Returns:
        dict: Archiving statistics (e.g., counts, timestamps, and related metadata).
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        stats = await archiving_service.get_archiving_stats()
        return stats
    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while retrieving archiving stats"
        )


@router.post("/archive/run")
async def run_data_archiving(
    current_user: dict = Depends(get_current_user),
    archiving_service: DataArchivingService = Depends(get_archiving_service),
):
    """
    Run the data archiving process; restricted to admin users.
    
    Invokes the archiving service to archive old sentiment data and returns the service result.
    
    Raises:
        HTTPException: 403 if the current user is not an admin.
        HTTPException: 500 if an error occurs while running the archiving process.
    
    Returns:
        dict: Result of the archiving operation as returned by the archiving service.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        result = await archiving_service.archive_old_sentiment_data()
        return result
    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while running archiving"
        )


@router.post("/archive/maintenance")
async def run_archive_maintenance(
    current_user: dict = Depends(get_current_user),
    archiving_service: DataArchivingService = Depends(get_archiving_service),
):
    """
    Run the full data archiving maintenance cycle.
    
    Requires the current user to have the "admin" role.
    
    Returns:
        result (dict): Result of the maintenance operation.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        result = await archiving_service.run_maintenance()
        return result
    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while running maintenance"
        )


@router.post("/archive/restore")
async def restore_archived_data(
    start_date: datetime,
    end_date: datetime,
    current_user: dict = Depends(get_current_user),
    archiving_service: DataArchivingService = Depends(get_archiving_service),
):
    """
    Restore archived analytics data for the given inclusive date range; requires admin role.
    
    Returns:
        The result returned by the archiving service describing the restore operation.
    
    Raises:
        HTTPException: 403 if the current user is not an admin.
        HTTPException: 500 if an error occurs while restoring archived data.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        result = await archiving_service.restore_from_archive(start_date, end_date)
        return result
    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred while restoring data")


# Database migration endpoints (admin only)


@router.get("/migrations/status")
async def get_migration_status(
    current_user: dict = Depends(get_current_user),
    migration_service: DatabaseMigrationService = Depends(get_migration_service),
):
    """
    Retrieve the current database migration status (admin only).
    
    Returns:
        Migration status information as returned by the migration service (typically a dict or structured status object).
    
    Raises:
        HTTPException: 403 if the current user is not an admin; 500 if an error occurs while retrieving status.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        status = await migration_service.get_migration_status()
        return status
    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while retrieving migration status"
        )


@router.post("/migrations/up")
async def run_migrations(
    target_version: Optional[str] = Query(None, description="Target migration version"),
    current_user: dict = Depends(get_current_user),
    migration_service: DatabaseMigrationService = Depends(get_migration_service),
):
    """
    Run database migrations to the specified target version.
    
    Applies pending migrations up to `target_version` (or to the latest if `target_version` is omitted) and returns the migration service's result.
    
    Parameters:
        target_version (str | None): Target migration version to apply; if omitted, migrations are applied up to the latest version.
    
    Returns:
        dict: Result returned by the migration service describing the outcome of the migration run.
    
    Raises:
        HTTPException: 403 if the current user is not an admin.
        HTTPException: 500 if an error occurs while running migrations.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        result = await migration_service.migrate_up(target_version)
        return result
    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while running migrations"
        )


@router.post("/migrations/down")
async def rollback_migrations(
    target_version: str,
    current_user: dict = Depends(get_current_user),
    migration_service: DatabaseMigrationService = Depends(get_migration_service),
):
    """
    Rollback the database schema to the specified migration version (admin only).
    
    Parameters:
        target_version (str): Target migration version to roll back to; interpretation depends on migration system (e.g., a revision id or timestamp).
    
    Returns:
        result: Result object or message returned by the migration service describing the outcome of the rollback.
    """
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        result = await migration_service.migrate_down(target_version)
        return result
    except HTTPException:
        raise
    except APIError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="An unexpected error occurred while rolling back migrations"
        )
