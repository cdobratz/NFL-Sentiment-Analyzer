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
from app.services.data_archiving_service import get_archiving_service, DataArchivingService
from app.services.database_migration_service import get_migration_service, DatabaseMigrationService
from app.models.sentiment import DataSource, SentimentCategory
from app.core.dependencies import get_current_user
from app.models.user import User

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/metrics")
async def get_sentiment_metrics(
    entity_type: str = Query(..., description="Entity type: team, player, or game"),
    entity_id: Optional[str] = Query(None, description="Specific entity ID"),
    start_date: Optional[datetime] = Query(None, description="Start date for analysis"),
    end_date: Optional[datetime] = Query(None, description="End date for analysis"),
    sources: Optional[List[DataSource]] = Query(None, description="Data sources to include"),
    categories: Optional[List[SentimentCategory]] = Query(None, description="Sentiment categories to include"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get aggregated sentiment metrics for entities"""
    try:
        metrics = await analytics_service.get_aggregated_sentiment_metrics(
            entity_type=entity_type,
            entity_id=entity_id,
            start_date=start_date,
            end_date=end_date,
            sources=sources,
            categories=categories
        )
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")


@router.get("/trends/{entity_type}/{entity_id}")
async def get_sentiment_trends(
    entity_type: str,
    entity_id: str,
    period: str = Query("24h", description="Time period: 1h, 24h, 7d, 30d"),
    interval: str = Query("hour", description="Interval: minute, hour, day"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get sentiment trends over time for a specific entity"""
    try:
        trends = await analytics_service.get_sentiment_trends(
            entity_type=entity_type,
            entity_id=entity_id,
            period=period,
            interval=interval
        )
        return {"trends": [trend.dict() for trend in trends]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving trends: {str(e)}")


@router.get("/leaderboards/{leaderboard_type}")
async def get_sentiment_leaderboards(
    leaderboard_type: str,
    entity_type: str = Query("team", description="Entity type: team or player"),
    limit: int = Query(10, description="Number of results to return"),
    time_period: str = Query("24h", description="Time period: 1h, 24h, 7d"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get sentiment leaderboards"""
    valid_types = ["most_positive", "most_negative", "most_volatile", "most_mentioned"]
    if leaderboard_type not in valid_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid leaderboard type. Must be one of: {valid_types}"
        )
    
    try:
        leaderboard = await analytics_service.get_sentiment_leaderboards(
            leaderboard_type=leaderboard_type,
            entity_type=entity_type,
            limit=limit,
            time_period=time_period
        )
        return {"leaderboard": leaderboard}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving leaderboard: {str(e)}")


@router.get("/comparison/{entity_type}/{entity_id}")
async def get_historical_comparison(
    entity_type: str,
    entity_id: str,
    periods: Optional[List[str]] = Query(["7d", "30d", "90d"], description="Comparison periods"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get historical sentiment comparison for an entity"""
    try:
        comparison = await analytics_service.get_historical_comparison(
            entity_type=entity_type,
            entity_id=entity_id,
            comparison_periods=periods
        )
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving comparison: {str(e)}")


@router.get("/insights/{entity_type}/{entity_id}")
async def get_sentiment_insights(
    entity_type: str,
    entity_id: str,
    analysis_period: str = Query("7d", description="Analysis period: 1d, 7d, 30d"),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Get advanced sentiment insights and recommendations"""
    try:
        insights = await analytics_service.get_sentiment_insights(
            entity_type=entity_type,
            entity_id=entity_id,
            analysis_period=analysis_period
        )
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")


@router.get("/export/{data_type}")
async def export_analytics_data(
    data_type: str,
    export_format: str = Query("json", description="Export format: json or csv"),
    entity_type: Optional[str] = Query(None),
    entity_id: Optional[str] = Query(None),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    analytics_service: AnalyticsService = Depends(get_analytics_service)
):
    """Export analytics data in specified format"""
    valid_data_types = ["metrics", "trends", "leaderboard"]
    if data_type not in valid_data_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid data type. Must be one of: {valid_data_types}"
        )
    
    valid_formats = ["json", "csv"]
    if export_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid export format. Must be one of: {valid_formats}"
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
                status_code=400,
                detail="entity_type is required for trends export"
            )
        if data_type == "trends" and not entity_id:
            raise HTTPException(
                status_code=400,
                detail="entity_id is required for trends export"
            )
        
        exported_data = await analytics_service.export_analytics_data(
            export_format=export_format,
            data_type=data_type,
            **export_kwargs
        )
        
        # Set appropriate content type and filename
        if export_format == "csv":
            media_type = "text/csv"
            filename = f"{data_type}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            media_type = "application/json"
            filename = f"{data_type}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Return as streaming response
        return StreamingResponse(
            io.StringIO(exported_data),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")


# Cache management endpoints (admin only)

@router.get("/cache/stats")
async def get_cache_stats(
    current_user: User = Depends(get_current_user),
    caching_service: CachingService = Depends(get_caching_service)
):
    """Get cache statistics (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        stats = await caching_service.get_cache_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving cache stats: {str(e)}")


@router.delete("/cache/invalidate")
async def invalidate_cache(
    pattern: Optional[str] = Query(None, description="Cache key pattern to invalidate"),
    entity_type: Optional[str] = Query(None, description="Entity type to invalidate"),
    entity_id: Optional[str] = Query(None, description="Entity ID to invalidate"),
    current_user: User = Depends(get_current_user),
    caching_service: CachingService = Depends(get_caching_service)
):
    """Invalidate cache entries (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        if pattern:
            deleted_count = await caching_service.delete_pattern(pattern)
            return {"message": f"Invalidated {deleted_count} cache entries matching pattern: {pattern}"}
        
        elif entity_type and entity_id:
            await caching_service.invalidate_entity_cache(entity_type, entity_id)
            return {"message": f"Invalidated cache for {entity_type}: {entity_id}"}
        
        else:
            await caching_service.invalidate_all_sentiment_cache()
            return {"message": "Invalidated all sentiment cache"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invalidating cache: {str(e)}")


# Data archiving endpoints (admin only)

@router.get("/archive/stats")
async def get_archiving_stats(
    current_user: User = Depends(get_current_user),
    archiving_service: DataArchivingService = Depends(get_archiving_service)
):
    """Get data archiving statistics (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        stats = await archiving_service.get_archiving_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving archiving stats: {str(e)}")


@router.post("/archive/run")
async def run_data_archiving(
    current_user: User = Depends(get_current_user),
    archiving_service: DataArchivingService = Depends(get_archiving_service)
):
    """Run data archiving process (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        result = await archiving_service.archive_old_sentiment_data()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running archiving: {str(e)}")


@router.post("/archive/maintenance")
async def run_archive_maintenance(
    current_user: User = Depends(get_current_user),
    archiving_service: DataArchivingService = Depends(get_archiving_service)
):
    """Run complete archiving maintenance cycle (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        result = await archiving_service.run_maintenance()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running maintenance: {str(e)}")


@router.post("/archive/restore")
async def restore_archived_data(
    start_date: datetime,
    end_date: datetime,
    current_user: User = Depends(get_current_user),
    archiving_service: DataArchivingService = Depends(get_archiving_service)
):
    """Restore archived data for a date range (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        result = await archiving_service.restore_from_archive(start_date, end_date)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error restoring data: {str(e)}")


# Database migration endpoints (admin only)

@router.get("/migrations/status")
async def get_migration_status(
    current_user: User = Depends(get_current_user),
    migration_service: DatabaseMigrationService = Depends(get_migration_service)
):
    """Get database migration status (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        status = await migration_service.get_migration_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving migration status: {str(e)}")


@router.post("/migrations/up")
async def run_migrations(
    target_version: Optional[str] = Query(None, description="Target migration version"),
    current_user: User = Depends(get_current_user),
    migration_service: DatabaseMigrationService = Depends(get_migration_service)
):
    """Run database migrations (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        result = await migration_service.migrate_up(target_version)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running migrations: {str(e)}")


@router.post("/migrations/down")
async def rollback_migrations(
    target_version: str,
    current_user: User = Depends(get_current_user),
    migration_service: DatabaseMigrationService = Depends(get_migration_service)
):
    """Rollback database migrations (admin only)"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        result = await migration_service.migrate_down(target_version)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rolling back migrations: {str(e)}")