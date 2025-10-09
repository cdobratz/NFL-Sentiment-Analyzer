from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import logging
import asyncio
import psutil
import time
from bson import ObjectId
from bson.errors import InvalidId

from ..core.database import get_database, get_redis
from ..core.dependencies import get_current_admin_user
from ..models.user import UserResponse
from ..services.mlops.mlops_service import mlops_service
from ..models.mlops import ModelRetrainingRequest
from ..services.scheduled_tasks_service import (
    get_scheduled_tasks_service,
    ScheduledTasksService,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=List[UserResponse])
async def get_users(
    limit: int = Query(50, ge=1, le=200, description="Number of users to return"),
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    current_admin: dict = Depends(get_current_admin_user),
    db=Depends(get_database),
):
    """
    Return a paginated list of users.

    Each returned user has the MongoDB `_id` converted to a string `id` and excludes the `hashed_password` field.

    Returns:
        List[UserResponse]: List of user objects with `id` as a string and without `hashed_password`.
    """
    cursor = db.users.find({}, {"hashed_password": 0}).skip(skip).limit(limit)
    users = []

    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        users.append(UserResponse(**doc))

    return users


@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str,
    current_admin: dict = Depends(get_current_admin_user),
    db=Depends(get_database),
):
    """
    Retrieve a user by ID and return its public representation.

    Returns:
        UserResponse: The user's data with `id` as a string and `hashed_password` excluded.

    Raises:
        HTTPException: 404 if the user with the given ID is not found.
    """
    # Validate ObjectId
    try:
        object_id = ObjectId(user_id)
    except (InvalidId, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user id"
        )
    
    user = await db.users.find_one({"_id": object_id}, {"hashed_password": 0})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    user["id"] = str(user["_id"])
    return UserResponse(**user)


@router.put("/users/{user_id}/deactivate")
async def deactivate_user(
    user_id: str,
    current_admin: dict = Depends(get_current_admin_user),
    db=Depends(get_database),
):
    """
    Deactivate a user account by ID.

    Parameters:
        user_id (str): The user's identifier (MongoDB `_id` as a string).

    Returns:
        dict: {"message": "User deactivated successfully"} on success.

    Raises:
        HTTPException: 404 if no user with the given `user_id` exists.
    """
    # Validate ObjectId
    try:
        object_id = ObjectId(user_id)
    except (InvalidId, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user id"
        )
    
    result = await db.users.update_one(
        {"_id": object_id},
        {"$set": {"is_active": False, "deactivated_at": datetime.utcnow()}},
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return {"message": "User deactivated successfully"}


@router.put("/users/{user_id}/activate")
async def activate_user(
    user_id: str,
    current_admin: dict = Depends(get_current_admin_user),
    db=Depends(get_database),
):
    """
    Activate the user with the given ID.

    Parameters:
        user_id (str): The user's MongoDB `_id` as a string.

    Returns:
        dict: A success message, e.g. {"message": "User activated successfully"}.

    Raises:
        HTTPException: HTTP 404 if no user with the given ID is found.
    """
    # Validate ObjectId
    try:
        object_id = ObjectId(user_id)
    except (InvalidId, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user id"
        )
    
    result = await db.users.update_one(
        {"_id": object_id},
        {"$set": {"is_active": True}, "$unset": {"deactivated_at": ""}},
    )

    if result.matched_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found"
        )

    return {"message": "User activated successfully"}


@router.get("/system-health")
async def get_system_health(
    current_admin: dict = Depends(get_current_admin_user),
    db=Depends(get_database),
    redis=Depends(get_redis),
):
    """
    Gather a comprehensive health report for system services and resources.

    Returns:
        dict: A dictionary containing:
            - timestamp (datetime): UTC time of the check.
            - services (dict): Health details per service (e.g., mongodb, redis, mlops). Each service entry includes a `status` and service-specific diagnostics such as response_time_ms, sizes, initialization state, or error messages.
            - system_resources (dict): Resource metrics (cpu_usage_percent, memory_usage_percent, memory_available_gb, disk_usage_percent, disk_free_gb) or an error message if unavailable.
            - overall_status (str): Overall health state, either "healthy" or "degraded".
    """
    health_status = {
        "timestamp": datetime.utcnow(),
        "services": {},
        "system_resources": {},
        "overall_status": "healthy",
    }

    # Check MongoDB
    try:
        start_time = time.time()
        await db.command("ping")
        response_time = (time.time() - start_time) * 1000

        # Get database stats
        db_stats = await db.command("dbStats")

        health_status["services"]["mongodb"] = {
            "status": "healthy",
            "response_time_ms": round(response_time, 2),
            "database_size_mb": round(db_stats.get("dataSize", 0) / (1024 * 1024), 2),
            "collections": db_stats.get("collections", 0),
            "indexes": db_stats.get("indexes", 0),
        }
    except Exception as e:
        health_status["services"]["mongodb"] = {"status": "unhealthy", "error": str(e)}
        health_status["overall_status"] = "degraded"

    # Check Redis
    if redis:
        try:
            start_time = time.time()
            await redis.ping()
            response_time = (time.time() - start_time) * 1000

            # Get Redis info
            redis_info = await redis.info()

            health_status["services"]["redis"] = {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "memory_used_mb": round(
                    redis_info.get("used_memory", 0) / (1024 * 1024), 2
                ),
                "connected_clients": redis_info.get("connected_clients", 0),
                "total_commands_processed": redis_info.get(
                    "total_commands_processed", 0
                ),
            }
        except Exception as e:
            health_status["services"]["redis"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["overall_status"] = "degraded"
    else:
        health_status["services"]["redis"] = {"status": "not_configured"}

    # Check MLOps services
    try:
        mlops_status = await mlops_service.get_service_status()
        health_status["services"]["mlops"] = {
            "status": "healthy" if mlops_status.get("initialized") else "unhealthy",
            "initialized": mlops_status.get("initialized", False),
            "services": mlops_status.get("services", {}),
        }

        if not mlops_status.get("initialized"):
            health_status["overall_status"] = "degraded"

    except Exception as e:
        health_status["services"]["mlops"] = {"status": "unhealthy", "error": str(e)}
        health_status["overall_status"] = "degraded"

    # Get system resource usage
    try:
        cpu_percent = await asyncio.to_thread(psutil.cpu_percent, interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        health_status["system_resources"] = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
        }

        # Mark as degraded if resources are high
        if cpu_percent > 80 or memory.percent > 85 or disk.percent > 90:
            health_status["overall_status"] = "degraded"

    except Exception as e:
        health_status["system_resources"] = {
            "error": f"Unable to get system resources: {str(e)}"
        }

    return health_status


@router.get("/analytics")
async def get_analytics(
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    current_admin: dict = Depends(get_current_admin_user),
    db=Depends(get_database),
):
    """
    Compile system analytics across users, sentiment analyses, model performance, API usage, and errors for a recent period.

    Parameters:
        days (int): Number of days to include in the analysis (1-30).

    Returns:
        dict: Analytics payload containing:
            - "period", "period_start", "period_end", "generated_at" (ISO 8601 timestamps)
            - "users": totals and daily registrations
            - "sentiment_analyses": totals, distribution, confidence averages, daily volumes, and top teams
            - "model_performance": per-model aggregated metrics (accuracy, F1, prediction time, predictions, error rate)
            - "api_usage": daily usage with total requests, unique users, and average response time
            - "errors": top error types with counts and last occurrence timestamps (ISO 8601)
    """
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)

    analytics = {
        "period": f"{days} days",
        "period_start": start_date.isoformat(),
        "period_end": end_date.isoformat(),
        "generated_at": datetime.utcnow().isoformat(),
    }

    # Get user statistics
    total_users = await db.users.count_documents({})
    active_users = await db.users.count_documents({"is_active": True})
    new_users = await db.users.count_documents({"created_at": {"$gte": start_date}})

    # Get daily user registrations
    daily_registrations_pipeline = [
        {"$match": {"created_at": {"$gte": start_date}}},
        {
            "$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    daily_registrations = {}
    async for doc in db.users.aggregate(daily_registrations_pipeline):
        daily_registrations[doc["_id"]] = doc["count"]

    analytics["users"] = {
        "total": total_users,
        "active": active_users,
        "inactive": total_users - active_users,
        "new": new_users,
        "daily_registrations": daily_registrations,
    }

    # Get sentiment analysis statistics
    total_analyses = await db.sentiment_analyses.count_documents({})
    recent_analyses = await db.sentiment_analyses.count_documents(
        {"timestamp": {"$gte": start_date}}
    )

    # Get sentiment distribution
    sentiment_pipeline = [
        {"$match": {"timestamp": {"$gte": start_date}}},
        {
            "$group": {
                "_id": "$sentiment",
                "count": {"$sum": 1},
                "avg_confidence": {"$avg": "$confidence"},
            }
        },
    ]

    sentiment_distribution = {}
    sentiment_confidence = {}
    async for doc in db.sentiment_analyses.aggregate(sentiment_pipeline):
        sentiment_distribution[doc["_id"]] = doc["count"]
        sentiment_confidence[doc["_id"]] = round(doc.get("avg_confidence", 0), 3)

    # Get daily sentiment analysis volume
    daily_analyses_pipeline = [
        {"$match": {"timestamp": {"$gte": start_date}}},
        {
            "$group": {
                "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}},
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id": 1}},
    ]

    daily_analyses = {}
    async for doc in db.sentiment_analyses.aggregate(daily_analyses_pipeline):
        daily_analyses[doc["_id"]] = doc["count"]

    # Get team-specific sentiment statistics
    team_sentiment_pipeline = [
        {"$match": {"timestamp": {"$gte": start_date}, "team_id": {"$exists": True}}},
        {
            "$group": {
                "_id": "$team_id",
                "total_analyses": {"$sum": 1},
                "avg_sentiment_score": {"$avg": "$sentiment_score"},
                "positive_count": {
                    "$sum": {"$cond": [{"$eq": ["$sentiment", "positive"]}, 1, 0]}
                },
                "negative_count": {
                    "$sum": {"$cond": [{"$eq": ["$sentiment", "negative"]}, 1, 0]}
                },
                "neutral_count": {
                    "$sum": {"$cond": [{"$eq": ["$sentiment", "neutral"]}, 1, 0]}
                },
            }
        },
        {"$sort": {"total_analyses": -1}},
        {"$limit": 10},
    ]

    team_sentiment_stats = []
    async for doc in db.sentiment_analyses.aggregate(team_sentiment_pipeline):
        team_sentiment_stats.append(
            {
                "team_id": doc["_id"],
                "total_analyses": doc["total_analyses"],
                "avg_sentiment_score": round(doc.get("avg_sentiment_score", 0), 3),
                "positive_count": doc["positive_count"],
                "negative_count": doc["negative_count"],
                "neutral_count": doc["neutral_count"],
            }
        )

    analytics["sentiment_analyses"] = {
        "total": total_analyses,
        "recent": recent_analyses,
        "distribution": sentiment_distribution,
        "confidence_by_sentiment": sentiment_confidence,
        "daily_volume": daily_analyses,
        "top_teams": team_sentiment_stats,
    }

    # Get model performance analytics
    try:
        model_performance_pipeline = [
            {"$match": {"timestamp": {"$gte": start_date}}},
            {
                "$group": {
                    "_id": "$model_id",
                    "avg_accuracy": {"$avg": "$accuracy"},
                    "avg_f1_score": {"$avg": "$f1_score"},
                    "avg_prediction_time": {"$avg": "$avg_prediction_time_ms"},
                    "total_predictions": {"$sum": "$prediction_count"},
                    "avg_error_rate": {"$avg": "$error_rate"},
                }
            },
            {"$sort": {"total_predictions": -1}},
        ]

        model_performance = []
        async for doc in db.performance_metrics.aggregate(model_performance_pipeline):
            model_performance.append(
                {
                    "model_id": doc["_id"],
                    "avg_accuracy": round(doc.get("avg_accuracy", 0), 3),
                    "avg_f1_score": round(doc.get("avg_f1_score", 0), 3),
                    "avg_prediction_time_ms": round(
                        doc.get("avg_prediction_time", 0), 2
                    ),
                    "total_predictions": doc["total_predictions"],
                    "avg_error_rate": round(doc.get("avg_error_rate", 0), 4),
                }
            )

        analytics["model_performance"] = model_performance

    except Exception as e:
        logger.warning(f"Could not get model performance analytics: {e}")
        analytics["model_performance"] = []

    # Get API usage statistics
    try:
        api_usage_pipeline = [
            {"$match": {"timestamp": {"$gte": start_date}}},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}
                    },
                    "total_requests": {"$sum": 1},
                    "unique_users": {"$addToSet": "$user_id"},
                    "avg_response_time": {"$avg": "$response_time_ms"},
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "total_requests": 1,
                    "unique_users": {"$size": "$unique_users"},
                    "avg_response_time": {"$round": ["$avg_response_time", 2]},
                }
            },
            {"$sort": {"_id": 1}},
        ]

        daily_api_usage = {}
        async for doc in db.api_logs.aggregate(api_usage_pipeline):
            daily_api_usage[doc["_id"]] = {
                "total_requests": doc["total_requests"],
                "unique_users": doc["unique_users"],
                "avg_response_time_ms": doc.get("avg_response_time", 0),
            }

        analytics["api_usage"] = {"daily_usage": daily_api_usage}

    except Exception as e:
        logger.warning(f"Could not get API usage analytics: {e}")
        analytics["api_usage"] = {"daily_usage": {}}

    # Get error statistics
    try:
        error_stats_pipeline = [
            {
                "$match": {
                    "timestamp": {"$gte": start_date},
                    "level": {"$in": ["ERROR", "CRITICAL"]},
                }
            },
            {
                "$group": {
                    "_id": "$error_type",
                    "count": {"$sum": 1},
                    "last_occurrence": {"$max": "$timestamp"},
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": 10},
        ]

        error_stats = []
        async for doc in db.error_logs.aggregate(error_stats_pipeline):
            error_stats.append(
                {
                    "error_type": doc["_id"],
                    "count": doc["count"],
                    "last_occurrence": doc["last_occurrence"].isoformat(),
                }
            )

        analytics["errors"] = {"top_errors": error_stats}

    except Exception as e:
        logger.warning(f"Could not get error statistics: {e}")
        analytics["errors"] = {"top_errors": []}

    return analytics


@router.post("/retrain-models")
async def retrain_models(
    request: Optional[ModelRetrainingRequest] = None,
    current_admin: dict = Depends(get_current_admin_user),
    db=Depends(get_database),
):
    """
    Trigger a retraining job for a machine learning model and record the job in the database.

    If `request` is omitted, defaults are used: model_name="sentiment_base", trigger_reason="manual_admin_trigger", training_config=None, and auto_deploy=False.

    Parameters:
        request (Optional[ModelRetrainingRequest]): Optional retraining request specifying `model_name`, `trigger_reason`, `training_config`, and `auto_deploy`. When omitted, server-default values are applied.

    Returns:
        dict: Information about the started retraining job including:
            - `message`: success message,
            - `job_id`: internal job identifier,
            - `experiment_id`: experiment identifier,
            - `model_name`: name of the model being retrained,
            - `status`: job status string,
            - `started_at`: ISO-formatted start timestamp.

    Raises:
        HTTPException: Raised with status 500 if triggering the retraining or recording the job fails.
    """
    try:
        # Default values if no request body provided
        model_name = request.model_name if request else "sentiment_base"
        trigger_reason = request.trigger_reason if request else "manual_admin_trigger"
        training_config = request.training_config if request else None
        auto_deploy = request.auto_deploy if request else False

        # Initialize MLOps service if not already done
        if not mlops_service.initialized:
            await mlops_service.initialize()

        # Trigger retraining through MLOps service
        experiment_run = await mlops_service.trigger_retraining(
            model_name=model_name, reason=trigger_reason, config=training_config
        )

        # Store job record in database
        retraining_job = {
            "job_id": experiment_run.run_id,
            "experiment_id": experiment_run.experiment_id,
            "model_name": model_name,
            "status": experiment_run.status.value,
            "trigger_reason": trigger_reason,
            "triggered_by": str(current_admin["_id"]),
            "auto_deploy": auto_deploy,
            "created_at": datetime.utcnow(),
            "started_at": experiment_run.started_at,
        }

        await db.ml_jobs.insert_one(retraining_job)

        return {
            "message": "Model retraining job started successfully",
            "job_id": experiment_run.run_id,
            "experiment_id": experiment_run.experiment_id,
            "model_name": model_name,
            "status": experiment_run.status.value,
            "started_at": experiment_run.started_at.isoformat(),
        }

    except Exception as e:
        logger.error(f"Error triggering model retraining: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger model retraining: {str(e)}",
        )


@router.get("/ml-jobs")
async def get_ml_jobs(
    limit: int = Query(20, ge=1, le=100, description="Number of jobs to return"),
    status: Optional[str] = Query(None, description="Filter by job status"),
    model_name: Optional[str] = Query(None, description="Filter by model name"),
    current_admin: dict = Depends(get_current_admin_user),
    db=Depends(get_database),
):
    """
    Return ML job history and summary statistics filtered by status and model name.

    Parameters:
        limit (int): Maximum number of jobs to return (1-100).
        status (Optional[str]): Filter jobs by their status when provided.
        model_name (Optional[str]): Filter jobs by model name when provided.

    Returns:
        dict: A mapping with:
            - "jobs": list of job documents where MongoDB `_id` is converted to string `id`
              and datetime fields (`created_at`, `started_at`, `completed_at`) are ISO-formatted strings.
            - "statistics": dict with counts `total`, `running`, `completed`, and `failed`.
    """
    # Build query
    query = {}
    if status:
        query["status"] = status
    if model_name:
        query["model_name"] = model_name

    cursor = db.ml_jobs.find(query).sort([("created_at", -1)]).limit(limit)
    jobs = []

    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        # Convert datetime objects to ISO strings
        if "created_at" in doc:
            doc["created_at"] = doc["created_at"].isoformat()
        if "started_at" in doc:
            doc["started_at"] = doc["started_at"].isoformat()
        if "completed_at" in doc:
            doc["completed_at"] = doc["completed_at"].isoformat()
        jobs.append(doc)

    # Get job statistics
    total_jobs = await db.ml_jobs.count_documents(query)
    running_jobs = await db.ml_jobs.count_documents({**query, "status": "running"})
    completed_jobs = await db.ml_jobs.count_documents({**query, "status": "completed"})
    failed_jobs = await db.ml_jobs.count_documents({**query, "status": "failed"})

    return {
        "jobs": jobs,
        "statistics": {
            "total": total_jobs,
            "running": running_jobs,
            "completed": completed_jobs,
            "failed": failed_jobs,
        },
    }


@router.get("/models")
async def get_models(current_admin: dict = Depends(get_current_admin_user)):
    """
    Retrieve available ML models, active deployments, and recent experiments.

    Returns:
        result (dict): Dictionary containing:
            - models: list of available model metadata.
            - deployments: list of active deployments.
            - recent_experiments: list of recent experiments (limited to 10).
            - timestamp: ISO-formatted UTC timestamp of when the data was retrieved.
    """
    try:
        # Initialize MLOps service if not already done
        if not mlops_service.initialized:
            await mlops_service.initialize()

        # Get available models
        models = await mlops_service.list_models()

        # Get active deployments
        deployments = await mlops_service.list_deployments()

        # Get recent experiments
        experiments = await mlops_service.list_experiments(limit=10)

        return {
            "models": models,
            "deployments": deployments,
            "recent_experiments": experiments,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get models: {str(e)}",
        )


@router.get("/models/{model_id}")
async def get_model_details(
    model_id: str, current_admin: dict = Depends(get_current_admin_user)
):
    """
    Retrieve detailed information for a specific model.

    Parameters:
        model_id (str): Identifier of the model to retrieve.

    Returns:
        dict: Model details object as returned by the MLOps service.

    Raises:
        HTTPException: `404` if the model is not found; `500` if an unexpected error occurs while fetching model details.
    """
    try:
        # Initialize MLOps service if not already done
        if not mlops_service.initialized:
            await mlops_service.initialize()

        # Get model information
        model_info = await mlops_service.get_model_info(model_id)

        if not model_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found",
            )

        return model_info

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model details: {str(e)}",
        )


@router.get("/model-performance")
async def get_model_performance(
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    current_admin: dict = Depends(get_current_admin_user),
    db=Depends(get_database),
):
    """
    Return aggregated performance metrics for models over a recent time window.

    Parameters:
        model_id (Optional[str]): Filter results to a single model by its identifier; if omitted, metrics for all models are returned.
        days (int): Number of past days to include in the analysis (minimum 1, maximum 30).

    Returns:
        dict: A dictionary containing:
            - "period": string describing the window (e.g., "7 days").
            - "model_id": the provided model_id or None.
            - "metrics": list of metric documents with ISO-formatted timestamps and string `id` fields.
            - "summary": aggregated values including:
                - "avg_accuracy": average accuracy rounded to three decimals.
                - "avg_f1_score": average F1 score rounded to three decimals.
                - "avg_response_time_ms": average prediction time in milliseconds rounded to two decimals.
                - "total_predictions": total number of predictions summed over the period.
                - "avg_error_rate": average error rate rounded to four decimals.
    """
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Build query
        query = {"timestamp": {"$gte": start_date}}
        if model_id:
            query["model_id"] = model_id

        # Get performance metrics
        cursor = db.performance_metrics.find(query).sort([("timestamp", -1)])
        metrics = []

        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            doc["timestamp"] = doc["timestamp"].isoformat()
            if "period_start" in doc:
                doc["period_start"] = doc["period_start"].isoformat()
            if "period_end" in doc:
                doc["period_end"] = doc["period_end"].isoformat()
            metrics.append(doc)

        # Calculate aggregated metrics
        if metrics:
            # Collect numeric values including zeros, excluding None
            accuracy_values = [
                m.get("accuracy") for m in metrics if m.get("accuracy") is not None
            ]
            f1_score_values = [
                m.get("f1_score") for m in metrics if m.get("f1_score") is not None
            ]
            response_time_values = [
                m.get("avg_prediction_time_ms")
                for m in metrics
                if m.get("avg_prediction_time_ms") is not None
            ]
            error_rate_values = [
                m.get("error_rate") for m in metrics if m.get("error_rate") is not None
            ]

            # Calculate averages with division guards
            avg_accuracy = (
                sum(accuracy_values) / len(accuracy_values)
                if len(accuracy_values) > 0
                else 0
            )
            avg_f1_score = (
                sum(f1_score_values) / len(f1_score_values)
                if len(f1_score_values) > 0
                else 0
            )
            avg_response_time = (
                sum(response_time_values) / len(response_time_values)
                if len(response_time_values) > 0
                else 0
            )
            avg_error_rate = (
                sum(error_rate_values) / len(error_rate_values)
                if len(error_rate_values) > 0
                else 0
            )
            total_predictions = sum(m.get("prediction_count", 0) for m in metrics)
        else:
            avg_accuracy = avg_f1_score = avg_response_time = total_predictions = (
                avg_error_rate
            ) = 0

        return {
            "period": f"{days} days",
            "model_id": model_id,
            "metrics": metrics,
            "summary": {
                "avg_accuracy": round(avg_accuracy, 3),
                "avg_f1_score": round(avg_f1_score, 3),
                "avg_response_time_ms": round(avg_response_time, 2),
                "total_predictions": total_predictions,
                "avg_error_rate": round(avg_error_rate, 4),
            },
        }

    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model performance: {str(e)}",
        )


@router.get("/alerts")
async def get_system_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200, description="Number of alerts to return"),
    current_admin: dict = Depends(get_current_admin_user),
    db=Depends(get_database),
):
    """
    Retrieve system alerts filtered by optional severity and status, returning the most recent alerts and summary statistics.

    Alerts are returned sorted by created_at descending and limited by `limit`. Each alert document includes an `id` string (MongoDB _id converted to string) and any datetime fields (`created_at`, `acknowledged_at`, `resolved_at`) are converted to ISO 8601 strings.

    Returns:
        dict: A mapping with two keys:
            - "alerts": list of alert documents with `id` and ISO-formatted date fields.
            - "statistics": dict with integer counts `total`, `active`, and `critical`.

    Raises:
        HTTPException: With status 500 if retrieving alerts or computing statistics fails.
    """
    try:
        # Build query
        query = {}
        if severity:
            query["severity"] = severity
        if status:
            query["status"] = status

        cursor = db.alerts.find(query).sort([("created_at", -1)]).limit(limit)
        alerts = []

        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            # Convert datetime objects to ISO strings
            for field in ["created_at", "acknowledged_at", "resolved_at"]:
                if field in doc and doc[field]:
                    doc[field] = doc[field].isoformat()
            alerts.append(doc)

        # Get alert statistics
        total_alerts = await db.alerts.count_documents(query)
        active_alerts = await db.alerts.count_documents({**query, "status": "active"})
        critical_alerts = await db.alerts.count_documents(
            {**query, "severity": "critical"}
        )

        return {
            "alerts": alerts,
            "statistics": {
                "total": total_alerts,
                "active": active_alerts,
                "critical": critical_alerts,
            },
        }

    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alerts: {str(e)}",
        )


@router.put("/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    current_admin: dict = Depends(get_current_admin_user),
    db=Depends(get_database),
):
    """
    Mark a system alert as acknowledged.

    Sets the alert's status to "acknowledged", records the acknowledgement timestamp, and records the acknowledging admin's id.

    Returns:
        dict: A dictionary containing a confirmation message, e.g. {"message": "Alert acknowledged successfully"}.

    Raises:
        HTTPException: With status 404 if no alert matches the given `alert_id`.
        HTTPException: With status 500 if an unexpected error occurs while updating the alert.
    """
    try:
        result = await db.alerts.update_one(
            {"alert_id": alert_id},
            {
                "$set": {
                    "status": "acknowledged",
                    "acknowledged_at": datetime.utcnow(),
                    "acknowledged_by": str(current_admin["_id"]),
                }
            },
        )

        if result.matched_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found"
            )

        return {"message": "Alert acknowledged successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to acknowledge alert: {str(e)}",
        )


@router.delete("/cache/clear")
async def clear_cache(
    cache_type: Optional[str] = Query(
        None, description="Type of cache to clear (redis, all)"
    ),
    current_admin: dict = Depends(get_current_admin_user),
    redis=Depends(get_redis),
):
    """
    Clear configured application caches according to the requested cache type.

    Parameters:
        cache_type (Optional[str]): Which cache to clear. One of "redis", "mlops", "all", or None to clear all available caches.

    Returns:
        dict: {
            "message": Human-readable summary of cleared caches,
            "cleared_caches": List[str] of cache names that were cleared
        }

    Raises:
        HTTPException: 503 if no cache services are available; 400 if an invalid cache_type is provided; 500 if an internal error occurs while clearing caches.
    """
    try:
        cleared_caches = []

        # Clear Redis cache
        if redis and (cache_type is None or cache_type in ["redis", "all"]):
            await redis.flushdb()
            cleared_caches.append("redis")

        # Clear MLOps cache
        if cache_type is None or cache_type in ["mlops", "all"]:
            if mlops_service.initialized:
                await mlops_service.cleanup_resources()
                cleared_caches.append("mlops")

        if not cleared_caches:
            if not redis:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="No cache services available",
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid cache type: {cache_type}",
                )

        return {
            "message": f"Cache cleared successfully: {', '.join(cleared_caches)}",
            "cleared_caches": cleared_caches,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}",
        )


@router.get("/scheduled-tasks")
async def get_scheduled_tasks_status(
    current_admin: dict = Depends(get_current_admin_user),
    scheduled_tasks: ScheduledTasksService = Depends(get_scheduled_tasks_service),
):
    """
    Get the current status of all scheduled tasks.

    Returns:
        dict: Mapping of task IDs to their status details.

    Raises:
        HTTPException: If retrieving the scheduled tasks status fails.
    """
    try:
        status = scheduled_tasks.get_task_status()
        return status
    except Exception as e:
        logger.error(f"Error getting scheduled tasks status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get scheduled tasks status: {str(e)}",
        )


@router.post("/scheduled-tasks/{task_id}/run")
async def run_scheduled_task_manually(
    task_id: str,
    current_admin: dict = Depends(get_current_admin_user),
    scheduled_tasks: ScheduledTasksService = Depends(get_scheduled_tasks_service),
):
    """
    Execute a scheduled task immediately by task ID.

    Parameters:
        task_id (str): Identifier of the scheduled task to run.

    Returns:
        result: The value returned by the scheduled tasks service for the executed task.

    Raises:
        HTTPException: Raised with status 400 if the task request is invalid (ValueError from the service),
                or status 500 for unexpected errors while running the task.
    """
    try:
        result = await scheduled_tasks.run_task_manually(task_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Error running scheduled task {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run scheduled task: {str(e)}",
        )
