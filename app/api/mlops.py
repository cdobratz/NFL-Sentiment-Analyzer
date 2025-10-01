"""
MLOps API endpoints for model management, deployment, and monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..models.mlops import (
    ModelRegistrationRequest, ModelDeploymentRequest, ModelRetrainingRequest,
    ExperimentCreateRequest, ModelMetadata, ModelDeployment, ExperimentRun,
    ModelPerformanceMetrics, ModelType, DeploymentStrategy
)
from ..services.mlops.mlops_service import mlops_service
from ..core.dependencies import get_current_user
from ..models.user import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/mlops", tags=["MLOps"])


@router.on_event("startup")
async def initialize_mlops():
    """Initialize MLOps services on startup"""
    try:
        await mlops_service.initialize()
        logger.info("MLOps services initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MLOps services: {e}")


# Model Management Endpoints

@router.post("/models/register", response_model=Dict[str, Any])
async def register_model(
    request: ModelRegistrationRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Register a new model in the MLOps pipeline
    """
    try:
        model_metadata = await mlops_service.register_model(
            model_name=request.model_name,
            model_path=request.model_path,
            model_type=request.model_type,
            version=request.version,
            metrics=request.metrics,
            description=request.description,
            tags=request.tags
        )
        
        return {
            "success": True,
            "model_id": model_metadata.model_id,
            "version": model_metadata.version,
            "status": model_metadata.status.value,
            "message": f"Model {request.model_name} registered successfully"
        }
        
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(
    current_user: User = Depends(get_current_user)
):
    """
    List available models
    """
    try:
        models = await mlops_service.list_models()
        return models
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}", response_model=Dict[str, Any])
async def get_model_info(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a specific model
    """
    try:
        model_info = await mlops_service.get_model_info(model_id)
        
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Deployment Endpoints

@router.post("/deployments", response_model=Dict[str, Any])
async def deploy_model(
    request: ModelDeploymentRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Deploy a model to the specified environment
    """
    try:
        deployment = await mlops_service.deploy_model(
            model_id=request.model_id,
            model_version=request.model_version,
            environment=request.environment,
            strategy=request.strategy,
            traffic_percentage=request.traffic_percentage
        )
        
        return {
            "success": True,
            "deployment_id": deployment.deployment_id,
            "status": deployment.status.value,
            "environment": deployment.environment,
            "message": f"Model deployed successfully to {request.environment}"
        }
        
    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments", response_model=List[Dict[str, Any]])
async def list_deployments(
    environment: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    List active deployments
    """
    try:
        deployments = await mlops_service.list_deployments(environment=environment)
        return deployments
        
    except Exception as e:
        logger.error(f"Error listing deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments/{deployment_id}", response_model=Dict[str, Any])
async def get_deployment_status(
    deployment_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get status of a specific deployment
    """
    try:
        status = await mlops_service.deployment_service.get_deployment_status(deployment_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Deployment not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deployment status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/deployments/{deployment_id}/rollback", response_model=Dict[str, Any])
async def rollback_deployment(
    deployment_id: str,
    reason: str,
    target_version: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Rollback a deployment to a previous version
    """
    try:
        success = await mlops_service.deployment_service.rollback_deployment(
            deployment_id=deployment_id,
            reason=reason,
            target_version=target_version
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Rollback failed")
        
        return {
            "success": True,
            "message": f"Deployment {deployment_id} rolled back successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rolling back deployment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# A/B Testing Endpoints

@router.post("/ab-tests", response_model=Dict[str, Any])
async def create_ab_test(
    test_name: str,
    model_a_id: str,
    model_a_version: str,
    model_b_id: str,
    model_b_version: str,
    traffic_split: float = 50.0,
    duration_days: int = 7,
    current_user: User = Depends(get_current_user)
):
    """
    Create an A/B test between two models
    """
    try:
        test_id = await mlops_service.create_ab_test(
            test_name=test_name,
            model_a_id=model_a_id,
            model_a_version=model_a_version,
            model_b_id=model_b_id,
            model_b_version=model_b_version,
            traffic_split=traffic_split,
            duration_days=duration_days
        )
        
        return {
            "success": True,
            "test_id": test_id,
            "message": f"A/B test '{test_name}' created successfully"
        }
        
    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Experiment Tracking Endpoints

@router.post("/experiments", response_model=Dict[str, Any])
async def start_experiment(
    request: ExperimentCreateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Start a new experiment for model training or evaluation
    """
    try:
        run_id = await mlops_service.start_experiment(
            experiment_name=request.experiment_name,
            config=request.hyperparameters,
            tags=request.tags,
            notes=request.notes
        )
        
        return {
            "success": True,
            "run_id": run_id,
            "experiment_name": request.experiment_name,
            "message": "Experiment started successfully"
        }
        
    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/metrics", response_model=Dict[str, Any])
async def log_experiment_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Log metrics to the current experiment
    """
    try:
        await mlops_service.log_metrics(metrics, step=step)
        
        return {
            "success": True,
            "message": "Metrics logged successfully"
        }
        
    except Exception as e:
        logger.error(f"Error logging metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/finish", response_model=Dict[str, Any])
async def finish_experiment(
    final_metrics: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Finish the current experiment
    """
    try:
        await mlops_service.finish_experiment(final_metrics=final_metrics)
        
        return {
            "success": True,
            "message": "Experiment finished successfully"
        }
        
    except Exception as e:
        logger.error(f"Error finishing experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments", response_model=List[Dict[str, Any]])
async def list_experiments(
    experiment_name: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """
    List experiment runs
    """
    try:
        experiments = await mlops_service.list_experiments(
            experiment_name=experiment_name,
            limit=limit
        )
        
        return experiments
        
    except Exception as e:
        logger.error(f"Error listing experiments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Retraining Endpoints

@router.post("/retraining/trigger", response_model=Dict[str, Any])
async def trigger_model_retraining(
    request: ModelRetrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
):
    """
    Manually trigger model retraining
    """
    try:
        run_id = await mlops_service.trigger_retraining(
            model_name=request.model_name,
            reason=request.trigger_reason,
            config=request.training_config
        )
        
        return {
            "success": True,
            "run_id": run_id,
            "message": f"Retraining triggered for {request.model_name}"
        }
        
    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retraining/jobs", response_model=List[Dict[str, Any]])
async def list_retraining_jobs(
    model_name: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """
    List retraining jobs
    """
    try:
        jobs = await mlops_service.retraining_service.list_retraining_jobs(
            model_name=model_name,
            limit=limit
        )
        
        return jobs
        
    except Exception as e:
        logger.error(f"Error listing retraining jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retraining/jobs/{run_id}", response_model=Dict[str, Any])
async def get_retraining_status(
    run_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get status of a specific retraining job
    """
    try:
        status = await mlops_service.retraining_service.get_retraining_status(run_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Retraining job not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting retraining status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retraining/triggers/{model_name}", response_model=Dict[str, Any])
async def check_retraining_triggers(
    model_name: str,
    current_user: User = Depends(get_current_user)
):
    """
    Check if any retraining triggers are activated for a model
    """
    try:
        trigger_status = await mlops_service.retraining_service.check_retraining_triggers(
            model_name
        )
        
        return trigger_status
        
    except Exception as e:
        logger.error(f"Error checking retraining triggers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model Prediction Endpoints

@router.post("/predict/sentiment", response_model=Dict[str, Any])
async def predict_sentiment(
    text: str,
    model_name: str = "sentiment_base",
    model_version: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """
    Predict sentiment using a deployed model
    """
    try:
        prediction = await mlops_service.predict_sentiment(
            text=text,
            model_name=model_name,
            model_version=model_version
        )
        
        return {
            "success": True,
            "prediction": prediction,
            "model_name": model_name,
            "model_version": model_version
        }
        
    except Exception as e:
        logger.error(f"Error predicting sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict/sentiment/batch", response_model=Dict[str, Any])
async def batch_predict_sentiment(
    texts: List[str],
    model_name: str = "sentiment_base",
    model_version: Optional[str] = None,
    batch_size: int = 32,
    current_user: User = Depends(get_current_user)
):
    """
    Batch predict sentiment for multiple texts
    """
    try:
        if len(texts) > 1000:
            raise HTTPException(status_code=400, detail="Maximum 1000 texts allowed per batch")
        
        predictions = await mlops_service.batch_predict_sentiment(
            texts=texts,
            model_name=model_name,
            model_version=model_version,
            batch_size=batch_size
        )
        
        return {
            "success": True,
            "predictions": predictions,
            "total_processed": len(predictions),
            "model_name": model_name,
            "model_version": model_version
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch sentiment prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Monitoring and Performance Endpoints

@router.post("/monitoring/performance", response_model=Dict[str, Any])
async def report_model_performance(
    model_id: str,
    model_version: str,
    performance_metrics: ModelPerformanceMetrics,
    current_user: User = Depends(get_current_user)
):
    """
    Report model performance metrics for monitoring
    """
    try:
        await mlops_service.monitor_model_performance(
            model_id=model_id,
            model_version=model_version,
            performance_metrics=performance_metrics
        )
        
        return {
            "success": True,
            "message": "Performance metrics reported successfully"
        }
        
    except Exception as e:
        logger.error(f"Error reporting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Feature Store Endpoints

@router.post("/features/update", response_model=Dict[str, Any])
async def update_features(
    sentiment_results: List[Dict[str, Any]],
    current_user: User = Depends(get_current_user)
):
    """
    Update feature store with sentiment analysis results
    """
    try:
        # Convert dict to SentimentResult objects (simplified)
        from ..models.sentiment import SentimentResult, SentimentLabel, DataSource, SentimentCategory, AnalysisContext
        
        results = []
        for result_data in sentiment_results:
            result = SentimentResult(
                text=result_data.get("text", ""),
                sentiment=SentimentLabel(result_data.get("sentiment", "NEUTRAL")),
                sentiment_score=result_data.get("sentiment_score", 0.0),
                confidence=result_data.get("confidence", 0.0),
                category=SentimentCategory(result_data.get("category", "general")),
                context=AnalysisContext(),
                source=DataSource(result_data.get("source", "user_input")),
                timestamp=datetime.utcnow(),
                model_version="1.0",
                processing_time_ms=0.0
            )
            results.append(result)
        
        success = await mlops_service.update_features_from_sentiment(results)
        
        return {
            "success": success,
            "updated_count": len(results) if success else 0,
            "message": "Features updated successfully" if success else "Failed to update features"
        }
        
    except Exception as e:
        logger.error(f"Error updating features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Status and Health Endpoints

@router.get("/status", response_model=Dict[str, Any])
async def get_mlops_status():
    """
    Get status of all MLOps services
    """
    try:
        status = await mlops_service.get_service_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting MLOps status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint for MLOps services
    """
    try:
        status = await mlops_service.get_service_status()
        
        return {
            "status": "healthy" if status.get("initialized", False) else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": status.get("services", {})
        }
        
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }