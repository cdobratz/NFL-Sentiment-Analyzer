"""
MLOps API endpoints for model management, deployment, and monitoring.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from ..models.mlops import (
    ModelRegistrationRequest,
    ModelDeploymentRequest,
    ModelRetrainingRequest,
    ExperimentCreateRequest,
    ModelMetadata,
    ModelDeployment,
    ExperimentRun,
    ModelPerformanceMetrics,
    ModelType,
    DeploymentStrategy,
)
from ..services.mlops.mlops_service import mlops_service
from ..core.dependencies import get_current_user

# User type is dict from get_current_user dependency

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/mlops", tags=["MLOps"])


@router.on_event("startup")
async def initialize_mlops():
    """
    Initialize MLOps services used by the application at startup.
    
    If initialization fails, the error is logged and the exception is suppressed so startup continues.
    """
    try:
        await mlops_service.initialize()
        logger.info("MLOps services initialized")
    except Exception as e:
        logger.error(f"Failed to initialize MLOps services: {e}")


# Model Management Endpoints


@router.post("/models/register", response_model=Dict[str, Any])
async def register_model(
    request: ModelRegistrationRequest, current_user: dict = Depends(get_current_user)
):
    """
    Register a new model and return its registration metadata.
    
    Returns:
        dict: A payload containing:
            - `success` (bool): `True` when registration succeeded.
            - `model_id` (str): The assigned model identifier.
            - `version` (str): The registered model version.
            - `status` (str): The model status.
            - `message` (str): Human-readable confirmation message.
    
    Raises:
        HTTPException: With status 500 when model registration fails.
    """
    try:
        model_metadata = await mlops_service.register_model(
            model_name=request.model_name,
            model_path=request.model_path,
            model_type=request.model_type,
            version=request.version,
            metrics=request.metrics,
            description=request.description,
            tags=request.tags,
        )

        return {
            "success": True,
            "model_id": model_metadata.model_id,
            "version": model_metadata.version,
            "status": model_metadata.status.value,
            "message": f"Model {request.model_name} registered successfully",
        }

    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models(current_user: dict = Depends(get_current_user)):
    """
    Retrieve the list of registered models.
    
    Returns:
        List[dict]: A list of model metadata dictionaries, each describing a registered model (for example: id, name, versions, status, and related metadata).
    """
    try:
        models = await mlops_service.list_models()
        return models

    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}", response_model=Dict[str, Any])
async def get_model_info(model_id: str, current_user: dict = Depends(get_current_user)):
    """
    Return detailed information for the specified model.
    
    Returns:
        dict: A dictionary containing the model's metadata, available versions, and current status.
    
    Raises:
        HTTPException: with status 404 if the model is not found, or 500 for other internal errors.
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
    request: ModelDeploymentRequest, current_user: dict = Depends(get_current_user)
):
    """
    Deploy a model to the specified environment and return deployment metadata.
    
    Returns:
        dict: Deployment result containing:
            - `success` (bool): True when deployment was initiated successfully.
            - `deployment_id` (str): Identifier of the created deployment.
            - `status` (str): Current deployment status.
            - `environment` (str): Target environment for the deployment.
            - `message` (str): Human-readable summary of the outcome.
    
    Raises:
        HTTPException: If an error occurs while creating the deployment.
    """
    try:
        deployment = await mlops_service.deploy_model(
            model_id=request.model_id,
            model_version=request.model_version,
            environment=request.environment,
            strategy=request.strategy,
            traffic_percentage=request.traffic_percentage,
        )

        return {
            "success": True,
            "deployment_id": deployment.deployment_id,
            "status": deployment.status.value,
            "environment": deployment.environment,
            "message": f"Model deployed successfully to {request.environment}",
        }

    except Exception as e:
        logger.error(f"Error deploying model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments", response_model=List[Dict[str, Any]])
async def list_deployments(
    environment: Optional[str] = None, current_user: dict = Depends(get_current_user)
):
    """
    List deployments, optionally filtered by environment.
    
    Parameters:
        environment (Optional[str]): If provided, only deployments for this environment (e.g., "staging", "production") are returned.
    
    Returns:
        List[dict]: A list of deployment records containing deployment metadata and status.
    """
    try:
        deployments = await mlops_service.list_deployments(environment=environment)
        return deployments

    except Exception as e:
        logger.error(f"Error listing deployments: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deployments/{deployment_id}", response_model=Dict[str, Any])
async def get_deployment_status(
    deployment_id: str, current_user: dict = Depends(get_current_user)
):
    """
    Retrieve the status information for a deployment by its identifier.
    
    Returns:
        status (Any): Deployment status object returned by the deployment service.
    
    Raises:
        HTTPException: 404 if the deployment is not found.
        HTTPException: 500 if an internal error occurs while retrieving the status.
    """
    try:
        status = await mlops_service.deployment_service.get_deployment_status(
            deployment_id
        )

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
    current_user: dict = Depends(get_current_user),
):
    """
    Rollback a deployment to a specified target version with a provided reason.
    
    Parameters:
        deployment_id (str): Identifier of the deployment to roll back.
        reason (str): Human-readable justification for performing the rollback.
        target_version (Optional[str]): Specific model/service version to revert to; if omitted, the service will select the previous stable version.
    
    Returns:
        dict: Response payload containing `success` (True) and a `message` describing the result.
    
    Raises:
        HTTPException: Raised with status code 400 if the rollback could not be performed, or 500 for unexpected internal errors.
    """
    try:
        success = await mlops_service.deployment_service.rollback_deployment(
            deployment_id=deployment_id, reason=reason, target_version=target_version
        )

        if not success:
            raise HTTPException(status_code=400, detail="Rollback failed")

        return {
            "success": True,
            "message": f"Deployment {deployment_id} rolled back successfully",
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
    current_user: dict = Depends(get_current_user),
):
    """
    Create an A/B test that splits traffic between two model versions.
    
    Parameters:
        test_name (str): Human-readable name for the A/B test.
        model_a_id (str): Identifier of model A.
        model_a_version (str): Version of model A to include in the test.
        model_b_id (str): Identifier of model B.
        model_b_version (str): Version of model B to include in the test.
        traffic_split (float): Percentage of traffic routed to model A (0-100). The remainder goes to model B.
        duration_days (int): Duration of the A/B test in days.
    
    Returns:
        dict: {
            "success": True if the test was created, False otherwise,
            "test_id": Identifier of the created A/B test,
            "message": Human-readable status message
        }
    
    Raises:
        HTTPException: If the A/B test creation fails.
    """
    try:
        test_id = await mlops_service.create_ab_test(
            test_name=test_name,
            model_a_id=model_a_id,
            model_a_version=model_a_version,
            model_b_id=model_b_id,
            model_b_version=model_b_version,
            traffic_split=traffic_split,
            duration_days=duration_days,
        )

        return {
            "success": True,
            "test_id": test_id,
            "message": f"A/B test '{test_name}' created successfully",
        }

    except Exception as e:
        logger.error(f"Error creating A/B test: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Experiment Tracking Endpoints


@router.post("/experiments", response_model=Dict[str, Any])
async def start_experiment(
    request: ExperimentCreateRequest, current_user: dict = Depends(get_current_user)
):
    """
    Start a new experiment run with the provided configuration and metadata.
    
    Parameters:
        request (ExperimentCreateRequest): Contains experiment_name, hyperparameters (config), tags, and notes.
    
    Returns:
        dict: A payload with keys:
            - "success": True when the experiment was started,
            - "run_id": the identifier of the created run,
            - "experiment_name": the name provided in the request,
            - "message": a short status message.
    
    Raises:
        HTTPException: Returned with status 500 if the experiment cannot be started.
    """
    try:
        run_id = await mlops_service.start_experiment(
            experiment_name=request.experiment_name,
            config=request.hyperparameters,
            tags=request.tags,
            notes=request.notes,
        )

        return {
            "success": True,
            "run_id": run_id,
            "experiment_name": request.experiment_name,
            "message": "Experiment started successfully",
        }

    except Exception as e:
        logger.error(f"Error starting experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/metrics", response_model=Dict[str, Any])
async def log_experiment_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    current_user: dict = Depends(get_current_user),
):
    """
    Log numeric metrics for the currently active experiment run.
    
    Parameters:
        metrics (Dict[str, float]): Mapping of metric names to their numeric values.
        step (Optional[int]): Optional step index (e.g., epoch or batch number) associated with these metrics.
    
    Returns:
        dict: A dictionary containing `success` (True on success) and `message` describing the outcome.
    
    Raises:
        HTTPException: Raised with status code 500 if logging fails; the exception detail contains the error message.
    """
    try:
        await mlops_service.log_metrics(metrics, step=step)

        return {"success": True, "message": "Metrics logged successfully"}

    except Exception as e:
        logger.error(f"Error logging metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/experiments/finish", response_model=Dict[str, Any])
async def finish_experiment(
    final_metrics: Optional[Dict[str, Any]] = None,
    current_user: dict = Depends(get_current_user),
):
    """
    Finalize the active experiment and record optional final metrics.
    
    Parameters:
        final_metrics (Optional[Dict[str, Any]]): Final metric values to attach to the experiment; keys are metric names and values are their final measurements.
    
    Returns:
        dict: A payload containing `success` (bool) and `message` (str) describing the result of the operation.
    """
    try:
        await mlops_service.finish_experiment(final_metrics=final_metrics)

        return {"success": True, "message": "Experiment finished successfully"}

    except Exception as e:
        logger.error(f"Error finishing experiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/experiments", response_model=List[Dict[str, Any]])
async def list_experiments(
    experiment_name: Optional[str] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
):
    """
    List experiment runs optionally filtered by name.
    
    Parameters:
        experiment_name (Optional[str]): If provided, only experiments matching this name are returned.
        limit (int): Maximum number of experiments to return.
    
    Returns:
        List[dict]: A list of experiment run records.
    """
    try:
        experiments = await mlops_service.list_experiments(
            experiment_name=experiment_name, limit=limit
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
    current_user: dict = Depends(get_current_user),
):
    """
    Trigger a retraining run for a model using the provided retraining request.
    
    Parameters:
        request (ModelRetrainingRequest): Contains model_name, trigger_reason, and training_config for the retraining run.
        background_tasks (BackgroundTasks): FastAPI background task manager used to schedule the retraining work.
    
    Returns:
        dict: {
            "success": True if the retraining run was scheduled successfully, False otherwise,
            "run_id": identifier of the created retraining run,
            "message": human-readable status message
        }
    """
    try:
        run_id = await mlops_service.trigger_retraining(
            model_name=request.model_name,
            reason=request.trigger_reason,
            config=request.training_config,
        )

        return {
            "success": True,
            "run_id": run_id,
            "message": f"Retraining triggered for {request.model_name}",
        }

    except Exception as e:
        logger.error(f"Error triggering retraining: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retraining/jobs", response_model=List[Dict[str, Any]])
async def list_retraining_jobs(
    model_name: Optional[str] = None,
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
):
    """
    Retrieve retraining jobs optionally filtered by model name.
    
    Parameters:
        model_name (Optional[str]): If provided, only return retraining jobs for this model.
        limit (int): Maximum number of jobs to return.
    
    Returns:
        List[dict]: A list of retraining job records matching the filter and constrained by `limit`.
    """
    try:
        jobs = await mlops_service.retraining_service.list_retraining_jobs(
            model_name=model_name, limit=limit
        )

        return jobs

    except Exception as e:
        logger.error(f"Error listing retraining jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retraining/jobs/{run_id}", response_model=Dict[str, Any])
async def get_retraining_status(
    run_id: str, current_user: dict = Depends(get_current_user)
):
    """
    Return the status of a retraining job identified by `run_id`.
    
    Returns:
        dict: Retraining job status payload.
    
    Raises:
        HTTPException: 404 if the retraining job is not found.
        HTTPException: 500 on internal server error with error details.
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
    model_name: str, current_user: dict = Depends(get_current_user)
):
    """
    Determines whether retraining triggers are active for a model.
    
    Returns:
        dict: A mapping describing retraining trigger statuses for the given model (e.g., which triggers are active and associated metadata).
    """
    try:
        trigger_status = (
            await mlops_service.retraining_service.check_retraining_triggers(model_name)
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
    current_user: dict = Depends(get_current_user),
):
    """
    Return a sentiment prediction for the given text using the specified deployed model.
    
    Returns:
        dict: A payload containing:
            - `success` (bool): `True` on successful prediction.
            - `prediction`: The model's prediction (value or structured object).
            - `model_name` (str): The name of the model used.
            - `model_version` (str | None): The model version used, or `None` if not specified.
    """
    try:
        prediction = await mlops_service.predict_sentiment(
            text=text, model_name=model_name, model_version=model_version
        )

        return {
            "success": True,
            "prediction": prediction,
            "model_name": model_name,
            "model_version": model_version,
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
    current_user: dict = Depends(get_current_user),
):
    """
    Batch-predict sentiment labels for a list of texts using a specified deployed model.
    
    Parameters:
        texts (List[str]): List of input texts to classify; maximum 1000 items.
        model_name (str): Name of the deployed model to use (default "sentiment_base").
        model_version (Optional[str]): Specific model version to use; if omitted, uses the served/default version.
        batch_size (int): Number of texts processed per prediction batch.
    
    Returns:
        dict: A payload containing:
            - success (bool): `True` when predictions were produced.
            - predictions (List[dict]): Model predictions for each input text.
            - total_processed (int): Number of predictions returned.
            - model_name (str): Echoed model_name used for prediction.
            - model_version (Optional[str]): Echoed model_version used for prediction.
    
    Raises:
        HTTPException: With status 400 if more than 1000 texts are provided, or with status 500 for internal prediction errors.
    """
    try:
        if len(texts) > 1000:
            raise HTTPException(
                status_code=400, detail="Maximum 1000 texts allowed per batch"
            )

        predictions = await mlops_service.batch_predict_sentiment(
            texts=texts,
            model_name=model_name,
            model_version=model_version,
            batch_size=batch_size,
        )

        return {
            "success": True,
            "predictions": predictions,
            "total_processed": len(predictions),
            "model_name": model_name,
            "model_version": model_version,
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
    current_user: dict = Depends(get_current_user),
):
    """
    Report model performance metrics to the monitoring subsystem.
    
    Parameters:
        performance_metrics (ModelPerformanceMetrics): Aggregated performance metrics and metadata for the model (e.g., accuracy, latency, throughput, evaluation timestamp).
    
    Returns:
        result (dict): Dictionary containing `success` (`True` if metrics were accepted, `False` otherwise) and a human-readable `message`.
    """
    try:
        await mlops_service.monitor_model_performance(
            model_id=model_id,
            model_version=model_version,
            performance_metrics=performance_metrics,
        )

        return {"success": True, "message": "Performance metrics reported successfully"}

    except Exception as e:
        logger.error(f"Error reporting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Feature Store Endpoints


@router.post("/features/update", response_model=Dict[str, Any])
async def update_features(
    sentiment_results: List[Dict[str, Any]],
    current_user: dict = Depends(get_current_user),
):
    """
    Update the feature store with sentiment analysis results.
    
    Parameters:
        sentiment_results (List[Dict[str, Any]]): List of sentiment result dictionaries. Each dictionary may include keys:
            - "text": analyzed text
            - "sentiment": label string (e.g., "POSITIVE", "NEGATIVE", "NEUTRAL")
            - "sentiment_score": numeric score for sentiment
            - "confidence": numeric confidence value
            - "category": category string (e.g., "general")
            - "source": source identifier (e.g., "user_input")
        current_user (dict, implicitly provided): Authenticated user (injected dependency; not documented here).
    
    Returns:
        Dict[str, Any]: A payload with:
            - "success" (bool): `true` if feature update succeeded, `false` otherwise.
            - "updated_count" (int): Number of records processed and updated when successful; `0` on failure.
            - "message" (str): Human-readable summary of the operation outcome.
    
    Raises:
        HTTPException: Raised with status code 500 if an internal error occurs while updating features.
    """
    try:
        # Convert dict to SentimentResult objects (simplified)
        from ..models.sentiment import (
            SentimentResult,
            SentimentLabel,
            DataSource,
            SentimentCategory,
            AnalysisContext,
        )

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
                processing_time_ms=0.0,
            )
            results.append(result)

        success = await mlops_service.update_features_from_sentiment(results)

        return {
            "success": success,
            "updated_count": len(results) if success else 0,
            "message": (
                "Features updated successfully"
                if success
                else "Failed to update features"
            ),
        }

    except Exception as e:
        logger.error(f"Error updating features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Status and Health Endpoints


@router.get("/status", response_model=Dict[str, Any])
async def get_mlops_status():
    """
    Retrieve the status of all MLOps services.
    
    Returns:
        dict: A mapping of service names to their current status and metadata.
    
    Raises:
        HTTPException: With status code 500 if an internal error occurs while obtaining status.
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
    Perform a health check of MLOps services and return a summarized status payload.
    
    Returns:
        dict: A payload describing health:
            - `status`: `"healthy"` if services report initialized, `"unhealthy"` otherwise.
            - `timestamp`: ISO 8601 UTC timestamp of the check.
            - `services`: mapping of individual service statuses when available.
            - `error` (optional): error message when the health check failed.
    """
    try:
        status = await mlops_service.get_service_status()

        return {
            "status": "healthy" if status.get("initialized", False) else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": status.get("services", {}),
        }

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }
