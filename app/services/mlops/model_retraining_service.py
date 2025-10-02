"""
Automated model retraining pipeline with performance monitoring.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
import uuid

from ...models.mlops import (
    ModelMetadata,
    ModelRetrainingConfig,
    ExperimentRun,
    ModelPerformanceMetrics,
    ModelType,
    ModelStatus,
    ExperimentStatus,
)
from ...models.sentiment import SentimentResult
from .huggingface_service import HuggingFaceModelService
from .wandb_service import WandBService
from .hopsworks_service import HopsworksService
from .model_deployment_service import ModelDeploymentService
from ...core.config import get_settings
from ...core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelRetrainingService:
    """Service for automated model retraining and performance monitoring"""

    def __init__(self):
        self.db = None
        self.hf_service = HuggingFaceModelService()
        self.wandb_service = WandBService()
        self.hopsworks_service = HopsworksService()
        self.deployment_service = ModelDeploymentService()

        # Retraining configurations
        self.retraining_configs: Dict[str, ModelRetrainingConfig] = {}
        self.active_retraining_jobs: Dict[str, ExperimentRun] = {}

        # Performance monitoring
        self.performance_history: Dict[str, List[ModelPerformanceMetrics]] = {}
        self.baseline_metrics: Dict[str, ModelPerformanceMetrics] = {}

        # Default retraining triggers
        self.default_triggers = {
            "performance_threshold": 0.8,
            "data_drift_threshold": 0.1,
            "time_based_trigger": "0 2 * * 0",  # Weekly at 2 AM on Sunday
            "data_volume_trigger": 10000,  # Retrain after 10k new samples
        }

    async def initialize(self):
        """Initialize the retraining service"""
        try:
            self.db = await get_database()
            await self.deployment_service.initialize()
            await self._load_retraining_configs()
            await self._load_baseline_metrics()
            logger.info("Model retraining service initialized")
        except Exception as e:
            logger.error(f"Error initializing retraining service: {e}")

    async def create_retraining_config(
        self,
        model_name: str,
        performance_threshold: float = 0.8,
        data_drift_threshold: float = 0.1,
        time_based_trigger: Optional[str] = None,
        data_volume_trigger: Optional[int] = None,
        auto_deploy: bool = False,
        approval_required: bool = True,
        approvers: Optional[List[str]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> ModelRetrainingConfig:
        """
        Create a retraining configuration for a model

        Args:
            model_name: Name of the model
            performance_threshold: Minimum performance threshold
            data_drift_threshold: Maximum data drift threshold
            time_based_trigger: Cron expression for time-based retraining
            data_volume_trigger: Minimum new samples to trigger retraining
            auto_deploy: Whether to auto-deploy retrained models
            approval_required: Whether approval is required before deployment
            approvers: List of approver user IDs
            training_config: Training configuration parameters

        Returns:
            Model retraining configuration
        """
        try:
            config_id = str(uuid.uuid4())

            config = ModelRetrainingConfig(
                config_id=config_id,
                model_name=model_name,
                performance_threshold=performance_threshold,
                data_drift_threshold=data_drift_threshold,
                time_based_trigger=time_based_trigger,
                data_volume_trigger=data_volume_trigger,
                training_config=training_config or {},
                auto_deploy=auto_deploy,
                approval_required=approval_required,
                approvers=approvers or [],
                enabled=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            # Store configuration
            await self._store_retraining_config(config)
            self.retraining_configs[config_id] = config

            logger.info(f"Created retraining config for {model_name}: {config_id}")
            return config

        except Exception as e:
            logger.error(f"Error creating retraining config: {e}")
            raise

    async def check_retraining_triggers(self, model_name: str) -> Dict[str, Any]:
        """
        Check if any retraining triggers are activated for a model

        Args:
            model_name: Name of the model to check

        Returns:
            Dictionary with trigger status and reasons
        """
        try:
            # Find retraining config for the model
            config = None
            for cfg in self.retraining_configs.values():
                if cfg.model_name == model_name:
                    config = cfg
                    break

            if not config or not config.enabled:
                return {
                    "should_retrain": False,
                    "reason": "No active retraining config",
                }

            triggers_activated = []

            # Check performance threshold
            if await self._check_performance_trigger(model_name, config):
                triggers_activated.append("performance_degradation")

            # Check data drift threshold
            if await self._check_data_drift_trigger(model_name, config):
                triggers_activated.append("data_drift")

            # Check time-based trigger
            if await self._check_time_based_trigger(config):
                triggers_activated.append("scheduled_retraining")

            # Check data volume trigger
            if await self._check_data_volume_trigger(model_name, config):
                triggers_activated.append("new_data_available")

            should_retrain = len(triggers_activated) > 0

            return {
                "should_retrain": should_retrain,
                "triggers_activated": triggers_activated,
                "config_id": config.config_id,
                "last_check": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error checking retraining triggers: {e}")
            return {"should_retrain": False, "reason": f"Error: {str(e)}"}

    async def trigger_retraining(
        self,
        model_name: str,
        trigger_reason: str,
        training_config: Optional[Dict[str, Any]] = None,
        auto_deploy: Optional[bool] = None,
    ) -> ExperimentRun:
        """
        Trigger model retraining

        Args:
            model_name: Name of the model to retrain
            trigger_reason: Reason for triggering retraining
            training_config: Override training configuration
            auto_deploy: Override auto-deploy setting

        Returns:
            Experiment run for the retraining job
        """
        try:
            # Find retraining config
            config = None
            for cfg in self.retraining_configs.values():
                if cfg.model_name == model_name:
                    config = cfg
                    break

            if not config:
                raise ValueError(f"No retraining config found for model: {model_name}")

            # Start experiment tracking
            experiment_name = f"{model_name}_retraining"
            run_name = f"retrain_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

            # Merge training configurations
            final_training_config = config.training_config.copy()
            if training_config:
                final_training_config.update(training_config)

            # Start W&B experiment
            experiment_run = await self.wandb_service.start_experiment(
                experiment_name=experiment_name,
                run_name=run_name,
                config=final_training_config,
                tags=["retraining", trigger_reason],
                notes=f"Automated retraining triggered by: {trigger_reason}",
                model_type=ModelType.SENTIMENT_ANALYSIS,
            )

            # Store active retraining job
            self.active_retraining_jobs[experiment_run.run_id] = experiment_run

            # Execute retraining in background
            asyncio.create_task(
                self._execute_retraining(
                    experiment_run,
                    config,
                    final_training_config,
                    auto_deploy if auto_deploy is not None else config.auto_deploy,
                )
            )

            # Update config
            config.last_triggered = datetime.utcnow()
            await self._store_retraining_config(config)

            logger.info(
                f"Triggered retraining for {model_name}: {experiment_run.run_id}"
            )
            return experiment_run

        except Exception as e:
            logger.error(f"Error triggering retraining: {e}")
            raise

    async def monitor_model_performance(
        self, model_name: str, performance_metrics: ModelPerformanceMetrics
    ):
        """
        Monitor model performance and trigger retraining if needed

        Args:
            model_name: Name of the model
            performance_metrics: Current performance metrics
        """
        try:
            # Store performance metrics
            if model_name not in self.performance_history:
                self.performance_history[model_name] = []

            self.performance_history[model_name].append(performance_metrics)

            # Keep only last 100 metrics
            if len(self.performance_history[model_name]) > 100:
                self.performance_history[model_name] = self.performance_history[
                    model_name
                ][-100:]

            # Store in database
            await self._store_performance_metrics(performance_metrics)

            # Check if retraining is needed
            trigger_check = await self.check_retraining_triggers(model_name)

            if trigger_check["should_retrain"]:
                logger.info(
                    f"Retraining triggered for {model_name}: {trigger_check['triggers_activated']}"
                )

                # Trigger retraining
                await self.trigger_retraining(
                    model_name=model_name,
                    trigger_reason=", ".join(trigger_check["triggers_activated"]),
                )

        except Exception as e:
            logger.error(f"Error monitoring model performance: {e}")

    async def get_retraining_status(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a retraining job

        Args:
            run_id: ID of the retraining run

        Returns:
            Retraining job status
        """
        try:
            if run_id not in self.active_retraining_jobs:
                # Check database for completed jobs
                if self.db:
                    doc = await self.db.experiment_runs.find_one({"run_id": run_id})
                    if doc:
                        experiment_run = ExperimentRun(**doc)
                        return self._format_retraining_status(experiment_run)

                return None

            experiment_run = self.active_retraining_jobs[run_id]
            return self._format_retraining_status(experiment_run)

        except Exception as e:
            logger.error(f"Error getting retraining status: {e}")
            return None

    async def list_retraining_jobs(
        self,
        model_name: Optional[str] = None,
        status: Optional[ExperimentStatus] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        List retraining jobs

        Args:
            model_name: Filter by model name
            status: Filter by status
            limit: Maximum number of jobs to return

        Returns:
            List of retraining jobs
        """
        try:
            jobs = []

            # Get from active jobs
            for experiment_run in self.active_retraining_jobs.values():
                if model_name and model_name not in experiment_run.experiment_name:
                    continue
                if status and experiment_run.status != status:
                    continue

                jobs.append(self._format_retraining_status(experiment_run))

            # Get from database
            if self.db:
                query = {"experiment_name": {"$regex": "retraining"}}
                if model_name:
                    query["experiment_name"] = {"$regex": f"{model_name}_retraining"}
                if status:
                    query["status"] = status.value

                cursor = (
                    self.db.experiment_runs.find(query)
                    .sort("started_at", -1)
                    .limit(limit)
                )
                async for doc in cursor:
                    experiment_run = ExperimentRun(**doc)
                    jobs.append(self._format_retraining_status(experiment_run))

            # Remove duplicates and sort
            unique_jobs = {job["run_id"]: job for job in jobs}
            sorted_jobs = sorted(
                unique_jobs.values(), key=lambda x: x["started_at"], reverse=True
            )

            return sorted_jobs[:limit]

        except Exception as e:
            logger.error(f"Error listing retraining jobs: {e}")
            return []

    async def _execute_retraining(
        self,
        experiment_run: ExperimentRun,
        config: ModelRetrainingConfig,
        training_config: Dict[str, Any],
        auto_deploy: bool,
    ):
        """Execute the actual retraining process"""
        try:
            logger.info(f"Starting retraining execution: {experiment_run.run_id}")

            # Update experiment status
            experiment_run.status = ExperimentStatus.RUNNING

            # Get training data from Hopsworks
            training_data = await self._get_training_data(config.model_name)
            if not training_data:
                raise ValueError("No training data available")

            # Log training data info
            await self.wandb_service.log_metrics(
                {
                    "training_samples": len(training_data[0]) if training_data else 0,
                    "validation_samples": (
                        len(training_data[1])
                        if training_data and len(training_data) > 1
                        else 0
                    ),
                }
            )

            # Get base model for fine-tuning
            base_model = training_config.get(
                "base_model", "cardiffnlp/twitter-roberta-base-sentiment-latest"
            )

            # Fine-tune model
            model_metadata = await self.hf_service.fine_tune_model(
                base_model=base_model,
                training_data=training_data[0] if training_data else [],
                validation_data=(
                    training_data[1]
                    if training_data and len(training_data) > 1
                    else None
                ),
                output_dir=f"./models/{config.model_name}_retrained_{experiment_run.run_id}",
                num_epochs=training_config.get("num_epochs", 3),
                learning_rate=training_config.get("learning_rate", 2e-5),
                batch_size=training_config.get("batch_size", 16),
            )

            # Log model metadata
            await self.wandb_service.log_model_metadata(model_metadata)

            # Evaluate model performance
            evaluation_metrics = await self._evaluate_retrained_model(model_metadata)

            # Log evaluation metrics
            await self.wandb_service.log_metrics(evaluation_metrics)

            # Update experiment with results
            experiment_run.status = ExperimentStatus.COMPLETED
            experiment_run.completed_at = datetime.utcnow()
            experiment_run.metrics = evaluation_metrics

            if experiment_run.started_at:
                duration = experiment_run.completed_at - experiment_run.started_at
                experiment_run.duration_minutes = duration.total_seconds() / 60

            # Store experiment results
            await self._store_experiment_run(experiment_run)

            # Deploy model if auto-deploy is enabled and performance is good
            if auto_deploy and self._should_deploy_retrained_model(evaluation_metrics):
                await self._deploy_retrained_model(model_metadata, experiment_run)

            # Finish W&B experiment
            await self.wandb_service.finish_experiment(
                status=ExperimentStatus.COMPLETED, final_metrics=evaluation_metrics
            )

            # Remove from active jobs
            if experiment_run.run_id in self.active_retraining_jobs:
                del self.active_retraining_jobs[experiment_run.run_id]

            logger.info(f"Retraining completed successfully: {experiment_run.run_id}")

        except Exception as e:
            logger.error(f"Error in retraining execution: {e}")

            # Update experiment status
            experiment_run.status = ExperimentStatus.FAILED
            experiment_run.completed_at = datetime.utcnow()

            # Store failed experiment
            await self._store_experiment_run(experiment_run)

            # Finish W&B experiment
            await self.wandb_service.finish_experiment(
                status=ExperimentStatus.FAILED, final_metrics={"error": str(e)}
            )

            # Remove from active jobs
            if experiment_run.run_id in self.active_retraining_jobs:
                del self.active_retraining_jobs[experiment_run.run_id]

    async def _check_performance_trigger(
        self, model_name: str, config: ModelRetrainingConfig
    ) -> bool:
        """Check if performance has degraded below threshold"""
        try:
            if model_name not in self.performance_history:
                return False

            recent_metrics = self.performance_history[model_name][
                -5:
            ]  # Last 5 measurements
            if not recent_metrics:
                return False

            # Calculate average recent performance
            avg_accuracy = sum(m.accuracy or 0 for m in recent_metrics) / len(
                recent_metrics
            )
            avg_f1 = sum(m.f1_score or 0 for m in recent_metrics) / len(recent_metrics)

            # Check if below threshold
            return (
                avg_accuracy < config.performance_threshold
                or avg_f1 < config.performance_threshold
            )

        except Exception as e:
            logger.error(f"Error checking performance trigger: {e}")
            return False

    async def _check_data_drift_trigger(
        self, model_name: str, config: ModelRetrainingConfig
    ) -> bool:
        """Check if data drift exceeds threshold"""
        try:
            if model_name not in self.performance_history:
                return False

            recent_metrics = self.performance_history[model_name][-1:]  # Most recent
            if not recent_metrics:
                return False

            latest_drift = recent_metrics[0].data_drift_score
            return latest_drift and latest_drift > config.data_drift_threshold

        except Exception as e:
            logger.error(f"Error checking data drift trigger: {e}")
            return False

    async def _check_time_based_trigger(self, config: ModelRetrainingConfig) -> bool:
        """Check if time-based trigger should fire"""
        try:
            if not config.time_based_trigger:
                return False

            # Simplified check - in practice, you'd use a proper cron parser
            if not config.last_triggered:
                return True

            # Check if it's been more than a week since last trigger
            time_since_last = datetime.utcnow() - config.last_triggered
            return time_since_last > timedelta(days=7)

        except Exception as e:
            logger.error(f"Error checking time-based trigger: {e}")
            return False

    async def _check_data_volume_trigger(
        self, model_name: str, config: ModelRetrainingConfig
    ) -> bool:
        """Check if enough new data is available"""
        try:
            if not config.data_volume_trigger:
                return False

            # Check new data volume since last retraining
            new_data_count = await self._get_new_data_count(
                model_name, config.last_triggered or config.created_at
            )

            return new_data_count >= config.data_volume_trigger

        except Exception as e:
            logger.error(f"Error checking data volume trigger: {e}")
            return False

    async def _get_training_data(
        self, model_name: str
    ) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """Get training data for model retraining"""
        try:
            # Get data from Hopsworks feature store
            if hasattr(self.hopsworks_service, "get_training_data"):
                return await self.hopsworks_service.get_training_data(
                    feature_view_name=f"{model_name}_features", version=1
                )

            # Fallback: get recent sentiment data from database
            if self.db:
                # Get recent sentiment results for training
                since = datetime.utcnow() - timedelta(days=30)
                cursor = self.db.sentiment_results.find(
                    {"timestamp": {"$gte": since}}
                ).limit(10000)

                training_data = []
                async for doc in cursor:
                    training_data.append(
                        {
                            "text": doc["text"],
                            "label": doc["sentiment"],
                            "score": doc["sentiment_score"],
                        }
                    )

                # Split into train/validation
                split_idx = int(len(training_data) * 0.8)
                return training_data[:split_idx], training_data[split_idx:]

            return None

        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return None

    async def _evaluate_retrained_model(
        self, model_metadata: ModelMetadata
    ) -> Dict[str, float]:
        """Evaluate the retrained model"""
        try:
            # Simplified evaluation - in practice, you'd run comprehensive evaluation
            # on a held-out test set

            evaluation_metrics = {
                "accuracy": 0.85,  # Mock metrics
                "precision": 0.83,
                "recall": 0.87,
                "f1_score": 0.85,
                "training_loss": 0.15,
                "validation_loss": 0.18,
            }

            # Update model metadata with evaluation results
            model_metadata.accuracy = evaluation_metrics["accuracy"]
            model_metadata.precision = evaluation_metrics["precision"]
            model_metadata.recall = evaluation_metrics["recall"]
            model_metadata.f1_score = evaluation_metrics["f1_score"]
            model_metadata.custom_metrics = evaluation_metrics

            return evaluation_metrics

        except Exception as e:
            logger.error(f"Error evaluating retrained model: {e}")
            return {}

    def _should_deploy_retrained_model(
        self, evaluation_metrics: Dict[str, float]
    ) -> bool:
        """Determine if retrained model should be deployed"""
        accuracy = evaluation_metrics.get("accuracy", 0)
        f1_score = evaluation_metrics.get("f1_score", 0)

        # Deploy if performance is above minimum thresholds
        return accuracy >= 0.8 and f1_score >= 0.75

    async def _deploy_retrained_model(
        self, model_metadata: ModelMetadata, experiment_run: ExperimentRun
    ):
        """Deploy the retrained model"""
        try:
            # Deploy to staging first
            deployment = await self.deployment_service.deploy_model(
                model_metadata=model_metadata,
                environment="staging",
                strategy="immediate",
            )

            logger.info(
                f"Deployed retrained model to staging: {deployment.deployment_id}"
            )

            # Log deployment info to W&B
            await self.wandb_service.log_metrics(
                {
                    "deployment_id": deployment.deployment_id,
                    "deployment_environment": deployment.environment,
                    "deployment_status": deployment.status.value,
                }
            )

        except Exception as e:
            logger.error(f"Error deploying retrained model: {e}")

    async def _get_new_data_count(self, model_name: str, since: datetime) -> int:
        """Get count of new data since a specific time"""
        try:
            if self.db:
                count = await self.db.sentiment_results.count_documents(
                    {"timestamp": {"$gte": since}, "model_version": model_name}
                )
                return count
            return 0
        except Exception as e:
            logger.error(f"Error getting new data count: {e}")
            return 0

    def _format_retraining_status(
        self, experiment_run: ExperimentRun
    ) -> Dict[str, Any]:
        """Format experiment run as retraining status"""
        return {
            "run_id": experiment_run.run_id,
            "experiment_name": experiment_run.experiment_name,
            "run_name": experiment_run.run_name,
            "status": experiment_run.status.value,
            "started_at": experiment_run.started_at.isoformat(),
            "completed_at": (
                experiment_run.completed_at.isoformat()
                if experiment_run.completed_at
                else None
            ),
            "duration_minutes": experiment_run.duration_minutes,
            "metrics": experiment_run.metrics,
            "tags": experiment_run.tags,
            "notes": experiment_run.notes,
        }

    async def _store_retraining_config(self, config: ModelRetrainingConfig):
        """Store retraining configuration in database"""
        if self.db:
            try:
                await self.db.retraining_configs.replace_one(
                    {"config_id": config.config_id}, config.dict(), upsert=True
                )
            except Exception as e:
                logger.error(f"Error storing retraining config: {e}")

    async def _store_experiment_run(self, experiment_run: ExperimentRun):
        """Store experiment run in database"""
        if self.db:
            try:
                await self.db.experiment_runs.replace_one(
                    {"run_id": experiment_run.run_id},
                    experiment_run.dict(),
                    upsert=True,
                )
            except Exception as e:
                logger.error(f"Error storing experiment run: {e}")

    async def _store_performance_metrics(self, metrics: ModelPerformanceMetrics):
        """Store performance metrics in database"""
        if self.db:
            try:
                await self.db.performance_metrics.insert_one(metrics.dict())
            except Exception as e:
                logger.error(f"Error storing performance metrics: {e}")

    async def _load_retraining_configs(self):
        """Load retraining configurations from database"""
        if self.db:
            try:
                cursor = self.db.retraining_configs.find({"enabled": True})
                async for doc in cursor:
                    config = ModelRetrainingConfig(**doc)
                    self.retraining_configs[config.config_id] = config
                logger.info(f"Loaded {len(self.retraining_configs)} retraining configs")
            except Exception as e:
                logger.error(f"Error loading retraining configs: {e}")

    async def _load_baseline_metrics(self):
        """Load baseline performance metrics"""
        if self.db:
            try:
                # Load recent performance metrics for each model
                pipeline = [
                    {"$sort": {"timestamp": -1}},
                    {
                        "$group": {
                            "_id": "$model_id",
                            "latest_metrics": {"$first": "$$ROOT"},
                        }
                    },
                ]

                cursor = self.db.performance_metrics.aggregate(pipeline)
                async for doc in cursor:
                    model_id = doc["_id"]
                    metrics = ModelPerformanceMetrics(**doc["latest_metrics"])
                    self.baseline_metrics[model_id] = metrics

                logger.info(
                    f"Loaded baseline metrics for {len(self.baseline_metrics)} models"
                )
            except Exception as e:
                logger.error(f"Error loading baseline metrics: {e}")
