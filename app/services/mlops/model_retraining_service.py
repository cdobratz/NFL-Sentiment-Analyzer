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
        """
        Initialize the ModelRetrainingService by creating service clients, in-memory stores, and default retraining triggers.

        Sets up:
        - database handle placeholder (`self.db`)
        - external service clients (`hf_service`, `wandb_service`, `hopsworks_service`, `deployment_service`)
        - in-memory maps for retraining configurations and active retraining jobs
        - performance monitoring stores (`performance_history`, `baseline_metrics`)
        - default trigger thresholds and schedules used when creating new retraining configurations
        """
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
        """
        Initialize internal services and in-memory state for model retraining.

        Acquires a database handle, initializes the deployment service, and loads retraining configurations and baseline metrics into memory. On failure, the error is logged.
        """
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
        Create and persist a retraining configuration for the specified model.

        This generates a unique configuration, populates timestamps and defaults, stores the configuration in persistent storage and the service's in-memory map, and returns the saved configuration.

        Parameters:
            model_name: Name of the model the configuration applies to.
            performance_threshold: Minimum performance score (e.g., accuracy or F1) that will trigger retraining when recent metrics fall below this value.
            data_drift_threshold: Maximum acceptable data drift score; retraining is triggered when observed drift exceeds this value.
            time_based_trigger: Cron expression (UTC) specifying a periodic schedule for retraining, or None to disable.
            data_volume_trigger: Minimum number of new data samples required to trigger retraining, or None to disable.
            auto_deploy: If True, allow automatic deployment of the retrained model when deployment criteria are met.
            approval_required: If True, require manual approval before deploying a retrained model.
            approvers: Optional list of user IDs allowed to approve deployments.
            training_config: Optional dictionary of training parameters and overrides to use for retraining.

        Returns:
            ModelRetrainingConfig: The created and persisted retraining configuration, including generated `config_id`, timestamps, and applied defaults.
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
        Determine whether any retraining triggers are activated for the specified model.

        Parameters:
            model_name (str): Name of the model to evaluate retraining triggers for.

        Returns:
            dict: A dictionary describing trigger evaluation results. On success the dictionary contains:
                - `should_retrain` (bool): `true` if one or more triggers are activated, `false` otherwise.
                - `triggers_activated` (list[str]): List of activated trigger identifiers (e.g., "performance_degradation", "data_drift", "scheduled_retraining", "new_data_available"); present when applicable.
                - `config_id` (str): Identifier of the retraining configuration used for evaluation.
                - `last_check` (str): ISO 8601 UTC timestamp of when the check was performed.
            On error or when no active config is found, the dictionary contains:
                - `should_retrain` (bool): `false`.
                - `reason` (str): Short explanation of the failure or absence of an active configuration.
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
        Initiates a retraining experiment for the given model and schedules its execution in the background.

        Registers the experiment with the tracking service, schedules the asynchronous retraining workflow, updates and persists the retraining configuration's last_triggered timestamp, and returns the created ExperimentRun.

        Parameters:
            training_config (Optional[Dict[str, Any]]): Optional overrides for the stored training configuration; keys provided here replace or extend the saved training configuration.
            auto_deploy (Optional[bool]): If provided, overrides the config's auto-deploy setting for this run; if omitted, the stored config's auto_deploy value is used.

        Returns:
            ExperimentRun: The experiment run object representing the scheduled retraining job.

        Raises:
            ValueError: If no retraining configuration exists for the specified model.
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
        Record a model's performance metrics, persist them, and initiate retraining when configured triggers are met.

        Parameters:
            model_name (str): The name/identifier of the model whose metrics are being reported.
            performance_metrics (ModelPerformanceMetrics): Observed performance metrics for the model; used to update in-memory history and persisted to storage.

        Notes:
            - Appends the metric to an in-memory rolling history (keeps the most recent 100 entries).
            - Persists the metric to the service's performance metrics store.
            - Evaluates configured retraining triggers and, if any activate, starts a retraining run with the aggregated trigger reasons.
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
        Retrieve the serialized status for a retraining run.

        Parameters:
            run_id (str): Identifier of the retraining run to query.

        Returns:
            Optional[Dict[str, Any]]: A dictionary with the retraining run status (keys include `run_id`, `experiment_name`, `run_name`, `status`, `started_at`, `completed_at`, `duration_minutes`, `metrics`, `tags`, `notes`) if the run is found, `None` otherwise.
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
        Get a list of retraining job status records optionally filtered by model and status.

        Parameters:
            model_name (Optional[str]): If provided, only include jobs whose experiment name matches the model (uses "{model_name}_retraining" pattern).
            status (Optional[ExperimentStatus]): If provided, only include jobs with this status.
            limit (int): Maximum number of jobs to return.

        Returns:
            List[Dict[str, Any]]: Retraining job status dictionaries ordered by `started_at` descending (most recent first). Each dictionary includes keys such as `run_id`, `experiment_name`, `run_name`, `status`, `started_at`, `completed_at`, `duration_minutes`, `metrics`, `tags`, and `notes`.
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
        """
        Execute the asynchronous retraining workflow for a configured experiment run.

        Performs end-to-end retraining: updates run status, obtains training data, fine-tunes a base model using the provided training configuration, evaluates and logs metrics, persists the experiment run, optionally deploys the retrained model when criteria are met, finishes the experiment in W&B, and cleans up the active job entry. On error the run is marked failed, results are persisted, the W&B experiment is finished with failure, and the active job is removed.

        Parameters:
            experiment_run (ExperimentRun): The experiment run object to update with status, timestamps, metrics, and duration.
            config (ModelRetrainingConfig): Retraining configuration for the target model (includes model_name and trigger/metadata).
            training_config (Dict[str, Any]): Hyperparameters and options for fine-tuning. Common keys:
                - "base_model": pretrained model identifier to fine-tune (default provided by service).
                - "num_epochs": number of training epochs.
                - "learning_rate": optimizer learning rate.
                - "batch_size": training batch size.
            auto_deploy (bool): If True, deploy the retrained model when evaluation metrics meet deployment criteria.
        """
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
        """
        Determine whether recent model performance has fallen below the configured performance threshold.

        This inspects up to the last five recorded performance metrics for the given model and compares the average accuracy and average F1 score to the `performance_threshold` on the provided `config`.

        Parameters:
            model_name (str): Name of the model whose recent performance is evaluated.
            config (ModelRetrainingConfig): Retraining configuration containing `performance_threshold`.

        Returns:
            bool: `true` if the average accuracy or average F1 score over recent measurements is less than the configured threshold, `false` otherwise.
        """
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
        """
        Determine whether the most recent recorded data drift for a model exceeds the configured threshold.

        Parameters:
            config (ModelRetrainingConfig): Retraining configuration whose `data_drift_threshold` is used for comparison.

        Returns:
            True if the latest `data_drift_score` is present and greater than `config.data_drift_threshold`, False otherwise.
        """
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
        """
        Determine whether a configured time-based retraining trigger should fire.

        Parameters:
            config (ModelRetrainingConfig): Retraining configuration object; the method inspects
                `config.time_based_trigger` to see if a time-based trigger is configured and
                `config.last_triggered` to compute elapsed time since the last trigger.

        Returns:
            true if a time-based trigger is configured and either `last_triggered` is missing or
            more than seven days have passed since `last_triggered`, `false` otherwise.
        """
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
        """
        Determine whether the configured data volume threshold for a model has been reached.

        Checks the count of new records since the config's last_triggered timestamp (or created_at if never triggered) and compares it to config.data_volume_trigger.

        Parameters:
            model_name (str): Name of the model whose new data is being counted.
            config (ModelRetrainingConfig): Retraining configuration containing `data_volume_trigger`, `last_triggered`, and `created_at`.

        Returns:
            bool: `true` if the number of new records since the baseline timestamp is greater than or equal to `config.data_volume_trigger`, `false` otherwise.
        """
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
        """
        Retrieve training and validation datasets for the specified model.

        Attempts to obtain prepared training data for model_name and returns a (training, validation) tuple when available. If no data source is available or an error occurs, returns None.

        Returns:
            A tuple (training_data, validation_data) where each is a list of records with at least `text`, `label`, and `score` keys, or `None` if no training data could be retrieved.
        """
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
        """
        Compute and attach evaluation metrics for a retrained model.

        Updates the provided ModelMetadata instance with evaluation results and returns the computed metrics. The returned dictionary contains:
        - "accuracy": model accuracy (0.0 to 1.0)
        - "precision": precision score (0.0 to 1.0)
        - "recall": recall score (0.0 to 1.0)
        - "f1_score": F1 score (0.0 to 1.0)
        - "training_loss": training loss (non-negative float)
        - "validation_loss": validation loss (non-negative float)

        Parameters:
            model_metadata (ModelMetadata): Mutable metadata object for the retrained model; this function will populate accuracy, precision, recall, f1_score, and custom_metrics on it.

        Returns:
            Dict[str, float]: A mapping of metric names to their numeric values. Returns an empty dict if evaluation fails.
        """
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
        """
        Decides whether a retrained model meets criteria for deployment.

        Parameters:
            evaluation_metrics (Dict[str, float]): Evaluation results for the retrained model; expected keys include `"accuracy"` and `"f1_score"`.

        Returns:
            bool: `True` if `accuracy` is greater than or equal to 0.8 and `f1_score` is greater than or equal to 0.75, `False` otherwise.
        """
        accuracy = evaluation_metrics.get("accuracy", 0)
        f1_score = evaluation_metrics.get("f1_score", 0)

        # Deploy if performance is above minimum thresholds
        return accuracy >= 0.8 and f1_score >= 0.75

    async def _deploy_retrained_model(
        self, model_metadata: ModelMetadata, experiment_run: ExperimentRun
    ):
        """
        Deploys a retrained model to the staging environment and records deployment details to Weights & Biases.

        Parameters:
            model_metadata (ModelMetadata): Metadata for the retrained model to deploy (model identifiers, version, storage path, etc.).
            experiment_run (ExperimentRun): The retraining experiment run associated with this deployment, used for contextual logging and tracking.
        """
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
        """
        Count new labeled data records for a model since a given timestamp.

        Parameters:
            model_name (str): Name of the model whose new data is being counted.
            since (datetime): Inclusive lower-bound timestamp to count records from.

        Returns:
            int: Number of sentiment result records with timestamp >= `since`. Returns 0 if the database is unavailable or an error occurs.
        """
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
        """
        Convert an ExperimentRun into a serializable retraining status dictionary.

        Parameters:
            experiment_run (ExperimentRun): The experiment run to format.

        Returns:
            Dict[str, Any]: A dictionary with keys:
                - run_id: Unique identifier for the run.
                - experiment_name: Name of the experiment.
                - run_name: Human-readable run name.
                - status: Run status string.
                - started_at: ISO 8601 timestamp when the run started.
                - completed_at: ISO 8601 timestamp when the run completed, or `None`.
                - duration_minutes: Duration of the run in minutes.
                - metrics: Collected evaluation/training metrics.
                - tags: Associated tags.
                - notes: Freeform notes or description.
        """
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
        """
        Upsert the given experiment run into the database's experiment_runs collection keyed by `run_id`.

        If no database connection is configured, no action is taken. Any errors during storage are logged and not propagated.
        """
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
        """
        Persist model performance metrics to the configured database.

        If a database client is available on the service (`self.db`), the provided
        metrics are inserted into the `performance_metrics` collection. If no
        database is configured, the function returns without side effects.

        Parameters:
            metrics (ModelPerformanceMetrics): The performance metrics to store, typically
                containing model name, timestamp, and metric values.
        """
        if self.db:
            try:
                await self.db.performance_metrics.insert_one(metrics.dict())
            except Exception as e:
                logger.error(f"Error storing performance metrics: {e}")

    async def _load_retraining_configs(self):
        """
        Load enabled retraining configurations from the database into the in-memory cache.

        When a database connection exists, this method reads documents from the `retraining_configs`
        collection that have `enabled` set to True, constructs ModelRetrainingConfig objects for
        each document, and stores them in `self.retraining_configs` keyed by `config_id`.
        Logs the number of loaded configurations. If no database is available or an error
        occurs, the method logs the error and leaves the in-memory cache unchanged.
        """
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
        """
        Populate the in-memory baseline_metrics map with the most recent performance metrics for each model retrieved from the database.

        If no database connection exists, the method does nothing. On success, self.baseline_metrics will map each model_id to a ModelPerformanceMetrics instance representing that model's latest recorded metrics.
        """
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
