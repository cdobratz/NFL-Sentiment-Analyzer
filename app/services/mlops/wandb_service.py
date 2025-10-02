"""
Weights & Biases integration service for experiment tracking and monitoring.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import asyncio

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from ...models.mlops import (
    ExperimentRun,
    ExperimentStatus,
    ModelMetadata,
    ModelPerformanceMetrics,
    ModelType,
)
from ...core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class WandBService:
    """Service for integrating with Weights & Biases for experiment tracking"""

    def __init__(self):
        """
        Initialize the WandB service configuration, internal state, and optional authentication.

        Sets project_name, entity, api_key, current_run (None), and an empty experiment_cache. If the wandb library is available and an API key is configured, attempts to authenticate with W&B; if wandb is unavailable or authentication fails, the service remains in offline/mock mode and logs an appropriate message.
        """
        self.project_name = getattr(settings, "WANDB_PROJECT", "nfl-sentiment-analyzer")
        self.entity = getattr(settings, "WANDB_ENTITY", None)
        self.api_key = getattr(settings, "WANDB_API_KEY", None)
        self.current_run = None
        self.experiment_cache: Dict[str, ExperimentRun] = {}

        if not WANDB_AVAILABLE:
            logger.warning(
                "Weights & Biases not available. Install with: pip install wandb"
            )
            return

        # Initialize wandb if API key is provided
        if self.api_key:
            try:
                wandb.login(key=self.api_key)
                logger.info("Successfully authenticated with Weights & Biases")
            except Exception as e:
                logger.warning(f"Failed to authenticate with W&B: {e}")
        else:
            logger.info("W&B API key not provided, using offline mode")

    async def start_experiment(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        model_type: ModelType = ModelType.SENTIMENT_ANALYSIS,
    ) -> ExperimentRun:
        """
        Start and register a new experiment run with Weights & Biases (W&B), falling back to a mock run when W&B is unavailable.

        When W&B is available, initializes a run, stores it as the current run, and caches an ExperimentRun record; otherwise creates and returns a mock ExperimentRun.

        Parameters:
            experiment_name (str): Logical name of the experiment/project-specific run group.
            run_name (Optional[str]): Optional human-readable name for this specific run; if omitted, a run name will be used/generated.
            config (Optional[Dict[str, Any]]): Hyperparameters or configuration values to associate with the run.
            tags (Optional[List[str]]): Tags to attach to the run for filtering and grouping.
            notes (Optional[str]): Freeform notes or description for the run.
            model_type (ModelType): Enum value indicating the model or task type for the run.

        Returns:
            ExperimentRun: A record representing the started experiment run (real or mock), including identifiers, status, hyperparameters, timestamps, and metadata.
        """
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available, creating mock experiment")
            return self._create_mock_experiment(
                experiment_name, run_name, config, tags, notes, model_type
            )

        try:
            # Initialize wandb run
            run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                name=run_name,
                config=config or {},
                tags=tags or [],
                notes=notes,
                reinit=True,
            )

            self.current_run = run

            # Create experiment run object
            experiment_run = ExperimentRun(
                experiment_id=f"{self.project_name}_{experiment_name}",
                run_id=run.id,
                experiment_name=experiment_name,
                run_name=run_name or run.name,
                status=ExperimentStatus.RUNNING,
                model_type=model_type,
                hyperparameters=config or {},
                started_at=datetime.utcnow(),
                tags=tags or [],
                notes=notes,
            )

            # Cache the experiment
            self.experiment_cache[run.id] = experiment_run

            logger.info(f"Started W&B experiment: {experiment_name} (run: {run.id})")
            return experiment_run

        except Exception as e:
            logger.error(f"Error starting W&B experiment: {e}")
            # Fallback to mock experiment
            return self._create_mock_experiment(
                experiment_name, run_name, config, tags, notes, model_type
            )

    async def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log a batch of scalar metrics to the active experiment run.

        Parameters:
            metrics (Dict[str, Union[float, int]]): Mapping of metric names to numeric values.
            step (Optional[int]): Optional step or epoch index to associate with these metrics.
            commit (bool): Whether to commit the metrics immediately as a recorded point.
        """
        if not WANDB_AVAILABLE or not self.current_run:
            logger.debug(f"Logging metrics (mock): {metrics}")
            return

        try:
            wandb.log(metrics, step=step, commit=commit)
            logger.debug(f"Logged metrics to W&B: {metrics}")

        except Exception as e:
            logger.error(f"Error logging metrics to W&B: {e}")

    async def log_model_performance(
        self, performance_metrics: ModelPerformanceMetrics, step: Optional[int] = None
    ):
        """
        Log standard and custom model performance metrics to the service.

        Parameters:
            performance_metrics (ModelPerformanceMetrics): Object containing standard metrics (accuracy, precision, recall, f1_score, auc_roc, prediction_count, avg_prediction_time_ms, error_rate, throughput_per_second, data_drift_score, prediction_drift_score) and a dictionary of `custom_metrics` to include.
            step (Optional[int]): Optional step index (for example, training epoch or evaluation step) to associate with the logged metrics.
        """
        metrics_dict = {
            "accuracy": performance_metrics.accuracy,
            "precision": performance_metrics.precision,
            "recall": performance_metrics.recall,
            "f1_score": performance_metrics.f1_score,
            "auc_roc": performance_metrics.auc_roc,
            "prediction_count": performance_metrics.prediction_count,
            "avg_prediction_time_ms": performance_metrics.avg_prediction_time_ms,
            "error_rate": performance_metrics.error_rate,
            "throughput_per_second": performance_metrics.throughput_per_second,
            "data_drift_score": performance_metrics.data_drift_score,
            "prediction_drift_score": performance_metrics.prediction_drift_score,
        }

        # Filter out None values
        metrics_dict = {k: v for k, v in metrics_dict.items() if v is not None}

        # Add custom metrics
        metrics_dict.update(performance_metrics.custom_metrics)

        await self.log_metrics(metrics_dict, step=step)

    async def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Record hyperparameters for the currently active experiment.

        If a W&B run is active, updates the run configuration with the provided hyperparameters; if no active run or W&B is unavailable, the call is a no-op.

        Parameters:
            hyperparameters (Dict[str, Any]): Mapping of hyperparameter names to their values to be recorded with the experiment.
        """
        if not WANDB_AVAILABLE or not self.current_run:
            logger.debug(f"Logging hyperparameters (mock): {hyperparameters}")
            return

        try:
            wandb.config.update(hyperparameters)
            logger.debug(f"Logged hyperparameters to W&B: {hyperparameters}")

        except Exception as e:
            logger.error(f"Error logging hyperparameters to W&B: {e}")

    async def log_artifact(
        self,
        artifact_path: str,
        artifact_name: str,
        artifact_type: str = "model",
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Upload a file or directory as an artifact to Weights & Biases and attach optional descriptive metadata.

        Parameters:
            artifact_path (str): Filesystem path to the artifact; may be a file or a directory. If the path does not exist, the artifact is not uploaded.
            artifact_name (str): Name to register the artifact under in W&B.
            artifact_type (str): Category/type for the artifact (for example, `"model"` or `"dataset"`).
            description (Optional[str]): Optional human-readable description for the artifact.
            metadata (Optional[Dict[str, Any]]): Optional key/value metadata to attach to the artifact.

        Notes:
            If W&B is unavailable or there is no active run, the method performs no upload and returns immediately.
        """
        if not WANDB_AVAILABLE or not self.current_run:
            logger.debug(f"Logging artifact (mock): {artifact_name}")
            return

        try:
            artifact = wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                description=description,
                metadata=metadata or {},
            )

            if os.path.isfile(artifact_path):
                artifact.add_file(artifact_path)
            elif os.path.isdir(artifact_path):
                artifact.add_dir(artifact_path)
            else:
                logger.warning(f"Artifact path does not exist: {artifact_path}")
                return

            wandb.log_artifact(artifact)
            logger.info(f"Logged artifact to W&B: {artifact_name}")

        except Exception as e:
            logger.error(f"Error logging artifact to W&B: {e}")

    async def log_model_metadata(self, model_metadata: ModelMetadata):
        """
        Record model metadata with the service and, when W&B is available, log a tabular representation to Weights & Biases.

        Parameters:
            model_metadata (ModelMetadata): Object containing model identifiers, training metrics, hyperparameters, custom metrics, tags, and description. The metadata is saved via hyperparameters under the "model_metadata" key and, if a W&B run is active, additionally logged as a two-column table ("metric", "value").
        """
        metadata_dict = {
            "model_id": model_metadata.model_id,
            "model_name": model_metadata.model_name,
            "version": model_metadata.version,
            "model_type": model_metadata.model_type.value,
            "framework": model_metadata.framework,
            "accuracy": model_metadata.accuracy,
            "precision": model_metadata.precision,
            "recall": model_metadata.recall,
            "f1_score": model_metadata.f1_score,
            "training_duration_minutes": model_metadata.training_duration_minutes,
            "epochs": model_metadata.epochs,
            "learning_rate": model_metadata.learning_rate,
            "batch_size": model_metadata.batch_size,
            "feature_importance": model_metadata.feature_importance,
            "custom_metrics": model_metadata.custom_metrics,
            "tags": model_metadata.tags,
            "description": model_metadata.description,
        }

        # Log as both config and table
        await self.log_hyperparameters({"model_metadata": metadata_dict})

        if WANDB_AVAILABLE and self.current_run:
            try:
                # Create a table for model metadata
                table = wandb.Table(
                    columns=["metric", "value"],
                    data=[
                        [k, str(v)] for k, v in metadata_dict.items() if v is not None
                    ],
                )
                wandb.log({"model_metadata_table": table})

            except Exception as e:
                logger.error(f"Error logging model metadata table: {e}")

    async def finish_experiment(
        self,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
        final_metrics: Optional[Dict[str, Any]] = None,
    ) -> Optional[ExperimentRun]:
        """
        Finalize the active experiment: record final metrics if provided, update its status and timing, finish the run, and return the updated ExperimentRun.

        Parameters:
            status (ExperimentStatus): Final status to set for the experiment.
            final_metrics (Optional[Dict[str, Any]]): Metrics to log and merge into the experiment's recorded metrics.

        Returns:
            Optional[ExperimentRun]: The updated ExperimentRun from the internal cache if present; otherwise `None`.
        """
        if not self.current_run:
            logger.warning("No active experiment to finish")
            return None

        try:
            run_id = self.current_run.id if WANDB_AVAILABLE else "mock_run"

            # Log final metrics if provided
            if final_metrics:
                await self.log_metrics(final_metrics)

            # Update experiment run
            if run_id in self.experiment_cache:
                experiment_run = self.experiment_cache[run_id]
                experiment_run.status = status
                experiment_run.completed_at = datetime.utcnow()

                if experiment_run.started_at:
                    duration = experiment_run.completed_at - experiment_run.started_at
                    experiment_run.duration_minutes = duration.total_seconds() / 60

                if final_metrics:
                    experiment_run.metrics.update(final_metrics)

            # Finish wandb run
            if WANDB_AVAILABLE and self.current_run:
                wandb.finish()

            self.current_run = None
            logger.info(f"Finished experiment with status: {status.value}")

            return self.experiment_cache.get(run_id)

        except Exception as e:
            logger.error(f"Error finishing experiment: {e}")
            return None

    async def get_experiment_history(
        self, experiment_name: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get history of experiment runs

        Args:
            experiment_name: Name of the experiment
            limit: Maximum number of runs to return

        Returns:
            List of experiment run data
        """
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available, returning empty history")
            return []

        try:
            api = wandb.Api()
            runs = api.runs(
                path=(
                    f"{self.entity}/{self.project_name}"
                    if self.entity
                    else self.project_name
                ),
                filters={"display_name": {"$regex": experiment_name}},
                per_page=limit,
            )

            history = []
            for run in runs:
                run_data = {
                    "id": run.id,
                    "name": run.name,
                    "state": run.state,
                    "created_at": run.created_at,
                    "updated_at": run.updated_at,
                    "config": dict(run.config),
                    "summary": dict(run.summary),
                    "tags": run.tags,
                    "notes": run.notes,
                    "url": run.url,
                }
                history.append(run_data)

            return history

        except Exception as e:
            logger.error(f"Error getting experiment history: {e}")
            return []

    async def compare_experiments(
        self, run_ids: List[str], metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate run details and selected metric values for a set of W&B runs.

        Retrieves each run's id, name, state, summary, and config and collects the requested metric values across runs.
        Runs that cannot be fetched are skipped and logged.

        Parameters:
            run_ids (List[str]): List of W&B run IDs to include in the comparison.
            metrics (Optional[List[str]]): Specific metric names to collect for each run; if omitted, metric comparison will be empty.

        Returns:
            Dict[str, Any]: A dictionary with keys:
                - "runs": list of run summaries (each with `id`, `name`, `state`, `summary`, `config`),
                - "metrics_comparison": mapping from metric name to a mapping of run_id to the metric value,
                - "config_comparison": mapping intended for config-level comparisons (may be empty).
        """
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available, returning empty comparison")
            return {}

        try:
            api = wandb.Api()
            comparison_data = {
                "runs": [],
                "metrics_comparison": {},
                "config_comparison": {},
            }

            for run_id in run_ids:
                try:
                    run = api.run(
                        f"{self.entity}/{self.project_name}/{run_id}"
                        if self.entity
                        else f"{self.project_name}/{run_id}"
                    )

                    run_data = {
                        "id": run.id,
                        "name": run.name,
                        "state": run.state,
                        "summary": dict(run.summary),
                        "config": dict(run.config),
                    }
                    comparison_data["runs"].append(run_data)

                    # Compare specific metrics
                    if metrics:
                        for metric in metrics:
                            if metric not in comparison_data["metrics_comparison"]:
                                comparison_data["metrics_comparison"][metric] = {}
                            comparison_data["metrics_comparison"][metric][run_id] = (
                                run.summary.get(metric)
                            )

                except Exception as e:
                    logger.warning(f"Could not fetch run {run_id}: {e}")

            return comparison_data

        except Exception as e:
            logger.error(f"Error comparing experiments: {e}")
            return {}

    def _create_mock_experiment(
        self,
        experiment_name: str,
        run_name: Optional[str],
        config: Optional[Dict[str, Any]],
        tags: Optional[List[str]],
        notes: Optional[str],
        model_type: ModelType,
    ) -> ExperimentRun:
        """
        Create and cache a mock ExperimentRun used when W&B is unavailable.

        Returns:
            ExperimentRun: A mock run with a generated UUID run_id, status set to RUNNING, started_at set to the current UTC time, and stored in the service's experiment_cache.
        """
        import uuid

        run_id = str(uuid.uuid4())
        experiment_run = ExperimentRun(
            experiment_id=f"mock_{experiment_name}",
            run_id=run_id,
            experiment_name=experiment_name,
            run_name=run_name or f"mock_run_{run_id[:8]}",
            status=ExperimentStatus.RUNNING,
            model_type=model_type,
            hyperparameters=config or {},
            started_at=datetime.utcnow(),
            tags=tags or [],
            notes=notes,
        )

        self.experiment_cache[run_id] = experiment_run
        return experiment_run

    async def create_sweep(
        self, sweep_config: Dict[str, Any], project: Optional[str] = None
    ) -> str:
        """
        Create a hyperparameter sweep in Weights & Biases and return its sweep identifier.

        Parameters:
            sweep_config (Dict[str, Any]): Configuration dictionary for the sweep following W&B's sweep schema.
            project (Optional[str]): Project name to create the sweep in; if omitted, the service's default project is used.

        Returns:
            str: The sweep ID returned by W&B. Returns "mock_sweep_id" when W&B is unavailable and "error_sweep_id" if sweep creation fails.
        """
        if not WANDB_AVAILABLE:
            logger.warning("W&B not available, returning mock sweep ID")
            return "mock_sweep_id"

        try:
            sweep_id = wandb.sweep(
                sweep=sweep_config,
                project=project or self.project_name,
                entity=self.entity,
            )

            logger.info(f"Created W&B sweep: {sweep_id}")
            return sweep_id

        except Exception as e:
            logger.error(f"Error creating sweep: {e}")
            return "error_sweep_id"

    def get_service_status(self) -> Dict[str, Any]:
        """
        Return a snapshot of the WandB service's current status.

        Returns:
            status (Dict[str, Any]): A dictionary with the following keys:
                - "wandb_available" (bool): Whether the wandb library is available.
                - "project_name" (Optional[str]): Configured project name.
                - "entity" (Optional[str]): Configured entity/organization.
                - "authenticated" (bool): Whether an API key is configured.
                - "current_run_active" (bool): Whether there is an active run.
                - "cached_experiments" (int): Number of experiment records currently cached.
        """
        return {
            "wandb_available": WANDB_AVAILABLE,
            "project_name": self.project_name,
            "entity": self.entity,
            "authenticated": bool(self.api_key),
            "current_run_active": self.current_run is not None,
            "cached_experiments": len(self.experiment_cache),
        }
