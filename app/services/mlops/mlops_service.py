"""
Main MLOps service that orchestrates model management, deployment, and monitoring.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from ...models.mlops import (
    ModelMetadata,
    ModelDeployment,
    ExperimentRun,
    ModelPerformanceMetrics,
    ModelRetrainingConfig,
    ModelType,
    ModelStatus,
    DeploymentStrategy,
)
from ...models.sentiment import SentimentResult
from .huggingface_service import HuggingFaceModelService
from .wandb_service import WandBService
from .hopsworks_service import HopsworksService
from .model_deployment_service import ModelDeploymentService
from .model_retraining_service import ModelRetrainingService
from ...core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class MLOpsService:
    """
    Main MLOps service that provides a unified interface for:
    - Model management and versioning
    - Experiment tracking
    - Feature store operations
    - Model deployment and monitoring
    - Automated retraining
    """

    def __init__(self):
        # Initialize component services
        """
        Initialize the MLOpsService by creating and wiring its internal service components.
        
        Creates instances of HuggingFaceModelService, WandBService, HopsworksService, ModelDeploymentService,
        and ModelRetrainingService, and initializes internal status flags (`initialized` set to False and
        an empty `services_status` dictionary).
        """
        self.hf_service = HuggingFaceModelService()
        self.wandb_service = WandBService()
        self.hopsworks_service = HopsworksService()
        self.deployment_service = ModelDeploymentService()
        self.retraining_service = ModelRetrainingService()

        # Service status
        self.initialized = False
        self.services_status = {}

    async def initialize(self):
        """
        Initialize internal MLOps components and populate service status.
        
        Initializes the deployment and retraining services, updates the service-status snapshot using
        cache/info from HuggingFace, Weights & Biases, and Hopsworks, and sets the instance's
        `initialized` flag to True.
        """
        try:
            logger.info("Initializing MLOps services...")

            # Initialize services in order
            await self.deployment_service.initialize()
            await self.retraining_service.initialize()

            # Check service availability
            self.services_status = {
                "huggingface": self.hf_service.get_cache_info(),
                "wandb": self.wandb_service.get_service_status(),
                "hopsworks": self.hopsworks_service.get_service_status(),
                "deployment": {"initialized": True},
                "retraining": {"initialized": True},
            }

            self.initialized = True
            logger.info("MLOps services initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing MLOps services: {e}")
            raise

    # Model Management Methods

    async def register_model(
        self,
        model_name: str,
        model_path: str,
        model_type: ModelType = ModelType.SENTIMENT_ANALYSIS,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> ModelMetadata:
        """
        Register a new model and create a default retraining configuration.
        
        Parameters:
            model_name (str): Human-friendly name for the model.
            model_path (str): Filesystem or storage path to the model artifacts.
            model_type (ModelType): Domain/type of the model (defaults to SENTIMENT_ANALYSIS).
            version (Optional[str]): Optional version identifier; service may generate one if omitted.
            metrics (Optional[Dict[str, float]]): Initial performance metrics to attach to the model.
            description (Optional[str]): Short description of the model.
            tags (Optional[List[str]]): List of tags for categorization and search.
        
        Returns:
            ModelMetadata: Metadata for the newly registered model, including its generated `model_id`.
        """
        try:
            # Register with HuggingFace
            model_metadata = await self.hf_service.register_model(
                model_name=model_name,
                model_path=model_path,
                model_type=model_type,
                metrics=metrics,
                description=description,
                tags=tags,
            )

            # Create retraining configuration
            await self.retraining_service.create_retraining_config(
                model_name=model_name,
                performance_threshold=0.8,
                data_drift_threshold=0.1,
                auto_deploy=False,
                approval_required=True,
            )

            logger.info(f"Registered model: {model_name} ({model_metadata.model_id})")
            return model_metadata

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise

    async def deploy_model(
        self,
        model_id: str,
        model_version: str,
        environment: str = "production",
        strategy: DeploymentStrategy = DeploymentStrategy.IMMEDIATE,
        traffic_percentage: float = 100.0,
    ) -> ModelDeployment:
        """
        Deploys the specified model version to the given environment using the configured deployment strategy.
        
        Parameters:
            model_id: Identifier of the model to deploy.
            model_version: Version of the model to deploy.
            environment: Target environment for deployment (e.g., "production", "staging").
            strategy: Deployment strategy to apply.
            traffic_percentage: Percentage of traffic to route to this deployment (0-100).
        
        Returns:
            ModelDeployment: The created deployment object.
        """
        try:
            # Get model metadata (this would typically come from a model registry)
            model_metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_id.split("/")[-1],
                version=model_version,
                model_type=ModelType.SENTIMENT_ANALYSIS,
                status=ModelStatus.VALIDATING,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by="system",
                framework="transformers",
                model_path=f"./models/{model_id.replace('/', '_')}_v{model_version}",
            )

            # Deploy using deployment service
            deployment = await self.deployment_service.deploy_model(
                model_metadata=model_metadata,
                environment=environment,
                strategy=strategy,
                traffic_percentage=traffic_percentage,
            )

            logger.info(f"Deployed model {model_id} v{model_version} to {environment}")
            return deployment

        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise

    async def create_ab_test(
        self,
        test_name: str,
        model_a_id: str,
        model_a_version: str,
        model_b_id: str,
        model_b_version: str,
        traffic_split: float = 50.0,
        duration_days: int = 7,
    ) -> str:
        """
        Create an A/B test that routes traffic between two deployed models.
        
        Parameters:
            test_name (str): Human-readable name for the A/B test.
            model_a_id (str): Identifier of the control model.
            model_a_version (str): Version of the control model.
            model_b_id (str): Identifier of the treatment model.
            model_b_version (str): Version of the treatment model.
            traffic_split (float): Percentage of incoming traffic routed to model B (0-100).
            duration_days (int): Duration of the A/B test in days.
        
        Returns:
            ab_test_id (str): Identifier of the created A/B test.
        """
        try:
            # Create model metadata for both models
            model_a_metadata = ModelMetadata(
                model_id=model_a_id,
                model_name=model_a_id.split("/")[-1],
                version=model_a_version,
                model_type=ModelType.SENTIMENT_ANALYSIS,
                status=ModelStatus.DEPLOYED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by="system",
                framework="transformers",
                model_path=f"./models/{model_a_id.replace('/', '_')}_v{model_a_version}",
            )

            model_b_metadata = ModelMetadata(
                model_id=model_b_id,
                model_name=model_b_id.split("/")[-1],
                version=model_b_version,
                model_type=ModelType.SENTIMENT_ANALYSIS,
                status=ModelStatus.DEPLOYED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by="system",
                framework="transformers",
                model_path=f"./models/{model_b_id.replace('/', '_')}_v{model_b_version}",
            )

            # Create A/B test
            ab_test = await self.deployment_service.create_ab_test(
                test_name=test_name,
                model_a_metadata=model_a_metadata,
                model_b_metadata=model_b_metadata,
                traffic_split_percentage=traffic_split,
                test_duration_days=duration_days,
            )

            logger.info(f"Created A/B test: {test_name} ({ab_test.test_id})")
            return ab_test.test_id

        except Exception as e:
            logger.error(f"Error creating A/B test: {e}")
            raise

    # Experiment Tracking Methods

    async def start_experiment(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> str:
        """
        Start a new experiment for model training or evaluation

        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration
            tags: Experiment tags
            notes: Experiment notes

        Returns:
            Experiment run ID
        """
        try:
            experiment_run = await self.wandb_service.start_experiment(
                experiment_name=experiment_name, config=config, tags=tags, notes=notes
            )

            logger.info(
                f"Started experiment: {experiment_name} ({experiment_run.run_id})"
            )
            return experiment_run.run_id

        except Exception as e:
            logger.error(f"Error starting experiment: {e}")
            raise

    async def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log provided metrics to the active experiment run.
        
        Parameters:
        	metrics (Dict[str, float]): Mapping of metric names to numeric values.
        	step (Optional[int]): Optional step or iteration number associated with these metrics.
        """
        try:
            await self.wandb_service.log_metrics(metrics, step=step)

        except Exception as e:
            logger.error(f"Error logging metrics: {e}")

    async def finish_experiment(self, final_metrics: Optional[Dict[str, Any]] = None):
        """
        Finish the current experiment and optionally log final metrics.
        
        Parameters:
            final_metrics (Optional[Dict[str, Any]]): Final metrics to record with the experiment before finishing.
        """
        try:
            await self.wandb_service.finish_experiment(final_metrics=final_metrics)

        except Exception as e:
            logger.error(f"Error finishing experiment: {e}")

    # Feature Store Methods

    async def update_features_from_sentiment(
        self, sentiment_results: List[SentimentResult]
    ) -> bool:
        """
        Update the feature store with a list of sentiment analysis results.
        
        Parameters:
            sentiment_results (List[SentimentResult]): Sentiment analysis results to be persisted as features in the feature store.
        
        Returns:
            bool: `True` if the feature store was updated successfully, `False` otherwise.
        """
        try:
            success = (
                await self.hopsworks_service.update_sentiment_features_from_results(
                    sentiment_results
                )
            )

            if success:
                logger.info(
                    f"Updated feature store with {len(sentiment_results)} sentiment results"
                )

            return success

        except Exception as e:
            logger.error(f"Error updating features: {e}")
            return False

    async def get_features_for_training(
        self,
        feature_view_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Optional[Any]:
        """
        Retrieve training features from the feature store for a given feature view.
        
        Parameters:
            feature_view_name (str): Name of the feature view to query.
            start_time (Optional[datetime]): Optional inclusive start time to filter the retrieved records.
            end_time (Optional[datetime]): Optional inclusive end time to filter the retrieved records.
        
        Returns:
            training_data (Any | None): The training dataset retrieved from the feature store, or `None` if retrieval failed.
        """
        try:
            training_data = await self.hopsworks_service.get_training_data(
                feature_view_name=feature_view_name,
                start_time=start_time,
                end_time=end_time,
            )

            if training_data:
                logger.info(
                    f"Retrieved training data from feature view: {feature_view_name}"
                )

            return training_data

        except Exception as e:
            logger.error(f"Error getting training features: {e}")
            return None

    # Model Monitoring and Retraining Methods

    async def monitor_model_performance(
        self,
        model_id: str,
        model_version: str,
        performance_metrics: ModelPerformanceMetrics,
    ):
        """
        Monitor model performance and trigger retraining if needed

        Args:
            model_id: ID of the model
            model_version: Version of the model
            performance_metrics: Current performance metrics
        """
        try:
            # Monitor with deployment service
            deployment_id = f"{model_id}_v{model_version}"  # Simplified mapping
            alerts = await self.deployment_service.monitor_model_performance(
                deployment_id, performance_metrics
            )

            # Monitor with retraining service
            model_name = model_id.split("/")[-1]
            await self.retraining_service.monitor_model_performance(
                model_name, performance_metrics
            )

            if alerts:
                logger.warning(f"Generated {len(alerts)} alerts for model {model_id}")

        except Exception as e:
            logger.error(f"Error monitoring model performance: {e}")

    async def trigger_retraining(
        self, model_name: str, reason: str, config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Trigger a manual retraining run for a model.
        
        Parameters:
            model_name (str): Name of the model to retrain.
            reason (str): Reason for initiating retraining.
            config (Optional[Dict[str, Any]]): Optional training configuration overrides.
        
        Returns:
            str: The retraining run ID.
        """
        try:
            experiment_run = await self.retraining_service.trigger_retraining(
                model_name=model_name, trigger_reason=reason, training_config=config
            )

            logger.info(
                f"Triggered retraining for {model_name}: {experiment_run.run_id}"
            )
            return experiment_run.run_id

        except Exception as e:
            logger.error(f"Error triggering retraining: {e}")
            raise

    # Model Prediction Methods

    async def predict_sentiment(
        self,
        text: str,
        model_name: str = "sentiment_base",
        model_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Predicts sentiment for the given text using the specified model.
        
        Parameters:
            text (str): Text to analyze.
            model_name (str): Model identifier to use for prediction; defaults to "sentiment_base".
            model_version (Optional[str]): Specific model version to use, if any.
        
        Returns:
            Dict[str, Any]: A dictionary containing the sentiment prediction and related metadata.
        """
        try:
            prediction = await self.hf_service.predict_sentiment(
                text=text, model_name=model_name, model_version=model_version
            )

            return prediction

        except Exception as e:
            logger.error(f"Error predicting sentiment: {e}")
            raise

    async def batch_predict_sentiment(
        self,
        texts: List[str],
        model_name: str = "sentiment_base",
        model_version: Optional[str] = None,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """
        Perform sentiment predictions for a list of texts using a specified model.
        
        Parameters:
            texts (List[str]): Input texts to analyze.
            model_name (str): Model identifier to use for predictions.
            model_version (Optional[str]): Specific model version to target, if any.
            batch_size (int): Number of texts processed per inference batch.
        
        Returns:
            List[Dict[str, Any]]: A list of prediction dictionaries corresponding to the input texts.
        """
        try:
            predictions = await self.hf_service.batch_predict_sentiment(
                texts=texts,
                model_name=model_name,
                model_version=model_version,
                batch_size=batch_size,
            )

            return predictions

        except Exception as e:
            logger.error(f"Error in batch sentiment prediction: {e}")
            raise

    # Status and Information Methods

    async def get_service_status(self) -> Dict[str, Any]:
        """
        Return the current status of the MLOps service and its subservices.
        
        When the service is initialized, the `services` entry is augmented with up-to-date status for `huggingface`, `wandb`, and `hopsworks`. The returned dictionary always includes:
        - `initialized` (bool)
        - `services` (dict)
        - `timestamp` (UTC ISO 8601 string)
        
        On failure, returns `{"initialized": False, "error": "<message>"}`.
        
        Returns:
            status (Dict[str, Any]): A dictionary containing the service status as described above.
        """
        try:
            status = {
                "initialized": self.initialized,
                "services": self.services_status,
                "timestamp": datetime.utcnow().isoformat(),
            }

            # Update with current status
            if self.initialized:
                status["services"]["huggingface"] = self.hf_service.get_cache_info()
                status["services"]["wandb"] = self.wandb_service.get_service_status()
                status["services"][
                    "hopsworks"
                ] = self.hopsworks_service.get_service_status()

            return status

        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {"initialized": False, "error": str(e)}

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Lists available sentiment-analysis models from the HuggingFace service.
        
        Returns:
            A list of model metadata dictionaries for available models; an empty list if none or on error.
        """
        try:
            models = await self.hf_service.list_available_models(
                task="sentiment-analysis", limit=50
            )

            return models

        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Retrieve detailed metadata for a model by its identifier.
        
        Parameters:
            model_id (str): Identifier of the model to retrieve.
        
        Returns:
            dict: A dictionary containing model information; an empty dict if retrieval fails.
        """
        try:
            model_info = await self.hf_service.get_model_info(model_id)
            return model_info

        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

    async def list_deployments(
        self, environment: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List active model deployments, optionally filtered by environment.
        
        Parameters:
            environment (Optional[str]): Environment to filter deployments by.
        
        Returns:
            List of deployments represented as dictionaries.
        """
        try:
            deployments = await self.deployment_service.list_active_deployments(
                environment=environment
            )

            return deployments

        except Exception as e:
            logger.error(f"Error listing deployments: {e}")
            return []

    async def list_experiments(
        self, experiment_name: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List experiment runs for a given experiment name.
        
        Parameters:
            experiment_name (Optional[str]): Name of the experiment to filter by. If omitted, returns runs across experiments.
            limit (int): Maximum number of experiment runs to return.
        
        Returns:
            List[Dict[str, Any]]: A list of experiment run records; returns an empty list if no runs are found or an error occurs.
        """
        try:
            experiments = await self.wandb_service.get_experiment_history(
                experiment_name=experiment_name or "", limit=limit
            )

            return experiments

        except Exception as e:
            logger.error(f"Error listing experiments: {e}")
            return []

    # Cleanup Methods

    async def cleanup_resources(self):
        """Clean up MLOps resources"""
        try:
            await self.hf_service.cleanup_cache()
            logger.info("Cleaned up MLOps resources")

        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")


# Global MLOps service instance
mlops_service = MLOpsService()
