"""
Main MLOps service that orchestrates model management, deployment, and monitoring.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio

from ...models.mlops import (
    ModelMetadata, ModelDeployment, ExperimentRun, 
    ModelPerformanceMetrics, ModelRetrainingConfig,
    ModelType, ModelStatus, DeploymentStrategy
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
        self.hf_service = HuggingFaceModelService()
        self.wandb_service = WandBService()
        self.hopsworks_service = HopsworksService()
        self.deployment_service = ModelDeploymentService()
        self.retraining_service = ModelRetrainingService()
        
        # Service status
        self.initialized = False
        self.services_status = {}
    
    async def initialize(self):
        """Initialize all MLOps services"""
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
                "retraining": {"initialized": True}
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
        tags: Optional[List[str]] = None
    ) -> ModelMetadata:
        """
        Register a new model in the MLOps pipeline
        
        Args:
            model_name: Name of the model
            model_path: Path to the model files
            model_type: Type of model
            version: Model version (auto-generated if not provided)
            metrics: Performance metrics
            description: Model description
            tags: Model tags
            
        Returns:
            Model metadata
        """
        try:
            # Register with HuggingFace
            model_metadata = await self.hf_service.register_model(
                model_name=model_name,
                model_path=model_path,
                model_type=model_type,
                metrics=metrics,
                description=description,
                tags=tags
            )
            
            # Create retraining configuration
            await self.retraining_service.create_retraining_config(
                model_name=model_name,
                performance_threshold=0.8,
                data_drift_threshold=0.1,
                auto_deploy=False,
                approval_required=True
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
        traffic_percentage: float = 100.0
    ) -> ModelDeployment:
        """
        Deploy a model to the specified environment
        
        Args:
            model_id: ID of the model to deploy
            model_version: Version of the model
            environment: Target environment
            strategy: Deployment strategy
            traffic_percentage: Traffic percentage for the deployment
            
        Returns:
            Model deployment object
        """
        try:
            # Get model metadata (this would typically come from a model registry)
            model_metadata = ModelMetadata(
                model_id=model_id,
                model_name=model_id.split('/')[-1],
                version=model_version,
                model_type=ModelType.SENTIMENT_ANALYSIS,
                status=ModelStatus.VALIDATING,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by="system",
                framework="transformers",
                model_path=f"./models/{model_id.replace('/', '_')}_v{model_version}"
            )
            
            # Deploy using deployment service
            deployment = await self.deployment_service.deploy_model(
                model_metadata=model_metadata,
                environment=environment,
                strategy=strategy,
                traffic_percentage=traffic_percentage
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
        duration_days: int = 7
    ) -> str:
        """
        Create an A/B test between two models
        
        Args:
            test_name: Name of the A/B test
            model_a_id: ID of model A (control)
            model_a_version: Version of model A
            model_b_id: ID of model B (treatment)
            model_b_version: Version of model B
            traffic_split: Percentage of traffic for model B
            duration_days: Duration of the test
            
        Returns:
            A/B test ID
        """
        try:
            # Create model metadata for both models
            model_a_metadata = ModelMetadata(
                model_id=model_a_id,
                model_name=model_a_id.split('/')[-1],
                version=model_a_version,
                model_type=ModelType.SENTIMENT_ANALYSIS,
                status=ModelStatus.DEPLOYED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by="system",
                framework="transformers",
                model_path=f"./models/{model_a_id.replace('/', '_')}_v{model_a_version}"
            )
            
            model_b_metadata = ModelMetadata(
                model_id=model_b_id,
                model_name=model_b_id.split('/')[-1],
                version=model_b_version,
                model_type=ModelType.SENTIMENT_ANALYSIS,
                status=ModelStatus.DEPLOYED,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by="system",
                framework="transformers",
                model_path=f"./models/{model_b_id.replace('/', '_')}_v{model_b_version}"
            )
            
            # Create A/B test
            ab_test = await self.deployment_service.create_ab_test(
                test_name=test_name,
                model_a_metadata=model_a_metadata,
                model_b_metadata=model_b_metadata,
                traffic_split_percentage=traffic_split,
                test_duration_days=duration_days
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
        notes: Optional[str] = None
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
                experiment_name=experiment_name,
                config=config,
                tags=tags,
                notes=notes
            )
            
            logger.info(f"Started experiment: {experiment_name} ({experiment_run.run_id})")
            return experiment_run.run_id
            
        except Exception as e:
            logger.error(f"Error starting experiment: {e}")
            raise
    
    async def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log metrics to the current experiment
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        try:
            await self.wandb_service.log_metrics(metrics, step=step)
            
        except Exception as e:
            logger.error(f"Error logging metrics: {e}")
    
    async def finish_experiment(
        self,
        final_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Finish the current experiment
        
        Args:
            final_metrics: Final metrics to log
        """
        try:
            await self.wandb_service.finish_experiment(final_metrics=final_metrics)
            
        except Exception as e:
            logger.error(f"Error finishing experiment: {e}")
    
    # Feature Store Methods
    
    async def update_features_from_sentiment(
        self,
        sentiment_results: List[SentimentResult]
    ) -> bool:
        """
        Update feature store with sentiment analysis results
        
        Args:
            sentiment_results: List of sentiment analysis results
            
        Returns:
            Success status
        """
        try:
            success = await self.hopsworks_service.update_sentiment_features_from_results(
                sentiment_results
            )
            
            if success:
                logger.info(f"Updated feature store with {len(sentiment_results)} sentiment results")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating features: {e}")
            return False
    
    async def get_features_for_training(
        self,
        feature_view_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[Any]:
        """
        Get features for model training
        
        Args:
            feature_view_name: Name of the feature view
            start_time: Start time for data
            end_time: End time for data
            
        Returns:
            Training data
        """
        try:
            training_data = await self.hopsworks_service.get_training_data(
                feature_view_name=feature_view_name,
                start_time=start_time,
                end_time=end_time
            )
            
            if training_data:
                logger.info(f"Retrieved training data from feature view: {feature_view_name}")
            
            return training_data
            
        except Exception as e:
            logger.error(f"Error getting training features: {e}")
            return None
    
    # Model Monitoring and Retraining Methods
    
    async def monitor_model_performance(
        self,
        model_id: str,
        model_version: str,
        performance_metrics: ModelPerformanceMetrics
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
            model_name = model_id.split('/')[-1]
            await self.retraining_service.monitor_model_performance(
                model_name, performance_metrics
            )
            
            if alerts:
                logger.warning(f"Generated {len(alerts)} alerts for model {model_id}")
            
        except Exception as e:
            logger.error(f"Error monitoring model performance: {e}")
    
    async def trigger_retraining(
        self,
        model_name: str,
        reason: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Manually trigger model retraining
        
        Args:
            model_name: Name of the model to retrain
            reason: Reason for retraining
            config: Optional training configuration
            
        Returns:
            Retraining job ID
        """
        try:
            experiment_run = await self.retraining_service.trigger_retraining(
                model_name=model_name,
                trigger_reason=reason,
                training_config=config
            )
            
            logger.info(f"Triggered retraining for {model_name}: {experiment_run.run_id}")
            return experiment_run.run_id
            
        except Exception as e:
            logger.error(f"Error triggering retraining: {e}")
            raise
    
    # Model Prediction Methods
    
    async def predict_sentiment(
        self,
        text: str,
        model_name: str = "sentiment_base",
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict sentiment using a deployed model
        
        Args:
            text: Text to analyze
            model_name: Name of the model to use
            model_version: Specific version of the model
            
        Returns:
            Sentiment prediction
        """
        try:
            prediction = await self.hf_service.predict_sentiment(
                text=text,
                model_name=model_name,
                model_version=model_version
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
        batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """
        Batch predict sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            model_name: Name of the model to use
            model_version: Specific version of the model
            batch_size: Batch size for processing
            
        Returns:
            List of sentiment predictions
        """
        try:
            predictions = await self.hf_service.batch_predict_sentiment(
                texts=texts,
                model_name=model_name,
                model_version=model_version,
                batch_size=batch_size
            )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in batch sentiment prediction: {e}")
            raise
    
    # Status and Information Methods
    
    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get status of all MLOps services
        
        Returns:
            Service status information
        """
        try:
            status = {
                "initialized": self.initialized,
                "services": self.services_status,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Update with current status
            if self.initialized:
                status["services"]["huggingface"] = self.hf_service.get_cache_info()
                status["services"]["wandb"] = self.wandb_service.get_service_status()
                status["services"]["hopsworks"] = self.hopsworks_service.get_service_status()
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting service status: {e}")
            return {"initialized": False, "error": str(e)}
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models
        
        Returns:
            List of available models
        """
        try:
            models = await self.hf_service.list_available_models(
                task="sentiment-analysis",
                limit=50
            )
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a model
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model information
        """
        try:
            model_info = await self.hf_service.get_model_info(model_id)
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}
    
    async def list_deployments(
        self,
        environment: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List active deployments
        
        Args:
            environment: Filter by environment
            
        Returns:
            List of deployments
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
        self,
        experiment_name: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        List experiment runs
        
        Args:
            experiment_name: Filter by experiment name
            limit: Maximum number of experiments to return
            
        Returns:
            List of experiments
        """
        try:
            experiments = await self.wandb_service.get_experiment_history(
                experiment_name=experiment_name or "",
                limit=limit
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