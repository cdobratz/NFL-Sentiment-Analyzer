"""
Model deployment and versioning system with A/B testing capabilities.
"""

import os
import logging
import json
import shutil
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import uuid
from pathlib import Path

from ...models.mlops import (
    ModelMetadata, ModelDeployment, ModelStatus, DeploymentStrategy,
    ABTestConfig, ModelPerformanceMetrics, ModelAlert
)
from ...core.config import get_settings
from ...core.database import get_database

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelDeploymentService:
    """Service for managing model deployments, versioning, and A/B testing"""
    
    def __init__(self):
        self.db = None
        self.models_dir = Path(getattr(settings, 'MODELS_DIR', './models'))
        self.deployments_dir = Path(getattr(settings, 'DEPLOYMENTS_DIR', './deployments'))
        self.active_deployments: Dict[str, ModelDeployment] = {}
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.performance_cache: Dict[str, ModelPerformanceMetrics] = {}
        
        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.deployments_dir.mkdir(exist_ok=True)
        
        # Performance thresholds
        self.performance_thresholds = {
            "accuracy": 0.8,
            "f1_score": 0.75,
            "error_rate": 0.05,
            "avg_prediction_time_ms": 100.0,
            "data_drift_score": 0.1
        }
    
    async def initialize(self):
        """Initialize the deployment service"""
        try:
            self.db = await get_database()
            await self._load_active_deployments()
            await self._load_active_ab_tests()
            logger.info("Model deployment service initialized")
        except Exception as e:
            logger.error(f"Error initializing deployment service: {e}")
    
    async def deploy_model(
        self,
        model_metadata: ModelMetadata,
        environment: str = "production",
        strategy: DeploymentStrategy = DeploymentStrategy.IMMEDIATE,
        traffic_percentage: float = 100.0,
        resource_config: Optional[Dict[str, Any]] = None
    ) -> ModelDeployment:
        """
        Deploy a model to the specified environment
        
        Args:
            model_metadata: Model metadata
            environment: Target environment (staging/production)
            strategy: Deployment strategy
            traffic_percentage: Percentage of traffic to route to this model
            resource_config: Resource configuration
            
        Returns:
            Model deployment object
        """
        try:
            deployment_id = str(uuid.uuid4())
            
            # Create deployment configuration
            deployment = ModelDeployment(
                deployment_id=deployment_id,
                model_id=model_metadata.model_id,
                model_version=model_metadata.version,
                environment=environment,
                strategy=strategy,
                status=ModelStatus.DEPLOYED,
                traffic_percentage=traffic_percentage,
                resource_limits=resource_config or {},
                health_check_url=f"/health/{deployment_id}",
                monitoring_enabled=True,
                alerts_enabled=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            # Execute deployment based on strategy
            success = await self._execute_deployment(deployment, model_metadata)
            
            if success:
                deployment.deployed_at = datetime.utcnow()
                deployment.status = ModelStatus.DEPLOYED
                
                # Store deployment
                await self._store_deployment(deployment)
                self.active_deployments[deployment_id] = deployment
                
                # Update model metadata
                model_metadata.status = ModelStatus.DEPLOYED
                model_metadata.deployment_config = {
                    "deployment_id": deployment_id,
                    "environment": environment,
                    "deployed_at": deployment.deployed_at.isoformat()
                }
                
                logger.info(f"Successfully deployed model {model_metadata.model_id} v{model_metadata.version}")
                
            else:
                deployment.status = ModelStatus.FAILED
                logger.error(f"Failed to deploy model {model_metadata.model_id}")
            
            return deployment
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
    
    async def create_ab_test(
        self,
        test_name: str,
        model_a_metadata: ModelMetadata,
        model_b_metadata: ModelMetadata,
        traffic_split_percentage: float = 50.0,
        success_metrics: Optional[List[str]] = None,
        test_duration_days: int = 7,
        minimum_sample_size: int = 1000
    ) -> ABTestConfig:
        """
        Create an A/B test between two models
        
        Args:
            test_name: Name of the A/B test
            model_a_metadata: Metadata for model A (control)
            model_b_metadata: Metadata for model B (treatment)
            traffic_split_percentage: Percentage of traffic for model B
            success_metrics: Metrics to track for success
            test_duration_days: Duration of the test in days
            minimum_sample_size: Minimum sample size for statistical significance
            
        Returns:
            A/B test configuration
        """
        try:
            test_id = str(uuid.uuid4())
            
            # Deploy both models to staging environment
            deployment_a = await self.deploy_model(
                model_a_metadata,
                environment="staging",
                strategy=DeploymentStrategy.A_B_TEST,
                traffic_percentage=100.0 - traffic_split_percentage
            )
            
            deployment_b = await self.deploy_model(
                model_b_metadata,
                environment="staging",
                strategy=DeploymentStrategy.A_B_TEST,
                traffic_percentage=traffic_split_percentage
            )
            
            # Create A/B test configuration
            ab_test = ABTestConfig(
                test_id=test_id,
                test_name=test_name,
                model_a_id=model_a_metadata.model_id,
                model_a_version=model_a_metadata.version,
                model_b_id=model_b_metadata.model_id,
                model_b_version=model_b_metadata.version,
                traffic_split_percentage=traffic_split_percentage,
                success_metrics=success_metrics or ["accuracy", "f1_score", "user_satisfaction"],
                minimum_sample_size=minimum_sample_size,
                test_duration_days=test_duration_days,
                status="running",
                started_at=datetime.utcnow()
            )
            
            # Store A/B test configuration
            await self._store_ab_test(ab_test)
            self.ab_tests[test_id] = ab_test
            
            logger.info(f"Created A/B test: {test_name} ({test_id})")
            return ab_test
            
        except Exception as e:
            logger.error(f"Error creating A/B test: {e}")
            raise
    
    async def monitor_model_performance(
        self,
        deployment_id: str,
        metrics: ModelPerformanceMetrics
    ) -> List[ModelAlert]:
        """
        Monitor model performance and generate alerts if needed
        
        Args:
            deployment_id: ID of the deployment to monitor
            metrics: Current performance metrics
            
        Returns:
            List of generated alerts
        """
        try:
            alerts = []
            
            if deployment_id not in self.active_deployments:
                logger.warning(f"Deployment {deployment_id} not found in active deployments")
                return alerts
            
            deployment = self.active_deployments[deployment_id]
            
            # Check performance thresholds
            alerts.extend(await self._check_performance_thresholds(deployment, metrics))
            
            # Check for data drift
            alerts.extend(await self._check_data_drift(deployment, metrics))
            
            # Check error rates
            alerts.extend(await self._check_error_rates(deployment, metrics))
            
            # Update performance cache
            self.performance_cache[deployment_id] = metrics
            
            # Store alerts
            for alert in alerts:
                await self._store_alert(alert)
            
            if alerts:
                logger.warning(f"Generated {len(alerts)} alerts for deployment {deployment_id}")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error monitoring model performance: {e}")
            return []
    
    async def rollback_deployment(
        self,
        deployment_id: str,
        reason: str,
        target_version: Optional[str] = None
    ) -> bool:
        """
        Rollback a deployment to a previous version
        
        Args:
            deployment_id: ID of the deployment to rollback
            reason: Reason for rollback
            target_version: Target version to rollback to (if None, uses previous)
            
        Returns:
            Success status
        """
        try:
            if deployment_id not in self.active_deployments:
                logger.error(f"Deployment {deployment_id} not found")
                return False
            
            deployment = self.active_deployments[deployment_id]
            
            # Find target version
            if not target_version:
                target_version = await self._get_previous_stable_version(
                    deployment.model_id, deployment.model_version
                )
            
            if not target_version:
                logger.error(f"No stable version found for rollback")
                return False
            
            # Get target model metadata
            target_metadata = await self._get_model_metadata(deployment.model_id, target_version)
            if not target_metadata:
                logger.error(f"Target model metadata not found: {deployment.model_id} v{target_version}")
                return False
            
            # Execute rollback
            success = await self._execute_rollback(deployment, target_metadata)
            
            if success:
                # Update deployment
                deployment.rollback_version = target_version
                deployment.rollback_reason = reason
                deployment.updated_at = datetime.utcnow()
                
                await self._store_deployment(deployment)
                
                logger.info(f"Successfully rolled back deployment {deployment_id} to version {target_version}")
                return True
            else:
                logger.error(f"Failed to rollback deployment {deployment_id}")
                return False
            
        except Exception as e:
            logger.error(f"Error rolling back deployment: {e}")
            return False
    
    async def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a specific deployment
        
        Args:
            deployment_id: ID of the deployment
            
        Returns:
            Deployment status information
        """
        try:
            if deployment_id not in self.active_deployments:
                return None
            
            deployment = self.active_deployments[deployment_id]
            performance = self.performance_cache.get(deployment_id)
            
            status = {
                "deployment_id": deployment_id,
                "model_id": deployment.model_id,
                "model_version": deployment.model_version,
                "environment": deployment.environment,
                "status": deployment.status.value,
                "traffic_percentage": deployment.traffic_percentage,
                "deployed_at": deployment.deployed_at.isoformat() if deployment.deployed_at else None,
                "health_status": await self._check_deployment_health(deployment),
                "performance_metrics": performance.dict() if performance else None,
                "recent_alerts": await self._get_recent_alerts(deployment_id)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting deployment status: {e}")
            return None
    
    async def list_active_deployments(
        self,
        environment: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all active deployments
        
        Args:
            environment: Filter by environment
            
        Returns:
            List of active deployments
        """
        try:
            deployments = []
            
            for deployment_id, deployment in self.active_deployments.items():
                if environment and deployment.environment != environment:
                    continue
                
                status = await self.get_deployment_status(deployment_id)
                if status:
                    deployments.append(status)
            
            return deployments
            
        except Exception as e:
            logger.error(f"Error listing active deployments: {e}")
            return []
    
    async def _execute_deployment(
        self,
        deployment: ModelDeployment,
        model_metadata: ModelMetadata
    ) -> bool:
        """Execute the actual deployment based on strategy"""
        try:
            if deployment.strategy == DeploymentStrategy.IMMEDIATE:
                return await self._immediate_deployment(deployment, model_metadata)
            elif deployment.strategy == DeploymentStrategy.BLUE_GREEN:
                return await self._blue_green_deployment(deployment, model_metadata)
            elif deployment.strategy == DeploymentStrategy.CANARY:
                return await self._canary_deployment(deployment, model_metadata)
            elif deployment.strategy == DeploymentStrategy.A_B_TEST:
                return await self._ab_test_deployment(deployment, model_metadata)
            else:
                logger.error(f"Unknown deployment strategy: {deployment.strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing deployment: {e}")
            return False
    
    async def _immediate_deployment(
        self,
        deployment: ModelDeployment,
        model_metadata: ModelMetadata
    ) -> bool:
        """Execute immediate deployment"""
        try:
            # Copy model files to deployment directory
            deployment_path = self.deployments_dir / deployment.deployment_id
            deployment_path.mkdir(exist_ok=True)
            
            # Copy model files
            if os.path.exists(model_metadata.model_path):
                shutil.copytree(
                    model_metadata.model_path,
                    deployment_path / "model",
                    dirs_exist_ok=True
                )
            
            # Create deployment configuration file
            config = {
                "deployment_id": deployment.deployment_id,
                "model_id": deployment.model_id,
                "model_version": deployment.model_version,
                "environment": deployment.environment,
                "traffic_percentage": deployment.traffic_percentage,
                "resource_limits": deployment.resource_limits,
                "deployed_at": datetime.utcnow().isoformat()
            }
            
            with open(deployment_path / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Immediate deployment completed: {deployment.deployment_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error in immediate deployment: {e}")
            return False
    
    async def _blue_green_deployment(
        self,
        deployment: ModelDeployment,
        model_metadata: ModelMetadata
    ) -> bool:
        """Execute blue-green deployment"""
        # Simplified implementation - in practice, this would involve
        # setting up parallel environments and switching traffic
        logger.info(f"Blue-green deployment: {deployment.deployment_id}")
        return await self._immediate_deployment(deployment, model_metadata)
    
    async def _canary_deployment(
        self,
        deployment: ModelDeployment,
        model_metadata: ModelMetadata
    ) -> bool:
        """Execute canary deployment"""
        # Simplified implementation - in practice, this would gradually
        # increase traffic to the new version
        logger.info(f"Canary deployment: {deployment.deployment_id}")
        return await self._immediate_deployment(deployment, model_metadata)
    
    async def _ab_test_deployment(
        self,
        deployment: ModelDeployment,
        model_metadata: ModelMetadata
    ) -> bool:
        """Execute A/B test deployment"""
        logger.info(f"A/B test deployment: {deployment.deployment_id}")
        return await self._immediate_deployment(deployment, model_metadata)
    
    async def _check_performance_thresholds(
        self,
        deployment: ModelDeployment,
        metrics: ModelPerformanceMetrics
    ) -> List[ModelAlert]:
        """Check if performance metrics exceed thresholds"""
        alerts = []
        
        for metric_name, threshold in self.performance_thresholds.items():
            current_value = getattr(metrics, metric_name, None)
            
            if current_value is None:
                continue
            
            # Check if threshold is exceeded
            threshold_exceeded = False
            if metric_name in ["error_rate", "avg_prediction_time_ms", "data_drift_score"]:
                # Lower is better
                threshold_exceeded = current_value > threshold
            else:
                # Higher is better
                threshold_exceeded = current_value < threshold
            
            if threshold_exceeded:
                alert = ModelAlert(
                    alert_id=str(uuid.uuid4()),
                    model_id=deployment.model_id,
                    model_version=deployment.model_version,
                    alert_type="performance_degradation",
                    severity="high" if abs(current_value - threshold) > threshold * 0.2 else "medium",
                    message=f"{metric_name} threshold exceeded",
                    description=f"{metric_name}: {current_value} (threshold: {threshold})",
                    threshold_value=threshold,
                    actual_value=current_value,
                    created_at=datetime.utcnow(),
                    recommended_actions=[
                        f"Investigate {metric_name} degradation",
                        "Consider model retraining",
                        "Check data quality"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    async def _check_data_drift(
        self,
        deployment: ModelDeployment,
        metrics: ModelPerformanceMetrics
    ) -> List[ModelAlert]:
        """Check for data drift"""
        alerts = []
        
        if metrics.data_drift_score and metrics.data_drift_score > 0.1:
            alert = ModelAlert(
                alert_id=str(uuid.uuid4()),
                model_id=deployment.model_id,
                model_version=deployment.model_version,
                alert_type="data_drift",
                severity="high",
                message="Data drift detected",
                description=f"Data drift score: {metrics.data_drift_score}",
                threshold_value=0.1,
                actual_value=metrics.data_drift_score,
                created_at=datetime.utcnow(),
                recommended_actions=[
                    "Investigate data distribution changes",
                    "Consider model retraining with recent data",
                    "Review data preprocessing pipeline"
                ]
            )
            alerts.append(alert)
        
        return alerts
    
    async def _check_error_rates(
        self,
        deployment: ModelDeployment,
        metrics: ModelPerformanceMetrics
    ) -> List[ModelAlert]:
        """Check error rates"""
        alerts = []
        
        if metrics.error_rate > 0.05:  # 5% error rate threshold
            alert = ModelAlert(
                alert_id=str(uuid.uuid4()),
                model_id=deployment.model_id,
                model_version=deployment.model_version,
                alert_type="error_spike",
                severity="critical" if metrics.error_rate > 0.1 else "high",
                message="High error rate detected",
                description=f"Error rate: {metrics.error_rate * 100:.2f}%",
                threshold_value=0.05,
                actual_value=metrics.error_rate,
                created_at=datetime.utcnow(),
                recommended_actions=[
                    "Investigate error causes",
                    "Check model input validation",
                    "Consider immediate rollback if critical"
                ]
            )
            alerts.append(alert)
        
        return alerts
    
    async def _store_deployment(self, deployment: ModelDeployment):
        """Store deployment in database"""
        if self.db:
            try:
                await self.db.deployments.replace_one(
                    {"deployment_id": deployment.deployment_id},
                    deployment.dict(),
                    upsert=True
                )
            except Exception as e:
                logger.error(f"Error storing deployment: {e}")
    
    async def _store_ab_test(self, ab_test: ABTestConfig):
        """Store A/B test configuration in database"""
        if self.db:
            try:
                await self.db.ab_tests.replace_one(
                    {"test_id": ab_test.test_id},
                    ab_test.dict(),
                    upsert=True
                )
            except Exception as e:
                logger.error(f"Error storing A/B test: {e}")
    
    async def _store_alert(self, alert: ModelAlert):
        """Store alert in database"""
        if self.db:
            try:
                await self.db.alerts.insert_one(alert.dict())
            except Exception as e:
                logger.error(f"Error storing alert: {e}")
    
    async def _load_active_deployments(self):
        """Load active deployments from database"""
        if self.db:
            try:
                cursor = self.db.deployments.find({"status": "deployed"})
                async for doc in cursor:
                    deployment = ModelDeployment(**doc)
                    self.active_deployments[deployment.deployment_id] = deployment
                logger.info(f"Loaded {len(self.active_deployments)} active deployments")
            except Exception as e:
                logger.error(f"Error loading active deployments: {e}")
    
    async def _load_active_ab_tests(self):
        """Load active A/B tests from database"""
        if self.db:
            try:
                cursor = self.db.ab_tests.find({"status": "running"})
                async for doc in cursor:
                    ab_test = ABTestConfig(**doc)
                    self.ab_tests[ab_test.test_id] = ab_test
                logger.info(f"Loaded {len(self.ab_tests)} active A/B tests")
            except Exception as e:
                logger.error(f"Error loading active A/B tests: {e}")
    
    async def _check_deployment_health(self, deployment: ModelDeployment) -> str:
        """Check health of a deployment"""
        # Simplified health check - in practice, this would ping the actual service
        return "healthy"
    
    async def _get_recent_alerts(self, deployment_id: str) -> List[Dict[str, Any]]:
        """Get recent alerts for a deployment"""
        if not self.db:
            return []
        
        try:
            deployment = self.active_deployments.get(deployment_id)
            if not deployment:
                return []
            
            # Get alerts from last 24 hours
            since = datetime.utcnow() - timedelta(hours=24)
            cursor = self.db.alerts.find({
                "model_id": deployment.model_id,
                "model_version": deployment.model_version,
                "created_at": {"$gte": since}
            }).sort("created_at", -1).limit(10)
            
            alerts = []
            async for doc in cursor:
                alerts.append(doc)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error getting recent alerts: {e}")
            return []
    
    async def _get_previous_stable_version(self, model_id: str, current_version: str) -> Optional[str]:
        """Get the previous stable version of a model"""
        # This would query the model registry for the previous stable version
        # For now, return a mock version
        return "1.0"
    
    async def _get_model_metadata(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """Get model metadata for a specific version"""
        # This would query the model registry
        # For now, return None
        return None
    
    async def _execute_rollback(
        self,
        deployment: ModelDeployment,
        target_metadata: ModelMetadata
    ) -> bool:
        """Execute rollback to target version"""
        try:
            # In practice, this would involve updating the deployment
            # to point to the target model version
            logger.info(f"Rolling back deployment {deployment.deployment_id} to {target_metadata.version}")
            return True
        except Exception as e:
            logger.error(f"Error executing rollback: {e}")
            return False