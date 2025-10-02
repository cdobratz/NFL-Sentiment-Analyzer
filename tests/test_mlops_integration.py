"""
Integration tests for MLOps pipeline components.
Tests end-to-end workflows and component interactions.
"""

import pytest
import asyncio
import tempfile
import json
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

from app.services.mlops.mlops_service import MLOpsService
from app.services.mlops.model_deployment_service import ModelDeploymentService
from app.services.mlops.model_retraining_service import ModelRetrainingService
from app.services.mlops.hopsworks_service import HopsworksService
from app.services.mlops.huggingface_service import HuggingFaceModelService
from app.services.mlops.wandb_service import WandBService

from app.models.mlops import (
    ModelMetadata, ModelDeployment, ModelPerformanceMetrics, ModelAlert,
    ModelRetrainingConfig, ExperimentRun, ABTestConfig,
    ModelType, ModelStatus, DeploymentStrategy, ExperimentStatus
)
from app.models.sentiment import SentimentResult, SentimentLabel, DataSource


class TestMLOpsEndToEndWorkflow:
    """Test complete MLOps workflow from training to deployment."""
    
    @pytest.fixture
    def mlops_service(self):
        """Create MLOps service with mocked components."""
        service = MLOpsService()
        
        # Mock all component services
        service.hf_service = MagicMock(spec=HuggingFaceModelService)
        service.wandb_service = MagicMock(spec=WandBService)
        service.hopsworks_service = MagicMock(spec=HopsworksService)
        service.deployment_service = MagicMock(spec=ModelDeploymentService)
        service.retraining_service = MagicMock(spec=ModelRetrainingService)
        
        service.initialized = True
        service.services_status = {
            "huggingface": {"cache_size": 0},
            "wandb": {"authenticated": True},
            "hopsworks": {"connected": True},
            "deployment": {"initialized": True},
            "retraining": {"initialized": True}
        }
        
        return service
    
    @pytest.mark.asyncio
    async def test_complete_model_lifecycle(self, mlops_service):
        """Test complete model lifecycle from training to production deployment."""
        
        # 1. Start training experiment
        experiment_run = ExperimentRun(
            experiment_id="nfl_sentiment_training",
            run_id="run_20240115_001",
            experiment_name="nfl_sentiment_training",
            run_name="baseline_training",
            status=ExperimentStatus.RUNNING,
            model_type=ModelType.SENTIMENT_ANALYSIS,
            hyperparameters={
                "learning_rate": 2e-5,
                "batch_size": 16,
                "num_epochs": 3,
                "max_length": 512
            },
            started_at=datetime.utcnow(),
            tags=["baseline", "nfl", "sentiment"]
        )
        
        mlops_service.wandb_service.start_experiment.return_value = experiment_run
        
        run_id = await mlops_service.start_experiment(
            experiment_name="nfl_sentiment_training",
            config=experiment_run.hyperparameters,
            tags=experiment_run.tags
        )
        
        assert run_id == experiment_run.run_id
        mlops_service.wandb_service.start_experiment.assert_called_once()
        
        # 2. Log training metrics during training
        training_metrics = [
            {"epoch": 1, "train_loss": 0.45, "val_loss": 0.38, "val_accuracy": 0.82},
            {"epoch": 2, "train_loss": 0.32, "val_loss": 0.29, "val_accuracy": 0.85},
            {"epoch": 3, "train_loss": 0.25, "val_loss": 0.27, "val_accuracy": 0.87}
        ]
        
        for metrics in training_metrics:
            await mlops_service.log_metrics(metrics, step=metrics["epoch"])
        
        assert mlops_service.wandb_service.log_metrics.call_count == 3
        
        # 3. Register trained model
        model_metadata = ModelMetadata(
            model_id="nfl_sentiment_v1_0",
            model_name="nfl_sentiment_analyzer",
            version="1.0",
            model_type=ModelType.SENTIMENT_ANALYSIS,
            status=ModelStatus.VALIDATING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="training_pipeline",
            framework="transformers",
            model_path="./models/nfl_sentiment_v1_0",
            accuracy=0.87,
            precision=0.85,
            recall=0.89,
            f1_score=0.87,
            training_dataset="nfl_sentiment_train_v1",
            validation_dataset="nfl_sentiment_val_v1",
            epochs=3,
            learning_rate=2e-5,
            batch_size=16
        )
        
        mlops_service.hf_service.register_model.return_value = model_metadata
        
        registered_model = await mlops_service.register_model(
            model_name="nfl_sentiment_analyzer",
            model_path="./models/nfl_sentiment_v1_0",
            metrics={"accuracy": 0.87, "f1_score": 0.87}
        )
        
        assert registered_model.model_id == model_metadata.model_id
        assert registered_model.accuracy == 0.87
        
        # 4. Deploy to staging for validation
        staging_deployment = ModelDeployment(
            deployment_id="staging_deployment_001",
            model_id=model_metadata.model_id,
            model_version=model_metadata.version,
            environment="staging",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            deployed_at=datetime.utcnow(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        mlops_service.deployment_service.deploy_model.return_value = staging_deployment
        
        deployed_staging = await mlops_service.deploy_model(
            model_id=model_metadata.model_id,
            model_version=model_metadata.version,
            environment="staging"
        )
        
        assert deployed_staging.environment == "staging"
        assert deployed_staging.status == ModelStatus.DEPLOYED
        
        # 5. Monitor staging performance
        staging_metrics = ModelPerformanceMetrics(
            model_id=model_metadata.model_id,
            model_version=model_metadata.version,
            timestamp=datetime.utcnow(),
            accuracy=0.86,
            precision=0.84,
            recall=0.88,
            f1_score=0.86,
            prediction_count=5000,
            avg_prediction_time_ms=45.0,
            error_rate=0.02,
            throughput_per_second=120.0,
            data_drift_score=0.03,
            period_start=datetime.utcnow() - timedelta(hours=24),
            period_end=datetime.utcnow(),
            sample_size=5000
        )
        
        await mlops_service.monitor_model_performance(
            model_id=model_metadata.model_id,
            model_version=model_metadata.version,
            performance_metrics=staging_metrics
        )
        
        # Verify monitoring was called
        mlops_service.deployment_service.monitor_model_performance.assert_called_once()
        mlops_service.retraining_service.monitor_model_performance.assert_called_once()
        
        # 6. Deploy to production after validation
        production_deployment = ModelDeployment(
            deployment_id="prod_deployment_001",
            model_id=model_metadata.model_id,
            model_version=model_metadata.version,
            environment="production",
            strategy=DeploymentStrategy.CANARY,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=10.0,  # Start with 10% traffic
            deployed_at=datetime.utcnow(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        mlops_service.deployment_service.deploy_model.return_value = production_deployment
        
        deployed_prod = await mlops_service.deploy_model(
            model_id=model_metadata.model_id,
            model_version=model_metadata.version,
            environment="production",
            strategy=DeploymentStrategy.CANARY,
            traffic_percentage=10.0
        )
        
        assert deployed_prod.environment == "production"
        assert deployed_prod.strategy == DeploymentStrategy.CANARY
        assert deployed_prod.traffic_percentage == 10.0
        
        # 7. Finish experiment
        experiment_run.status = ExperimentStatus.COMPLETED
        experiment_run.completed_at = datetime.utcnow()
        experiment_run.metrics = {"final_accuracy": 0.87, "final_f1": 0.87}
        
        mlops_service.wandb_service.finish_experiment.return_value = experiment_run
        
        final_experiment = await mlops_service.finish_experiment(
            final_metrics={"final_accuracy": 0.87, "final_f1": 0.87}
        )
        
        mlops_service.wandb_service.finish_experiment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_automated_retraining_workflow(self, mlops_service):
        """Test automated retraining workflow triggered by performance degradation."""
        
        # 1. Setup existing model with retraining config
        model_name = "nfl_sentiment_analyzer"
        
        retraining_config = ModelRetrainingConfig(
            config_id="retraining_config_001",
            model_name=model_name,
            performance_threshold=0.8,
            data_drift_threshold=0.1,
            auto_deploy=False,
            approval_required=True,
            enabled=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        mlops_service.retraining_service.create_retraining_config.return_value = retraining_config
        
        config = await mlops_service.retraining_service.create_retraining_config(
            model_name=model_name,
            performance_threshold=0.8,
            data_drift_threshold=0.1
        )
        
        assert config.model_name == model_name
        
        # 2. Simulate performance degradation
        degraded_metrics = ModelPerformanceMetrics(
            model_id=model_name,
            model_version="1.0",
            timestamp=datetime.utcnow(),
            accuracy=0.75,  # Below threshold
            precision=0.73,
            recall=0.77,
            f1_score=0.75,  # Below threshold
            prediction_count=10000,
            avg_prediction_time_ms=55.0,
            error_rate=0.04,
            data_drift_score=0.12,  # Above threshold
            period_start=datetime.utcnow() - timedelta(hours=24),
            period_end=datetime.utcnow(),
            sample_size=10000
        )
        
        # 3. Check retraining triggers
        mlops_service.retraining_service.check_retraining_triggers.return_value = {
            "should_retrain": True,
            "triggers_activated": ["performance_degradation", "data_drift"],
            "config_id": config.config_id
        }
        
        trigger_result = await mlops_service.retraining_service.check_retraining_triggers(model_name)
        
        assert trigger_result["should_retrain"] is True
        assert "performance_degradation" in trigger_result["triggers_activated"]
        assert "data_drift" in trigger_result["triggers_activated"]
        
        # 4. Trigger retraining
        retraining_run = ExperimentRun(
            experiment_id="nfl_sentiment_retraining",
            run_id="retrain_20240115_001",
            experiment_name="nfl_sentiment_analyzer_retraining",
            run_name="auto_retrain_perf_degradation",
            status=ExperimentStatus.RUNNING,
            model_type=ModelType.SENTIMENT_ANALYSIS,
            hyperparameters={
                "learning_rate": 1e-5,  # Lower learning rate for fine-tuning
                "batch_size": 16,
                "num_epochs": 2,
                "base_model": "nfl_sentiment_v1_0"
            },
            started_at=datetime.utcnow(),
            tags=["retraining", "auto", "performance_degradation"]
        )
        
        mlops_service.retraining_service.trigger_retraining.return_value = retraining_run
        
        retrain_run = await mlops_service.trigger_retraining(
            model_name=model_name,
            trigger_reason="performance_degradation,data_drift"
        )
        
        assert retrain_run.run_id == retraining_run.run_id
        assert "retraining" in retrain_run.tags
        
        # 5. Monitor retraining progress
        mlops_service.retraining_service.get_retraining_status.return_value = {
            "run_id": retraining_run.run_id,
            "status": "running",
            "progress": 0.6,
            "current_epoch": 2,
            "estimated_completion": datetime.utcnow() + timedelta(minutes=30)
        }
        
        status = await mlops_service.retraining_service.get_retraining_status(retrain_run.run_id)
        
        assert status["status"] == "running"
        assert status["progress"] == 0.6
    
    @pytest.mark.asyncio
    async def test_ab_testing_workflow(self, mlops_service):
        """Test A/B testing workflow between model versions."""
        
        # 1. Create two model versions for comparison
        model_a = ModelMetadata(
            model_id="nfl_sentiment_v1_0",
            model_name="nfl_sentiment_analyzer",
            version="1.0",
            model_type=ModelType.SENTIMENT_ANALYSIS,
            status=ModelStatus.DEPLOYED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="training_pipeline",
            framework="transformers",
            model_path="./models/nfl_sentiment_v1_0",
            accuracy=0.85,
            f1_score=0.83
        )
        
        model_b = ModelMetadata(
            model_id="nfl_sentiment_v1_1",
            model_name="nfl_sentiment_analyzer",
            version="1.1",
            model_type=ModelType.SENTIMENT_ANALYSIS,
            status=ModelStatus.DEPLOYED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="training_pipeline",
            framework="transformers",
            model_path="./models/nfl_sentiment_v1_1",
            accuracy=0.87,
            f1_score=0.85
        )
        
        # 2. Create A/B test
        ab_test = ABTestConfig(
            test_id="ab_test_001",
            test_name="sentiment_v1_0_vs_v1_1",
            model_a_id=model_a.model_id,
            model_a_version=model_a.version,
            model_b_id=model_b.model_id,
            model_b_version=model_b.version,
            traffic_split_percentage=50.0,
            success_metrics=["accuracy", "f1_score", "latency", "user_satisfaction"],
            minimum_sample_size=10000,
            test_duration_days=14,
            status="running",
            started_at=datetime.utcnow()
        )
        
        mlops_service.deployment_service.create_ab_test.return_value = ab_test
        
        created_test = await mlops_service.create_ab_test(
            test_name="sentiment_v1_0_vs_v1_1",
            model_a_id=model_a.model_id,
            model_a_version=model_a.version,
            model_b_id=model_b.model_id,
            model_b_version=model_b.version,
            traffic_split=50.0,
            duration_days=14
        )
        
        assert created_test == ab_test.test_id
        
        # 3. Simulate A/B test data collection
        ab_test_metrics = {
            "model_a_performance": {
                "accuracy": 0.84,
                "f1_score": 0.82,
                "avg_latency_ms": 48.0,
                "user_satisfaction": 4.1,
                "sample_size": 12500,
                "error_rate": 0.03
            },
            "model_b_performance": {
                "accuracy": 0.86,
                "f1_score": 0.84,
                "avg_latency_ms": 52.0,
                "user_satisfaction": 4.3,
                "sample_size": 12500,
                "error_rate": 0.025
            },
            "statistical_significance": {
                "accuracy_p_value": 0.02,
                "f1_score_p_value": 0.03,
                "latency_p_value": 0.15,
                "user_satisfaction_p_value": 0.01
            }
        }
        
        # 4. Analyze A/B test results
        analysis_result = self._analyze_ab_test_results(ab_test_metrics)
        
        assert analysis_result["winner"] == "model_b"
        assert analysis_result["confidence_level"] > 0.95
        assert "accuracy" in analysis_result["significant_improvements"]
        assert "f1_score" in analysis_result["significant_improvements"]
        
        # 5. Deploy winning model based on results
        if analysis_result["recommendation"] == "deploy_model_b":
            production_deployment = ModelDeployment(
                deployment_id="prod_deployment_ab_winner",
                model_id=model_b.model_id,
                model_version=model_b.version,
                environment="production",
                strategy=DeploymentStrategy.BLUE_GREEN,
                status=ModelStatus.DEPLOYED,
                traffic_percentage=100.0,
                deployed_at=datetime.utcnow(),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            mlops_service.deployment_service.deploy_model.return_value = production_deployment
            
            winner_deployment = await mlops_service.deploy_model(
                model_id=model_b.model_id,
                model_version=model_b.version,
                environment="production",
                strategy=DeploymentStrategy.BLUE_GREEN
            )
            
            assert winner_deployment.model_id == model_b.model_id
            assert winner_deployment.strategy == DeploymentStrategy.BLUE_GREEN
    
    @pytest.mark.asyncio
    async def test_feature_store_integration_workflow(self, mlops_service):
        """Test feature store integration workflow."""
        
        # 1. Generate sentiment analysis results
        sentiment_results = [
            SentimentResult(
                text="Amazing touchdown pass by the quarterback!",
                sentiment=SentimentLabel.POSITIVE,
                sentiment_score=0.85,
                confidence=0.92,
                category="performance",
                context={"nfl_context": {"team_mentions": ["patriots"], "position_mentions": ["QB"]}},
                source=DataSource.TWITTER,
                timestamp=datetime.utcnow(),
                team_id="patriots",
                model_version="1.0",
                processing_time_ms=45.0,
                aspect_sentiments={"performance": 0.85, "coaching": 0.1}
            ),
            SentimentResult(
                text="Terrible fumble in the red zone",
                sentiment=SentimentLabel.NEGATIVE,
                sentiment_score=-0.78,
                confidence=0.88,
                category="performance",
                context={"nfl_context": {"team_mentions": ["cowboys"], "performance_metrics": ["fumble"]}},
                source=DataSource.ESPN,
                timestamp=datetime.utcnow(),
                team_id="cowboys",
                model_version="1.0",
                processing_time_ms=42.0,
                aspect_sentiments={"performance": -0.78, "execution": -0.65}
            ),
            SentimentResult(
                text="Player injury concerns for next week",
                sentiment=SentimentLabel.NEGATIVE,
                sentiment_score=-0.45,
                confidence=0.75,
                category="injury",
                context={"nfl_context": {"team_mentions": ["packers"], "injury_related": True}},
                source=DataSource.NEWS,
                timestamp=datetime.utcnow(),
                team_id="packers",
                player_id="player_123",
                model_version="1.0",
                processing_time_ms=38.0,
                aspect_sentiments={"injury": -0.45, "availability": -0.55}
            )
        ]
        
        # 2. Update feature store with sentiment results
        mlops_service.hopsworks_service.update_sentiment_features_from_results.return_value = True
        
        success = await mlops_service.update_features_from_sentiment(sentiment_results)
        
        assert success is True
        mlops_service.hopsworks_service.update_sentiment_features_from_results.assert_called_once_with(
            sentiment_results
        )
        
        # 3. Retrieve features for model training
        training_data = {
            "features": [
                {
                    "entity_id": "patriots",
                    "entity_type": "team",
                    "sentiment_score": 0.85,
                    "confidence": 0.92,
                    "positive_mentions": 1,
                    "negative_mentions": 0,
                    "total_mentions": 1,
                    "performance_sentiment": 0.85
                },
                {
                    "entity_id": "cowboys",
                    "entity_type": "team",
                    "sentiment_score": -0.78,
                    "confidence": 0.88,
                    "positive_mentions": 0,
                    "negative_mentions": 1,
                    "total_mentions": 1,
                    "performance_sentiment": -0.78
                }
            ],
            "labels": ["POSITIVE", "NEGATIVE"]
        }
        
        mlops_service.hopsworks_service.get_training_data.return_value = (
            training_data["features"],
            training_data["labels"]
        )
        
        retrieved_data = await mlops_service.get_features_for_training(
            feature_view_name="nfl_sentiment_features",
            start_time=datetime.utcnow() - timedelta(days=30),
            end_time=datetime.utcnow()
        )
        
        assert retrieved_data is not None
        features, labels = retrieved_data
        assert len(features) == 2
        assert len(labels) == 2
    
    @pytest.mark.asyncio
    async def test_model_rollback_workflow(self, mlops_service):
        """Test model rollback workflow due to performance issues."""
        
        # 1. Setup current production deployment
        current_deployment = ModelDeployment(
            deployment_id="prod_deployment_current",
            model_id="nfl_sentiment_v2_0",
            model_version="2.0",
            environment="production",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            deployed_at=datetime.utcnow() - timedelta(hours=2),
            created_at=datetime.utcnow() - timedelta(hours=2),
            updated_at=datetime.utcnow() - timedelta(hours=2)
        )
        
        # 2. Detect critical performance degradation
        critical_metrics = ModelPerformanceMetrics(
            model_id="nfl_sentiment_v2_0",
            model_version="2.0",
            timestamp=datetime.utcnow(),
            accuracy=0.62,  # Critical drop
            precision=0.60,
            recall=0.65,
            f1_score=0.62,
            prediction_count=15000,
            avg_prediction_time_ms=180.0,  # High latency
            error_rate=0.15,  # High error rate
            throughput_per_second=25.0,  # Low throughput
            data_drift_score=0.25,  # High drift
            period_start=datetime.utcnow() - timedelta(hours=1),
            period_end=datetime.utcnow(),
            sample_size=15000
        )
        
        # 3. Generate critical alerts
        critical_alerts = [
            ModelAlert(
                alert_id="alert_critical_001",
                model_id="nfl_sentiment_v2_0",
                model_version="2.0",
                alert_type="performance_degradation",
                severity="critical",
                message="Critical accuracy drop detected",
                description="Accuracy dropped to 62% from baseline 85%",
                threshold_value=0.8,
                actual_value=0.62,
                created_at=datetime.utcnow(),
                recommended_actions=["Immediate rollback", "Investigate data quality"]
            ),
            ModelAlert(
                alert_id="alert_critical_002",
                model_id="nfl_sentiment_v2_0",
                model_version="2.0",
                alert_type="error_spike",
                severity="critical",
                message="Critical error rate spike",
                description="Error rate increased to 15%",
                threshold_value=0.05,
                actual_value=0.15,
                created_at=datetime.utcnow(),
                recommended_actions=["Immediate rollback", "Check model integrity"]
            )
        ]
        
        mlops_service.deployment_service.monitor_model_performance.return_value = critical_alerts
        
        alerts = await mlops_service.deployment_service.monitor_model_performance(
            current_deployment.deployment_id,
            critical_metrics
        )
        
        assert len(alerts) == 2
        assert all(alert.severity == "critical" for alert in alerts)
        
        # 4. Trigger automatic rollback
        mlops_service.deployment_service.rollback_deployment.return_value = True
        
        rollback_success = await mlops_service.deployment_service.rollback_deployment(
            deployment_id=current_deployment.deployment_id,
            reason="Critical performance degradation - automatic rollback",
            target_version="1.5"  # Previous stable version
        )
        
        assert rollback_success is True
        
        # 5. Verify rollback deployment
        rollback_deployment = ModelDeployment(
            deployment_id="prod_deployment_rollback",
            model_id="nfl_sentiment_v1_5",
            model_version="1.5",
            environment="production",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            deployed_at=datetime.utcnow(),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            rollback_version="1.5",
            rollback_reason="Critical performance degradation - automatic rollback"
        )
        
        mlops_service.deployment_service.deploy_model.return_value = rollback_deployment
        
        # Deploy the rollback version
        rollback_deployed = await mlops_service.deploy_model(
            model_id="nfl_sentiment_v1_5",
            model_version="1.5",
            environment="production"
        )
        
        assert rollback_deployed.model_version == "1.5"
        assert rollback_deployed.rollback_reason is not None
    
    def _analyze_ab_test_results(self, metrics: Dict) -> Dict:
        """Analyze A/B test results and determine winner."""
        model_a_perf = metrics["model_a_performance"]
        model_b_perf = metrics["model_b_performance"]
        significance = metrics["statistical_significance"]
        
        # Calculate overall performance scores
        a_score = (model_a_perf["accuracy"] + model_a_perf["f1_score"]) / 2
        b_score = (model_b_perf["accuracy"] + model_b_perf["f1_score"]) / 2
        
        winner = "model_b" if b_score > a_score else "model_a"
        
        # Check statistical significance
        significant_improvements = []
        for metric, p_value in significance.items():
            if p_value < 0.05:  # 95% confidence
                metric_name = metric.replace("_p_value", "")
                if model_b_perf[metric_name] > model_a_perf[metric_name]:
                    significant_improvements.append(metric_name)
        
        confidence_level = 1 - max(significance.values())
        
        # Make recommendation
        if len(significant_improvements) >= 2 and winner == "model_b":
            recommendation = "deploy_model_b"
        elif len(significant_improvements) >= 2 and winner == "model_a":
            recommendation = "keep_model_a"
        else:
            recommendation = "extend_test"
        
        return {
            "winner": winner,
            "confidence_level": confidence_level,
            "significant_improvements": significant_improvements,
            "recommendation": recommendation,
            "performance_lift": {
                "accuracy": model_b_perf["accuracy"] - model_a_perf["accuracy"],
                "f1_score": model_b_perf["f1_score"] - model_a_perf["f1_score"]
            }
        }


class TestMLOpsServiceIntegration:
    """Test MLOps service integration with external services."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test MLOps service initialization with all components."""
        service = MLOpsService()
        
        # Mock component initialization
        with patch.object(service.deployment_service, 'initialize', new_callable=AsyncMock) as mock_deploy_init, \
             patch.object(service.retraining_service, 'initialize', new_callable=AsyncMock) as mock_retrain_init:
            
            await service.initialize()
            
            assert service.initialized is True
            mock_deploy_init.assert_called_once()
            mock_retrain_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_service_status_monitoring(self):
        """Test service status monitoring and health checks."""
        service = MLOpsService()
        service.initialized = True
        
        # Mock service status methods
        service.hf_service.get_cache_info = MagicMock(return_value={"cache_size": 5})
        service.wandb_service.get_service_status = MagicMock(return_value={"authenticated": True})
        service.hopsworks_service.get_service_status = MagicMock(return_value={"connected": True})
        
        status = await service.get_service_status()
        
        assert status["initialized"] is True
        assert "services" in status
        assert status["services"]["huggingface"]["cache_size"] == 5
        assert status["services"]["wandb"]["authenticated"] is True
        assert status["services"]["hopsworks"]["connected"] is True
    
    @pytest.mark.asyncio
    async def test_model_prediction_integration(self):
        """Test model prediction integration with HuggingFace service."""
        service = MLOpsService()
        service.initialized = True
        
        # Mock prediction
        mock_prediction = {
            "label": "POSITIVE",
            "sentiment_score": 0.82,
            "confidence": 0.91,
            "raw_prediction": {"label": "POSITIVE", "score": 0.91}
        }
        
        service.hf_service.predict_sentiment = AsyncMock(return_value=mock_prediction)
        
        prediction = await service.predict_sentiment(
            text="Great touchdown pass by the quarterback!",
            model_name="nfl_sentiment_base"
        )
        
        assert prediction["label"] == "POSITIVE"
        assert prediction["sentiment_score"] == 0.82
        assert prediction["confidence"] == 0.91
        
        service.hf_service.predict_sentiment.assert_called_once_with(
            text="Great touchdown pass by the quarterback!",
            model_name="nfl_sentiment_base",
            model_version=None
        )
    
    @pytest.mark.asyncio
    async def test_batch_prediction_integration(self):
        """Test batch prediction integration."""
        service = MLOpsService()
        service.initialized = True
        
        texts = [
            "Amazing performance by the team!",
            "Terrible fumble in the end zone",
            "Game scheduled for Sunday"
        ]
        
        mock_predictions = [
            {"label": "POSITIVE", "sentiment_score": 0.85, "confidence": 0.92},
            {"label": "NEGATIVE", "sentiment_score": -0.78, "confidence": 0.88},
            {"label": "NEUTRAL", "sentiment_score": 0.02, "confidence": 0.65}
        ]
        
        service.hf_service.batch_predict_sentiment = AsyncMock(return_value=mock_predictions)
        
        predictions = await service.batch_predict_sentiment(
            texts=texts,
            model_name="nfl_sentiment_base",
            batch_size=16
        )
        
        assert len(predictions) == 3
        assert predictions[0]["label"] == "POSITIVE"
        assert predictions[1]["label"] == "NEGATIVE"
        assert predictions[2]["label"] == "NEUTRAL"
    
    @pytest.mark.asyncio
    async def test_experiment_tracking_integration(self):
        """Test experiment tracking integration with W&B."""
        service = MLOpsService()
        service.initialized = True
        
        # Mock experiment run
        experiment_run = ExperimentRun(
            experiment_id="test_experiment",
            run_id="run_123",
            experiment_name="model_training",
            run_name="training_run_001",
            status=ExperimentStatus.RUNNING,
            model_type=ModelType.SENTIMENT_ANALYSIS,
            hyperparameters={"lr": 2e-5, "batch_size": 16},
            started_at=datetime.utcnow()
        )
        
        service.wandb_service.start_experiment = AsyncMock(return_value=experiment_run)
        service.wandb_service.log_metrics = AsyncMock()
        service.wandb_service.finish_experiment = AsyncMock(return_value=experiment_run)
        
        # Start experiment
        run_id = await service.start_experiment(
            experiment_name="model_training",
            config={"lr": 2e-5, "batch_size": 16}
        )
        
        assert run_id == experiment_run.run_id
        
        # Log metrics
        await service.log_metrics({"accuracy": 0.85, "loss": 0.15})
        service.wandb_service.log_metrics.assert_called_once()
        
        # Finish experiment
        await service.finish_experiment({"final_accuracy": 0.87})
        service.wandb_service.finish_experiment.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deployment_integration(self):
        """Test deployment integration with deployment service."""
        service = MLOpsService()
        service.initialized = True
        
        # Mock deployment
        deployment = ModelDeployment(
            deployment_id="deployment_123",
            model_id="test_model",
            model_version="1.0",
            environment="production",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        service.deployment_service.deploy_model = AsyncMock(return_value=deployment)
        service.deployment_service.list_active_deployments = AsyncMock(return_value=[deployment.dict()])
        
        # Deploy model
        deployed = await service.deploy_model(
            model_id="test_model",
            model_version="1.0",
            environment="production"
        )
        
        assert deployed.deployment_id == deployment.deployment_id
        
        # List deployments
        deployments = await service.list_deployments(environment="production")
        assert len(deployments) == 1
        assert deployments[0]["deployment_id"] == deployment.deployment_id


class TestMLOpsErrorHandling:
    """Test error handling and resilience in MLOps pipeline."""
    
    @pytest.mark.asyncio
    async def test_service_initialization_failure(self):
        """Test handling of service initialization failures."""
        service = MLOpsService()
        
        # Mock initialization failure
        with patch.object(service.deployment_service, 'initialize', side_effect=Exception("Database connection failed")):
            with pytest.raises(Exception, match="Database connection failed"):
                await service.initialize()
            
            assert service.initialized is False
    
    @pytest.mark.asyncio
    async def test_prediction_service_failure(self):
        """Test handling of prediction service failures."""
        service = MLOpsService()
        service.initialized = True
        
        # Mock prediction failure
        service.hf_service.predict_sentiment = AsyncMock(side_effect=Exception("Model not found"))
        
        with pytest.raises(Exception, match="Model not found"):
            await service.predict_sentiment("Test text", "nonexistent_model")
    
    @pytest.mark.asyncio
    async def test_deployment_failure_recovery(self):
        """Test deployment failure recovery mechanisms."""
        service = MLOpsService()
        service.initialized = True
        
        # Mock deployment failure
        service.deployment_service.deploy_model = AsyncMock(side_effect=Exception("Deployment failed"))
        
        with pytest.raises(Exception, match="Deployment failed"):
            await service.deploy_model("test_model", "1.0", "production")
    
    @pytest.mark.asyncio
    async def test_monitoring_service_resilience(self):
        """Test monitoring service resilience to failures."""
        service = MLOpsService()
        service.initialized = True
        
        # Mock monitoring failure (should not crash the service)
        service.deployment_service.monitor_model_performance = AsyncMock(side_effect=Exception("Monitoring error"))
        
        # Should handle the error gracefully
        try:
            await service.monitor_model_performance("test_model", "1.0", MagicMock())
        except Exception as e:
            # The service should handle this gracefully or re-raise with context
            assert "Monitoring error" in str(e)
    
    @pytest.mark.asyncio
    async def test_feature_store_connection_failure(self):
        """Test feature store connection failure handling."""
        service = MLOpsService()
        service.initialized = True
        
        # Mock feature store failure
        service.hopsworks_service.update_sentiment_features_from_results = AsyncMock(return_value=False)
        
        success = await service.update_features_from_sentiment([])
        
        assert success is False  # Should return False on failure, not crash