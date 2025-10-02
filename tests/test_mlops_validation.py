"""
MLOps testing and validation module.
Tests model validation, deployment procedures, data drift detection, and performance monitoring.
"""

import pytest
import asyncio
import tempfile
import shutil
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pandas as pd

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


class TestModelValidation:
    """Test automated model validation with benchmark datasets."""
    
    @pytest.fixture
    def mock_mlops_service(self):
        """Mock MLOps service for testing."""
        service = MLOpsService()
        service.hf_service = MagicMock()
        service.wandb_service = MagicMock()
        service.deployment_service = MagicMock()
        service.retraining_service = MagicMock()
        return service
    
    @pytest.fixture
    def benchmark_dataset(self):
        """Create benchmark dataset for model validation."""
        return [
            {"text": "Amazing touchdown pass!", "label": "POSITIVE", "expected_score": 0.8},
            {"text": "Terrible fumble by the quarterback", "label": "NEGATIVE", "expected_score": -0.7},
            {"text": "The game is scheduled for Sunday", "label": "NEUTRAL", "expected_score": 0.0},
            {"text": "Incredible performance by the team", "label": "POSITIVE", "expected_score": 0.9},
            {"text": "Disappointing loss in overtime", "label": "NEGATIVE", "expected_score": -0.6},
            {"text": "Player statistics for the season", "label": "NEUTRAL", "expected_score": 0.1},
            {"text": "Outstanding defensive play", "label": "POSITIVE", "expected_score": 0.7},
            {"text": "Injury concerns for key players", "label": "NEGATIVE", "expected_score": -0.5},
            {"text": "Game postponed due to weather", "label": "NEUTRAL", "expected_score": 0.0},
            {"text": "Championship winning performance", "label": "POSITIVE", "expected_score": 0.95}
        ]
    
    @pytest.fixture
    def model_metadata(self):
        """Create test model metadata."""
        return ModelMetadata(
            model_id="test_model_v1",
            model_name="test_sentiment_model",
            version="1.0",
            model_type=ModelType.SENTIMENT_ANALYSIS,
            status=ModelStatus.VALIDATING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="test_user",
            framework="transformers",
            model_path="./test_models/sentiment_v1"
        )
    
    @pytest.mark.asyncio
    async def test_validate_model_accuracy(self, mock_mlops_service, benchmark_dataset, model_metadata):
        """Test model accuracy validation against benchmark dataset."""
        # Mock model predictions
        mock_predictions = [
            {"label": "POSITIVE", "sentiment_score": 0.75, "confidence": 0.9},
            {"label": "NEGATIVE", "sentiment_score": -0.65, "confidence": 0.8},
            {"label": "NEUTRAL", "sentiment_score": 0.05, "confidence": 0.6},
            {"label": "POSITIVE", "sentiment_score": 0.85, "confidence": 0.95},
            {"label": "NEGATIVE", "sentiment_score": -0.55, "confidence": 0.7},
            {"label": "NEUTRAL", "sentiment_score": 0.1, "confidence": 0.5},
            {"label": "POSITIVE", "sentiment_score": 0.65, "confidence": 0.8},
            {"label": "NEGATIVE", "sentiment_score": -0.45, "confidence": 0.75},
            {"label": "NEUTRAL", "sentiment_score": 0.0, "confidence": 0.6},
            {"label": "POSITIVE", "sentiment_score": 0.9, "confidence": 0.98}
        ]
        
        mock_mlops_service.hf_service.batch_predict_sentiment.return_value = mock_predictions
        
        # Perform validation
        validation_results = await self._validate_model_accuracy(
            mock_mlops_service, model_metadata, benchmark_dataset
        )
        
        # Assertions
        assert validation_results["accuracy"] >= 0.8  # Should have good accuracy
        assert validation_results["precision"] > 0.0
        assert validation_results["recall"] > 0.0
        assert validation_results["f1_score"] > 0.0
        assert validation_results["total_samples"] == len(benchmark_dataset)
        assert "confusion_matrix" in validation_results
        assert "classification_report" in validation_results
    
    @pytest.mark.asyncio
    async def test_validate_model_performance_metrics(self, mock_mlops_service, model_metadata):
        """Test model performance metrics validation."""
        # Mock performance data
        mock_performance = {
            "avg_prediction_time_ms": 45.0,
            "throughput_per_second": 100.0,
            "memory_usage_mb": 512.0,
            "cpu_usage_percent": 25.0
        }
        
        # Perform performance validation
        performance_results = await self._validate_model_performance(
            mock_mlops_service, model_metadata, mock_performance
        )
        
        # Assertions
        assert performance_results["avg_prediction_time_ms"] < 100.0  # Should be fast
        assert performance_results["throughput_per_second"] > 50.0  # Should have good throughput
        assert performance_results["memory_usage_mb"] < 1024.0  # Should be memory efficient
        assert performance_results["performance_score"] > 0.7  # Overall good performance
    
    @pytest.mark.asyncio
    async def test_validate_model_robustness(self, mock_mlops_service, model_metadata):
        """Test model robustness with edge cases and adversarial inputs."""
        edge_cases = [
            "",  # Empty string
            "a" * 1000,  # Very long text
            "!@#$%^&*()",  # Special characters only
            "123456789",  # Numbers only
            "ALLCAPS TEXT HERE",  # All caps
            "mixed CaSe TeXt",  # Mixed case
            "Text with émojis 😀 🏈 ⚡",  # Unicode characters
            "Repeated word word word word word",  # Repetitive text
            "Short",  # Very short text
            "Text with\nnewlines\tand\ttabs"  # Special whitespace
        ]
        
        # Mock predictions for edge cases
        mock_predictions = [
            {"label": "NEUTRAL", "sentiment_score": 0.0, "confidence": 0.3}
        ] * len(edge_cases)
        
        mock_mlops_service.hf_service.batch_predict_sentiment.return_value = mock_predictions
        
        # Perform robustness validation
        robustness_results = await self._validate_model_robustness(
            mock_mlops_service, model_metadata, edge_cases
        )
        
        # Assertions
        assert robustness_results["total_edge_cases"] == len(edge_cases)
        assert robustness_results["successful_predictions"] >= 0
        assert robustness_results["error_rate"] <= 0.2  # Should handle most edge cases
        assert "edge_case_performance" in robustness_results
    
    @pytest.mark.asyncio
    async def test_validate_model_bias_detection(self, mock_mlops_service, model_metadata):
        """Test model bias detection across different team contexts."""
        team_contexts = [
            ("patriots", "Great performance by the team"),
            ("cowboys", "Great performance by the team"),
            ("packers", "Great performance by the team"),
            ("steelers", "Great performance by the team"),
            ("49ers", "Great performance by the team")
        ]
        
        # Mock predictions with slight bias
        mock_predictions = [
            {"label": "POSITIVE", "sentiment_score": 0.8, "confidence": 0.9},
            {"label": "POSITIVE", "sentiment_score": 0.75, "confidence": 0.85},
            {"label": "POSITIVE", "sentiment_score": 0.82, "confidence": 0.88},
            {"label": "POSITIVE", "sentiment_score": 0.78, "confidence": 0.87},
            {"label": "POSITIVE", "sentiment_score": 0.79, "confidence": 0.86}
        ]
        
        mock_mlops_service.hf_service.predict_sentiment.side_effect = mock_predictions
        
        # Perform bias validation
        bias_results = await self._validate_model_bias(
            mock_mlops_service, model_metadata, team_contexts
        )
        
        # Assertions
        assert "team_bias_scores" in bias_results
        assert "overall_bias_score" in bias_results
        assert bias_results["overall_bias_score"] <= 0.1  # Should have low bias
        assert len(bias_results["team_bias_scores"]) == len(team_contexts)
    
    async def _validate_model_accuracy(
        self, 
        mlops_service: MLOpsService, 
        model_metadata: ModelMetadata, 
        benchmark_dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate model accuracy against benchmark dataset."""
        texts = [item["text"] for item in benchmark_dataset]
        expected_labels = [item["label"] for item in benchmark_dataset]
        
        # Get predictions
        predictions = await mlops_service.hf_service.batch_predict_sentiment(
            texts=texts,
            model_name=model_metadata.model_name
        )
        
        # Calculate metrics
        predicted_labels = [pred["label"] for pred in predictions]
        
        # Calculate accuracy
        correct = sum(1 for pred, exp in zip(predicted_labels, expected_labels) if pred == exp)
        accuracy = correct / len(expected_labels)
        
        # Calculate precision, recall, F1 for each class
        classes = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        precision_scores = {}
        recall_scores = {}
        f1_scores = {}
        
        for cls in classes:
            tp = sum(1 for pred, exp in zip(predicted_labels, expected_labels) 
                    if pred == cls and exp == cls)
            fp = sum(1 for pred, exp in zip(predicted_labels, expected_labels) 
                    if pred == cls and exp != cls)
            fn = sum(1 for pred, exp in zip(predicted_labels, expected_labels) 
                    if pred != cls and exp == cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precision_scores[cls] = precision
            recall_scores[cls] = recall
            f1_scores[cls] = f1
        
        # Overall metrics
        avg_precision = sum(precision_scores.values()) / len(precision_scores)
        avg_recall = sum(recall_scores.values()) / len(recall_scores)
        avg_f1 = sum(f1_scores.values()) / len(f1_scores)
        
        return {
            "accuracy": accuracy,
            "precision": avg_precision,
            "recall": avg_recall,
            "f1_score": avg_f1,
            "total_samples": len(benchmark_dataset),
            "correct_predictions": correct,
            "class_precision": precision_scores,
            "class_recall": recall_scores,
            "class_f1": f1_scores,
            "confusion_matrix": self._calculate_confusion_matrix(predicted_labels, expected_labels),
            "classification_report": {
                "precision": precision_scores,
                "recall": recall_scores,
                "f1_score": f1_scores
            }
        }
    
    async def _validate_model_performance(
        self,
        mlops_service: MLOpsService,
        model_metadata: ModelMetadata,
        performance_data: Dict[str, float]
    ) -> Dict[str, Any]:
        """Validate model performance metrics."""
        # Performance thresholds
        thresholds = {
            "max_prediction_time_ms": 100.0,
            "min_throughput_per_second": 50.0,
            "max_memory_usage_mb": 1024.0,
            "max_cpu_usage_percent": 50.0
        }
        
        # Calculate performance score
        performance_score = 1.0
        
        if performance_data["avg_prediction_time_ms"] > thresholds["max_prediction_time_ms"]:
            performance_score -= 0.3
        
        if performance_data["throughput_per_second"] < thresholds["min_throughput_per_second"]:
            performance_score -= 0.2
        
        if performance_data["memory_usage_mb"] > thresholds["max_memory_usage_mb"]:
            performance_score -= 0.3
        
        if performance_data["cpu_usage_percent"] > thresholds["max_cpu_usage_percent"]:
            performance_score -= 0.2
        
        performance_score = max(0.0, performance_score)
        
        return {
            **performance_data,
            "performance_score": performance_score,
            "meets_latency_requirement": performance_data["avg_prediction_time_ms"] <= thresholds["max_prediction_time_ms"],
            "meets_throughput_requirement": performance_data["throughput_per_second"] >= thresholds["min_throughput_per_second"],
            "meets_memory_requirement": performance_data["memory_usage_mb"] <= thresholds["max_memory_usage_mb"],
            "meets_cpu_requirement": performance_data["cpu_usage_percent"] <= thresholds["max_cpu_usage_percent"]
        }
    
    async def _validate_model_robustness(
        self,
        mlops_service: MLOpsService,
        model_metadata: ModelMetadata,
        edge_cases: List[str]
    ) -> Dict[str, Any]:
        """Validate model robustness with edge cases."""
        successful_predictions = 0
        failed_predictions = 0
        edge_case_results = []
        
        for i, text in enumerate(edge_cases):
            try:
                prediction = await mlops_service.hf_service.predict_sentiment(
                    text=text,
                    model_name=model_metadata.model_name
                )
                
                # Check if prediction is valid
                if (prediction and 
                    "label" in prediction and 
                    "sentiment_score" in prediction and 
                    "confidence" in prediction):
                    successful_predictions += 1
                    edge_case_results.append({
                        "text": text[:50] + "..." if len(text) > 50 else text,
                        "prediction": prediction,
                        "success": True
                    })
                else:
                    failed_predictions += 1
                    edge_case_results.append({
                        "text": text[:50] + "..." if len(text) > 50 else text,
                        "error": "Invalid prediction format",
                        "success": False
                    })
                    
            except Exception as e:
                failed_predictions += 1
                edge_case_results.append({
                    "text": text[:50] + "..." if len(text) > 50 else text,
                    "error": str(e),
                    "success": False
                })
        
        error_rate = failed_predictions / len(edge_cases)
        
        return {
            "total_edge_cases": len(edge_cases),
            "successful_predictions": successful_predictions,
            "failed_predictions": failed_predictions,
            "error_rate": error_rate,
            "robustness_score": 1.0 - error_rate,
            "edge_case_performance": edge_case_results
        }
    
    async def _validate_model_bias(
        self,
        mlops_service: MLOpsService,
        model_metadata: ModelMetadata,
        team_contexts: List[tuple]
    ) -> Dict[str, Any]:
        """Validate model for bias across different team contexts."""
        team_scores = {}
        
        for team, text in team_contexts:
            prediction = await mlops_service.hf_service.predict_sentiment(
                text=text,
                model_name=model_metadata.model_name
            )
            
            team_scores[team] = {
                "sentiment_score": prediction["sentiment_score"],
                "confidence": prediction["confidence"],
                "label": prediction["label"]
            }
        
        # Calculate bias metrics
        sentiment_scores = [scores["sentiment_score"] for scores in team_scores.values()]
        confidence_scores = [scores["confidence"] for scores in team_scores.values()]
        
        # Calculate standard deviation as bias measure
        sentiment_std = pd.Series(sentiment_scores).std()
        confidence_std = pd.Series(confidence_scores).std()
        
        # Overall bias score (lower is better)
        overall_bias_score = (sentiment_std + confidence_std) / 2
        
        return {
            "team_bias_scores": team_scores,
            "sentiment_score_std": sentiment_std,
            "confidence_score_std": confidence_std,
            "overall_bias_score": overall_bias_score,
            "bias_threshold_met": overall_bias_score <= 0.1
        }
    
    def _calculate_confusion_matrix(self, predicted: List[str], expected: List[str]) -> Dict[str, Dict[str, int]]:
        """Calculate confusion matrix."""
        classes = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        matrix = {cls: {cls2: 0 for cls2 in classes} for cls in classes}
        
        for pred, exp in zip(predicted, expected):
            if pred in classes and exp in classes:
                matrix[exp][pred] += 1
        
        return matrix


class TestModelDeploymentAndRollback:
    """Test model deployment and rollback procedures."""
    
    @pytest.fixture
    def deployment_service(self):
        """Mock deployment service."""
        service = ModelDeploymentService()
        service.db = MagicMock()
        return service
    
    @pytest.fixture
    def model_metadata(self):
        """Create test model metadata."""
        return ModelMetadata(
            model_id="test_model_v2",
            model_name="test_sentiment_model",
            version="2.0",
            model_type=ModelType.SENTIMENT_ANALYSIS,
            status=ModelStatus.VALIDATING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="test_user",
            framework="transformers",
            model_path="./test_models/sentiment_v2"
        )
    
    @pytest.mark.asyncio
    async def test_immediate_deployment(self, deployment_service, model_metadata):
        """Test immediate deployment strategy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup temporary model path
            model_metadata.model_path = temp_dir
            deployment_service.deployments_dir = Path(temp_dir) / "deployments"
            deployment_service.deployments_dir.mkdir(exist_ok=True)
            
            # Deploy model
            deployment = await deployment_service.deploy_model(
                model_metadata=model_metadata,
                environment="staging",
                strategy=DeploymentStrategy.IMMEDIATE,
                traffic_percentage=100.0
            )
            
            # Assertions
            assert deployment.model_id == model_metadata.model_id
            assert deployment.model_version == model_metadata.version
            assert deployment.environment == "staging"
            assert deployment.strategy == DeploymentStrategy.IMMEDIATE
            assert deployment.traffic_percentage == 100.0
            assert deployment.status == ModelStatus.DEPLOYED
    
    @pytest.mark.asyncio
    async def test_blue_green_deployment(self, deployment_service, model_metadata):
        """Test blue-green deployment strategy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_metadata.model_path = temp_dir
            deployment_service.deployments_dir = Path(temp_dir) / "deployments"
            deployment_service.deployments_dir.mkdir(exist_ok=True)
            
            deployment = await deployment_service.deploy_model(
                model_metadata=model_metadata,
                environment="production",
                strategy=DeploymentStrategy.BLUE_GREEN,
                traffic_percentage=100.0
            )
            
            assert deployment.strategy == DeploymentStrategy.BLUE_GREEN
            assert deployment.status == ModelStatus.DEPLOYED
    
    @pytest.mark.asyncio
    async def test_canary_deployment(self, deployment_service, model_metadata):
        """Test canary deployment strategy."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_metadata.model_path = temp_dir
            deployment_service.deployments_dir = Path(temp_dir) / "deployments"
            deployment_service.deployments_dir.mkdir(exist_ok=True)
            
            deployment = await deployment_service.deploy_model(
                model_metadata=model_metadata,
                environment="production",
                strategy=DeploymentStrategy.CANARY,
                traffic_percentage=10.0
            )
            
            assert deployment.strategy == DeploymentStrategy.CANARY
            assert deployment.traffic_percentage == 10.0
            assert deployment.status == ModelStatus.DEPLOYED
    
    @pytest.mark.asyncio
    async def test_ab_test_deployment(self, deployment_service):
        """Test A/B test deployment creation."""
        # Create two model versions
        model_a = ModelMetadata(
            model_id="test_model_a",
            model_name="test_sentiment_model",
            version="1.0",
            model_type=ModelType.SENTIMENT_ANALYSIS,
            status=ModelStatus.DEPLOYED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="test_user",
            framework="transformers",
            model_path="./test_models/sentiment_v1"
        )
        
        model_b = ModelMetadata(
            model_id="test_model_b",
            model_name="test_sentiment_model",
            version="2.0",
            model_type=ModelType.SENTIMENT_ANALYSIS,
            status=ModelStatus.DEPLOYED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="test_user",
            framework="transformers",
            model_path="./test_models/sentiment_v2"
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            deployment_service.deployments_dir = Path(temp_dir) / "deployments"
            deployment_service.deployments_dir.mkdir(exist_ok=True)
            
            # Mock the deploy_model method to avoid file operations
            deployment_service.deploy_model = AsyncMock(return_value=ModelDeployment(
                deployment_id="test_deployment",
                model_id="test_model",
                model_version="1.0",
                environment="staging",
                strategy=DeploymentStrategy.A_B_TEST,
                status=ModelStatus.DEPLOYED,
                traffic_percentage=50.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            ))
            
            ab_test = await deployment_service.create_ab_test(
                test_name="sentiment_model_ab_test",
                model_a_metadata=model_a,
                model_b_metadata=model_b,
                traffic_split_percentage=50.0,
                test_duration_days=7
            )
            
            assert ab_test.test_name == "sentiment_model_ab_test"
            assert ab_test.model_a_id == model_a.model_id
            assert ab_test.model_b_id == model_b.model_id
            assert ab_test.traffic_split_percentage == 50.0
            assert ab_test.test_duration_days == 7
            assert ab_test.status == "running"
    
    @pytest.mark.asyncio
    async def test_deployment_rollback(self, deployment_service):
        """Test deployment rollback functionality."""
        # Create a deployment
        deployment = ModelDeployment(
            deployment_id="test_deployment_rollback",
            model_id="test_model",
            model_version="2.0",
            environment="production",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        deployment_service.active_deployments[deployment.deployment_id] = deployment
        
        # Mock methods
        deployment_service._get_previous_stable_version = AsyncMock(return_value="1.0")
        deployment_service._get_model_metadata = AsyncMock(return_value=ModelMetadata(
            model_id="test_model",
            model_name="test_model",
            version="1.0",
            model_type=ModelType.SENTIMENT_ANALYSIS,
            status=ModelStatus.DEPLOYED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="test_user",
            framework="transformers",
            model_path="./test_models/sentiment_v1"
        ))
        deployment_service._execute_rollback = AsyncMock(return_value=True)
        deployment_service._store_deployment = AsyncMock()
        
        # Perform rollback
        success = await deployment_service.rollback_deployment(
            deployment_id=deployment.deployment_id,
            reason="Performance degradation detected",
            target_version="1.0"
        )
        
        assert success is True
        assert deployment.rollback_version == "1.0"
        assert deployment.rollback_reason == "Performance degradation detected"
    
    @pytest.mark.asyncio
    async def test_deployment_health_monitoring(self, deployment_service):
        """Test deployment health monitoring."""
        deployment = ModelDeployment(
            deployment_id="test_deployment_health",
            model_id="test_model",
            model_version="1.0",
            environment="production",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        deployment_service.active_deployments[deployment.deployment_id] = deployment
        
        # Mock health check
        deployment_service._check_deployment_health = AsyncMock(return_value="healthy")
        deployment_service._get_recent_alerts = AsyncMock(return_value=[])
        
        status = await deployment_service.get_deployment_status(deployment.deployment_id)
        
        assert status is not None
        assert status["deployment_id"] == deployment.deployment_id
        assert status["health_status"] == "healthy"
        assert status["status"] == ModelStatus.DEPLOYED.value


class TestDataDriftDetection:
    """Test data drift detection and model performance monitoring."""
    
    @pytest.fixture
    def performance_metrics(self):
        """Create test performance metrics."""
        return ModelPerformanceMetrics(
            model_id="test_model",
            model_version="1.0",
            timestamp=datetime.utcnow(),
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            auc_roc=0.92,
            prediction_count=1000,
            avg_prediction_time_ms=45.0,
            error_rate=0.02,
            throughput_per_second=100.0,
            data_drift_score=0.05,
            prediction_drift_score=0.03,
            period_start=datetime.utcnow() - timedelta(hours=1),
            period_end=datetime.utcnow(),
            sample_size=1000,
            custom_metrics={"nfl_accuracy": 0.88}
        )
    
    @pytest.mark.asyncio
    async def test_data_drift_detection_normal(self, performance_metrics):
        """Test data drift detection with normal drift levels."""
        deployment_service = ModelDeploymentService()
        
        # Create deployment
        deployment = ModelDeployment(
            deployment_id="test_drift_normal",
            model_id="test_model",
            model_version="1.0",
            environment="production",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Monitor performance (should not generate alerts)
        alerts = await deployment_service.monitor_model_performance(
            deployment_id=deployment.deployment_id,
            metrics=performance_metrics
        )
        
        # Should not generate data drift alerts for normal levels
        data_drift_alerts = [alert for alert in alerts if alert.alert_type == "data_drift"]
        assert len(data_drift_alerts) == 0
    
    @pytest.mark.asyncio
    async def test_data_drift_detection_high_drift(self, performance_metrics):
        """Test data drift detection with high drift levels."""
        deployment_service = ModelDeploymentService()
        
        # Set high drift score
        performance_metrics.data_drift_score = 0.15  # Above threshold of 0.1
        
        deployment = ModelDeployment(
            deployment_id="test_drift_high",
            model_id="test_model",
            model_version="1.0",
            environment="production",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        deployment_service.active_deployments[deployment.deployment_id] = deployment
        deployment_service._store_alert = AsyncMock()
        
        alerts = await deployment_service.monitor_model_performance(
            deployment_id=deployment.deployment_id,
            metrics=performance_metrics
        )
        
        # Should generate data drift alert
        data_drift_alerts = [alert for alert in alerts if alert.alert_type == "data_drift"]
        assert len(data_drift_alerts) > 0
        assert data_drift_alerts[0].severity == "high"
        assert data_drift_alerts[0].actual_value == 0.15
    
    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self, performance_metrics):
        """Test performance degradation detection."""
        deployment_service = ModelDeploymentService()
        
        # Set low accuracy (below threshold)
        performance_metrics.accuracy = 0.75  # Below threshold of 0.8
        performance_metrics.f1_score = 0.70  # Below threshold of 0.75
        
        deployment = ModelDeployment(
            deployment_id="test_perf_degradation",
            model_id="test_model",
            model_version="1.0",
            environment="production",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        deployment_service.active_deployments[deployment.deployment_id] = deployment
        deployment_service._store_alert = AsyncMock()
        
        alerts = await deployment_service.monitor_model_performance(
            deployment_id=deployment.deployment_id,
            metrics=performance_metrics
        )
        
        # Should generate performance degradation alerts
        perf_alerts = [alert for alert in alerts if alert.alert_type == "performance_degradation"]
        assert len(perf_alerts) >= 2  # One for accuracy, one for f1_score
    
    @pytest.mark.asyncio
    async def test_error_rate_spike_detection(self, performance_metrics):
        """Test error rate spike detection."""
        deployment_service = ModelDeploymentService()
        
        # Set high error rate
        performance_metrics.error_rate = 0.08  # Above threshold of 0.05
        
        deployment = ModelDeployment(
            deployment_id="test_error_spike",
            model_id="test_model",
            model_version="1.0",
            environment="production",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        deployment_service.active_deployments[deployment.deployment_id] = deployment
        deployment_service._store_alert = AsyncMock()
        
        alerts = await deployment_service.monitor_model_performance(
            deployment_id=deployment.deployment_id,
            metrics=performance_metrics
        )
        
        # Should generate error spike alert
        error_alerts = [alert for alert in alerts if alert.alert_type == "error_spike"]
        assert len(error_alerts) > 0
        assert error_alerts[0].severity == "high"
        assert error_alerts[0].actual_value == 0.08


class TestMLOpsPipelineIntegration:
    """Test integration of MLOps pipeline components."""
    
    @pytest.fixture
    def mlops_service(self):
        """Mock MLOps service with all components."""
        service = MLOpsService()
        service.hf_service = MagicMock()
        service.wandb_service = MagicMock()
        service.hopsworks_service = MagicMock()
        service.deployment_service = MagicMock()
        service.retraining_service = MagicMock()
        service.initialized = True
        return service
    
    @pytest.mark.asyncio
    async def test_end_to_end_model_lifecycle(self, mlops_service):
        """Test complete model lifecycle from training to deployment."""
        # 1. Start experiment
        experiment_run = ExperimentRun(
            experiment_id="test_experiment",
            run_id="test_run_123",
            experiment_name="nfl_sentiment_training",
            run_name="test_training_run",
            status=ExperimentStatus.RUNNING,
            model_type=ModelType.SENTIMENT_ANALYSIS,
            hyperparameters={"learning_rate": 2e-5, "batch_size": 16},
            started_at=datetime.utcnow()
        )
        
        mlops_service.wandb_service.start_experiment.return_value = experiment_run
        
        run_id = await mlops_service.start_experiment(
            experiment_name="nfl_sentiment_training",
            config={"learning_rate": 2e-5, "batch_size": 16}
        )
        
        assert run_id == experiment_run.run_id
        
        # 2. Register model
        model_metadata = ModelMetadata(
            model_id="nfl_sentiment_v1",
            model_name="nfl_sentiment_model",
            version="1.0",
            model_type=ModelType.SENTIMENT_ANALYSIS,
            status=ModelStatus.VALIDATING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            created_by="test_user",
            framework="transformers",
            model_path="./models/nfl_sentiment_v1"
        )
        
        mlops_service.hf_service.register_model.return_value = model_metadata
        
        registered_model = await mlops_service.register_model(
            model_name="nfl_sentiment_model",
            model_path="./models/nfl_sentiment_v1",
            metrics={"accuracy": 0.85, "f1_score": 0.83}
        )
        
        assert registered_model.model_id == model_metadata.model_id
        
        # 3. Deploy model
        deployment = ModelDeployment(
            deployment_id="test_deployment_123",
            model_id=model_metadata.model_id,
            model_version=model_metadata.version,
            environment="production",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        mlops_service.deployment_service.deploy_model.return_value = deployment
        
        deployed_model = await mlops_service.deploy_model(
            model_id=model_metadata.model_id,
            model_version=model_metadata.version,
            environment="production"
        )
        
        assert deployed_model.deployment_id == deployment.deployment_id
        
        # 4. Monitor performance
        performance_metrics = ModelPerformanceMetrics(
            model_id=model_metadata.model_id,
            model_version=model_metadata.version,
            timestamp=datetime.utcnow(),
            accuracy=0.82,  # Slight degradation
            precision=0.80,
            recall=0.84,
            f1_score=0.82,
            prediction_count=5000,
            avg_prediction_time_ms=50.0,
            error_rate=0.03,
            period_start=datetime.utcnow() - timedelta(hours=1),
            period_end=datetime.utcnow(),
            sample_size=5000
        )
        
        await mlops_service.monitor_model_performance(
            model_id=model_metadata.model_id,
            model_version=model_metadata.version,
            performance_metrics=performance_metrics
        )
        
        # Verify monitoring was called
        mlops_service.deployment_service.monitor_model_performance.assert_called_once()
        mlops_service.retraining_service.monitor_model_performance.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_automated_retraining_trigger(self, mlops_service):
        """Test automated retraining trigger based on performance degradation."""
        # Setup retraining service
        retraining_service = ModelRetrainingService()
        retraining_service.retraining_configs = {
            "config_1": ModelRetrainingConfig(
                config_id="config_1",
                model_name="nfl_sentiment_model",
                performance_threshold=0.8,
                data_drift_threshold=0.1,
                enabled=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
        }
        
        # Mock trigger check to return True
        retraining_service.check_retraining_triggers = AsyncMock(return_value={
            "should_retrain": True,
            "triggers_activated": ["performance_degradation"],
            "config_id": "config_1"
        })
        
        # Mock trigger retraining
        experiment_run = ExperimentRun(
            experiment_id="retraining_experiment",
            run_id="retrain_run_123",
            experiment_name="nfl_sentiment_model_retraining",
            run_name="retrain_20240115_120000",
            status=ExperimentStatus.RUNNING,
            model_type=ModelType.SENTIMENT_ANALYSIS,
            started_at=datetime.utcnow()
        )
        
        retraining_service.trigger_retraining = AsyncMock(return_value=experiment_run)
        
        mlops_service.retraining_service = retraining_service
        
        # Trigger retraining check
        trigger_result = await mlops_service.retraining_service.check_retraining_triggers(
            "nfl_sentiment_model"
        )
        
        assert trigger_result["should_retrain"] is True
        assert "performance_degradation" in trigger_result["triggers_activated"]
        
        # Trigger retraining
        if trigger_result["should_retrain"]:
            retraining_run = await mlops_service.trigger_retraining(
                model_name="nfl_sentiment_model",
                trigger_reason="performance_degradation"
            )
            
            assert retraining_run.run_id == experiment_run.run_id
    
    @pytest.mark.asyncio
    async def test_feature_store_integration(self, mlops_service):
        """Test feature store integration for training data."""
        # Mock Hopsworks service
        hopsworks_service = HopsworksService()
        
        # Mock sentiment results
        sentiment_results = [
            SentimentResult(
                text="Great touchdown pass!",
                sentiment=SentimentLabel.POSITIVE,
                sentiment_score=0.8,
                confidence=0.9,
                category="performance",
                context={},
                source=DataSource.TWITTER,
                timestamp=datetime.utcnow(),
                team_id="patriots",
                model_version="1.0",
                processing_time_ms=50.0
            ),
            SentimentResult(
                text="Terrible fumble",
                sentiment=SentimentLabel.NEGATIVE,
                sentiment_score=-0.7,
                confidence=0.85,
                category="performance",
                context={},
                source=DataSource.TWITTER,
                timestamp=datetime.utcnow(),
                team_id="cowboys",
                model_version="1.0",
                processing_time_ms=45.0
            )
        ]
        
        # Mock feature update
        hopsworks_service.update_sentiment_features_from_results = AsyncMock(return_value=True)
        mlops_service.hopsworks_service = hopsworks_service
        
        # Update features
        success = await mlops_service.update_features_from_sentiment(sentiment_results)
        
        assert success is True
        hopsworks_service.update_sentiment_features_from_results.assert_called_once_with(
            sentiment_results
        )
    
    @pytest.mark.asyncio
    async def test_experiment_tracking_integration(self, mlops_service):
        """Test experiment tracking integration with W&B."""
        # Mock W&B service
        wandb_service = WandBService()
        
        # Mock experiment operations
        experiment_run = ExperimentRun(
            experiment_id="test_experiment",
            run_id="test_run_456",
            experiment_name="model_evaluation",
            run_name="evaluation_run",
            status=ExperimentStatus.RUNNING,
            model_type=ModelType.SENTIMENT_ANALYSIS,
            started_at=datetime.utcnow()
        )
        
        wandb_service.start_experiment = AsyncMock(return_value=experiment_run)
        wandb_service.log_metrics = AsyncMock()
        wandb_service.finish_experiment = AsyncMock(return_value=experiment_run)
        
        mlops_service.wandb_service = wandb_service
        
        # Start experiment
        run_id = await mlops_service.start_experiment(
            experiment_name="model_evaluation",
            config={"test_param": 1.0}
        )
        
        assert run_id == experiment_run.run_id
        
        # Log metrics
        await mlops_service.log_metrics({"accuracy": 0.85, "loss": 0.15})
        wandb_service.log_metrics.assert_called_once_with({"accuracy": 0.85, "loss": 0.15}, step=None)
        
        # Finish experiment
        final_experiment = await mlops_service.finish_experiment({"final_accuracy": 0.87})
        wandb_service.finish_experiment.assert_called_once()