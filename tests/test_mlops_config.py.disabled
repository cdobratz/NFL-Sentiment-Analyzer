"""
MLOps testing configuration and utilities.
"""

import pytest
import tempfile
import shutil
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

from app.models.mlops import (
    ModelMetadata, ModelDeployment, ModelPerformanceMetrics, ModelAlert,
    ModelRetrainingConfig, ExperimentRun, ABTestConfig,
    ModelType, ModelStatus, DeploymentStrategy, ExperimentStatus
)
from app.models.sentiment import SentimentResult, SentimentLabel, DataSource


class MLOpsTestConfig:
    """Configuration for MLOps testing."""
    
    # Test model configurations
    TEST_MODELS = {
        "sentiment_base": {
            "model_id": "test_sentiment_base",
            "model_name": "sentiment_base_model",
            "version": "1.0",
            "accuracy": 0.85,
            "f1_score": 0.83,
            "model_path": "./test_models/sentiment_base"
        },
        "sentiment_enhanced": {
            "model_id": "test_sentiment_enhanced",
            "model_name": "sentiment_enhanced_model",
            "version": "2.0",
            "accuracy": 0.87,
            "f1_score": 0.85,
            "model_path": "./test_models/sentiment_enhanced"
        }
    }
    
    # Test deployment configurations
    TEST_DEPLOYMENTS = {
        "staging": {
            "environment": "staging",
            "strategy": DeploymentStrategy.IMMEDIATE,
            "traffic_percentage": 100.0
        },
        "production_canary": {
            "environment": "production",
            "strategy": DeploymentStrategy.CANARY,
            "traffic_percentage": 10.0
        },
        "production_blue_green": {
            "environment": "production",
            "strategy": DeploymentStrategy.BLUE_GREEN,
            "traffic_percentage": 100.0
        }
    }
    
    # Performance thresholds for testing
    PERFORMANCE_THRESHOLDS = {
        "accuracy": 0.8,
        "precision": 0.75,
        "recall": 0.75,
        "f1_score": 0.75,
        "error_rate": 0.05,
        "avg_prediction_time_ms": 100.0,
        "data_drift_score": 0.1
    }
    
    # Test data configurations
    BENCHMARK_DATASETS = {
        "small": 100,
        "medium": 1000,
        "large": 10000
    }


@pytest.fixture(scope="session")
def mlops_test_config():
    """Provide MLOps test configuration."""
    return MLOpsTestConfig()


@pytest.fixture
def temp_model_directory():
    """Create temporary directory for model files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_model_metadata():
    """Create mock model metadata for testing."""
    return ModelMetadata(
        model_id="test_model_123",
        model_name="test_sentiment_model",
        version="1.0",
        model_type=ModelType.SENTIMENT_ANALYSIS,
        status=ModelStatus.VALIDATING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        created_by="test_user",
        framework="transformers",
        model_path="./test_models/sentiment_v1",
        accuracy=0.85,
        precision=0.83,
        recall=0.87,
        f1_score=0.85,
        training_dataset="test_dataset_v1",
        validation_dataset="test_val_dataset_v1",
        epochs=3,
        learning_rate=2e-5,
        batch_size=16,
        feature_importance={"sentiment_keywords": 0.4, "context": 0.3, "length": 0.3},
        custom_metrics={"nfl_accuracy": 0.88, "team_bias_score": 0.02},
        tags=["test", "sentiment", "nfl"],
        description="Test sentiment analysis model for NFL content"
    )


@pytest.fixture
def mock_deployment():
    """Create mock deployment for testing."""
    return ModelDeployment(
        deployment_id="test_deployment_456",
        model_id="test_model_123",
        model_version="1.0",
        environment="staging",
        strategy=DeploymentStrategy.IMMEDIATE,
        status=ModelStatus.DEPLOYED,
        traffic_percentage=100.0,
        resource_limits={"cpu": "2", "memory": "4Gi", "replicas": 2},
        health_check_url="/health/test_deployment_456",
        monitoring_enabled=True,
        alerts_enabled=True,
        deployed_at=datetime.utcnow(),
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


@pytest.fixture
def mock_performance_metrics():
    """Create mock performance metrics for testing."""
    return ModelPerformanceMetrics(
        model_id="test_model_123",
        model_version="1.0",
        timestamp=datetime.utcnow(),
        accuracy=0.85,
        precision=0.83,
        recall=0.87,
        f1_score=0.85,
        auc_roc=0.92,
        prediction_count=5000,
        avg_prediction_time_ms=45.0,
        error_rate=0.02,
        throughput_per_second=120.0,
        data_drift_score=0.03,
        prediction_drift_score=0.02,
        period_start=datetime.utcnow() - timedelta(hours=24),
        period_end=datetime.utcnow(),
        sample_size=5000,
        custom_metrics={
            "nfl_specific_accuracy": 0.88,
            "team_bias_score": 0.02,
            "position_accuracy": 0.86
        }
    )


@pytest.fixture
def mock_experiment_run():
    """Create mock experiment run for testing."""
    return ExperimentRun(
        experiment_id="test_experiment_789",
        run_id="test_run_101112",
        experiment_name="nfl_sentiment_training",
        run_name="baseline_training_v1",
        status=ExperimentStatus.RUNNING,
        model_type=ModelType.SENTIMENT_ANALYSIS,
        hyperparameters={
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 3,
            "max_length": 512,
            "warmup_steps": 500
        },
        started_at=datetime.utcnow(),
        tags=["baseline", "nfl", "sentiment", "test"],
        notes="Test experiment for NFL sentiment analysis model training",
        metrics={
            "train_loss": 0.25,
            "val_loss": 0.27,
            "val_accuracy": 0.85,
            "val_f1": 0.83
        }
    )


@pytest.fixture
def mock_retraining_config():
    """Create mock retraining configuration for testing."""
    return ModelRetrainingConfig(
        config_id="test_retrain_config_131415",
        model_name="test_sentiment_model",
        performance_threshold=0.8,
        data_drift_threshold=0.1,
        time_based_trigger="0 2 * * 0",  # Weekly on Sunday at 2 AM
        data_volume_trigger=10000,
        training_config={
            "learning_rate": 1e-5,
            "batch_size": 16,
            "num_epochs": 2,
            "early_stopping_patience": 3
        },
        auto_deploy=False,
        approval_required=True,
        approvers=["admin@example.com", "ml_engineer@example.com"],
        enabled=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


@pytest.fixture
def mock_ab_test_config():
    """Create mock A/B test configuration for testing."""
    return ABTestConfig(
        test_id="test_ab_161718",
        test_name="sentiment_v1_vs_v2",
        model_a_id="test_model_v1",
        model_a_version="1.0",
        model_b_id="test_model_v2",
        model_b_version="2.0",
        traffic_split_percentage=50.0,
        success_metrics=["accuracy", "f1_score", "latency", "user_satisfaction"],
        minimum_sample_size=5000,
        test_duration_days=14,
        status="running",
        started_at=datetime.utcnow(),
        model_a_performance={
            "accuracy": 0.85,
            "f1_score": 0.83,
            "avg_latency_ms": 45.0,
            "sample_size": 2500
        },
        model_b_performance={
            "accuracy": 0.87,
            "f1_score": 0.85,
            "avg_latency_ms": 48.0,
            "sample_size": 2500
        }
    )


@pytest.fixture
def mock_sentiment_results():
    """Create mock sentiment analysis results for testing."""
    return [
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
            aspect_sentiments={"performance": 0.85, "coaching": 0.1, "execution": 0.8}
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
            aspect_sentiments={"performance": -0.78, "execution": -0.65, "coaching": -0.2}
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
            aspect_sentiments={"injury": -0.45, "availability": -0.55, "performance": -0.2}
        ),
        SentimentResult(
            text="Game scheduled for Sunday afternoon",
            sentiment=SentimentLabel.NEUTRAL,
            sentiment_score=0.02,
            confidence=0.65,
            category="general",
            context={"nfl_context": {"team_mentions": ["steelers", "ravens"]}},
            source=DataSource.ESPN,
            timestamp=datetime.utcnow(),
            game_id="game_456",
            model_version="1.0",
            processing_time_ms=35.0,
            aspect_sentiments={"scheduling": 0.02, "anticipation": 0.1}
        )
    ]


@pytest.fixture
def benchmark_dataset_small():
    """Create small benchmark dataset for testing."""
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
def mock_training_data():
    """Create mock training data for testing."""
    np.random.seed(42)  # For reproducible results
    
    # Generate synthetic training data
    texts = [
        "Great touchdown pass!",
        "Terrible interception",
        "Game starts at 1 PM",
        "Amazing defensive play",
        "Poor coaching decision",
        "Weather looks good",
        "Incredible rushing yards",
        "Missed field goal",
        "Halftime show scheduled",
        "Outstanding quarterback performance"
    ]
    
    labels = ["POSITIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "NEGATIVE", 
              "NEUTRAL", "POSITIVE", "NEGATIVE", "NEUTRAL", "POSITIVE"]
    
    features = np.random.rand(len(texts), 10)  # 10 synthetic features
    
    return {
        "texts": texts,
        "labels": labels,
        "features": features,
        "metadata": {
            "dataset_size": len(texts),
            "feature_count": features.shape[1],
            "label_distribution": {
                "POSITIVE": labels.count("POSITIVE"),
                "NEGATIVE": labels.count("NEGATIVE"),
                "NEUTRAL": labels.count("NEUTRAL")
            }
        }
    }


@pytest.fixture
def mock_drift_data():
    """Create mock data for drift detection testing."""
    np.random.seed(42)
    
    # Baseline data (training distribution)
    baseline_data = {
        "sentiment_scores": np.random.normal(0.1, 0.3, 1000),
        "confidence_scores": np.random.beta(2, 2, 1000),
        "text_lengths": np.random.lognormal(3, 0.5, 1000),
        "keyword_counts": np.random.poisson(3, 1000),
        "team_distribution": np.random.choice(['patriots', 'cowboys', 'packers'], 1000),
        "source_distribution": np.random.choice(['twitter', 'espn', 'news'], 1000, p=[0.6, 0.3, 0.1])
    }
    
    # Current data with no drift
    current_data_normal = {
        "sentiment_scores": np.random.normal(0.12, 0.31, 500),
        "confidence_scores": np.random.beta(2.1, 2.1, 500),
        "text_lengths": np.random.lognormal(3.05, 0.52, 500),
        "keyword_counts": np.random.poisson(3.1, 500),
        "team_distribution": np.random.choice(['patriots', 'cowboys', 'packers'], 500),
        "source_distribution": np.random.choice(['twitter', 'espn', 'news'], 500, p=[0.58, 0.32, 0.1])
    }
    
    # Current data with significant drift
    current_data_drift = {
        "sentiment_scores": np.random.normal(0.3, 0.4, 500),  # Shifted distribution
        "confidence_scores": np.random.beta(3, 1.5, 500),     # Different parameters
        "text_lengths": np.random.lognormal(2.5, 0.8, 500),   # Different parameters
        "keyword_counts": np.random.poisson(5, 500),          # Higher rate
        "team_distribution": np.random.choice(['patriots', 'cowboys', 'packers'], 500, p=[0.8, 0.1, 0.1]),  # Biased
        "source_distribution": np.random.choice(['twitter', 'espn', 'news'], 500, p=[0.3, 0.5, 0.2])  # Different distribution
    }
    
    return {
        "baseline": baseline_data,
        "current_normal": current_data_normal,
        "current_drift": current_data_drift
    }


class MockMLOpsServices:
    """Mock MLOps services for testing."""
    
    @staticmethod
    def create_mock_huggingface_service():
        """Create mock HuggingFace service."""
        service = MagicMock()
        
        # Mock prediction methods
        service.predict_sentiment = AsyncMock(return_value={
            "label": "POSITIVE",
            "sentiment_score": 0.8,
            "confidence": 0.9
        })
        
        service.batch_predict_sentiment = AsyncMock(return_value=[
            {"label": "POSITIVE", "sentiment_score": 0.8, "confidence": 0.9},
            {"label": "NEGATIVE", "sentiment_score": -0.7, "confidence": 0.85}
        ])
        
        service.register_model = AsyncMock()
        service.list_available_models = AsyncMock(return_value=[])
        service.get_cache_info = MagicMock(return_value={"cache_size": 0})
        
        return service
    
    @staticmethod
    def create_mock_wandb_service():
        """Create mock W&B service."""
        service = MagicMock()
        
        # Mock experiment methods
        service.start_experiment = AsyncMock(return_value=ExperimentRun(
            experiment_id="mock_experiment",
            run_id="mock_run_123",
            experiment_name="mock_training",
            run_name="mock_run",
            status=ExperimentStatus.RUNNING,
            model_type=ModelType.SENTIMENT_ANALYSIS,
            started_at=datetime.utcnow()
        ))
        
        service.log_metrics = AsyncMock()
        service.finish_experiment = AsyncMock()
        service.get_service_status = MagicMock(return_value={"authenticated": True})
        
        return service
    
    @staticmethod
    def create_mock_hopsworks_service():
        """Create mock Hopsworks service."""
        service = MagicMock()
        
        # Mock feature store methods
        service.update_sentiment_features_from_results = AsyncMock(return_value=True)
        service.get_training_data = AsyncMock(return_value=([], []))
        service.get_service_status = MagicMock(return_value={"connected": True})
        
        return service
    
    @staticmethod
    def create_mock_deployment_service():
        """Create mock deployment service."""
        service = MagicMock()
        
        # Mock deployment methods
        service.deploy_model = AsyncMock(return_value=ModelDeployment(
            deployment_id="mock_deployment",
            model_id="mock_model",
            model_version="1.0",
            environment="staging",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        ))
        
        service.monitor_model_performance = AsyncMock(return_value=[])
        service.rollback_deployment = AsyncMock(return_value=True)
        service.list_active_deployments = AsyncMock(return_value=[])
        service.initialize = AsyncMock()
        
        return service
    
    @staticmethod
    def create_mock_retraining_service():
        """Create mock retraining service."""
        service = MagicMock()
        
        # Mock retraining methods
        service.check_retraining_triggers = AsyncMock(return_value={
            "should_retrain": False,
            "triggers_activated": [],
            "config_id": None
        })
        
        service.trigger_retraining = AsyncMock(return_value=ExperimentRun(
            experiment_id="mock_retraining",
            run_id="mock_retrain_123",
            experiment_name="mock_retraining",
            run_name="mock_retrain_run",
            status=ExperimentStatus.RUNNING,
            model_type=ModelType.SENTIMENT_ANALYSIS,
            started_at=datetime.utcnow()
        ))
        
        service.monitor_model_performance = AsyncMock()
        service.initialize = AsyncMock()
        
        return service


@pytest.fixture
def mock_mlops_services():
    """Provide mock MLOps services."""
    return MockMLOpsServices()


class TestDataGenerator:
    """Generate test data for MLOps testing."""
    
    @staticmethod
    def generate_performance_metrics_series(
        model_id: str,
        model_version: str,
        days: int = 7,
        base_accuracy: float = 0.85,
        trend: str = "stable"  # "stable", "improving", "degrading"
    ) -> List[ModelPerformanceMetrics]:
        """Generate a series of performance metrics over time."""
        metrics_series = []
        base_time = datetime.utcnow() - timedelta(days=days)
        
        for day in range(days):
            # Apply trend
            if trend == "improving":
                accuracy = base_accuracy + (day * 0.01)
                error_rate = max(0.01, 0.05 - (day * 0.005))
            elif trend == "degrading":
                accuracy = base_accuracy - (day * 0.01)
                error_rate = 0.05 + (day * 0.005)
            else:  # stable
                accuracy = base_accuracy + np.random.normal(0, 0.01)
                error_rate = 0.05 + np.random.normal(0, 0.005)
            
            metrics = ModelPerformanceMetrics(
                model_id=model_id,
                model_version=model_version,
                timestamp=base_time + timedelta(days=day),
                accuracy=max(0.0, min(1.0, accuracy)),
                precision=max(0.0, min(1.0, accuracy - 0.02)),
                recall=max(0.0, min(1.0, accuracy + 0.02)),
                f1_score=max(0.0, min(1.0, accuracy - 0.01)),
                prediction_count=np.random.randint(1000, 5000),
                avg_prediction_time_ms=np.random.uniform(40, 60),
                error_rate=max(0.0, min(1.0, error_rate)),
                throughput_per_second=np.random.uniform(80, 120),
                data_drift_score=np.random.uniform(0.01, 0.08),
                period_start=base_time + timedelta(days=day-1),
                period_end=base_time + timedelta(days=day),
                sample_size=np.random.randint(1000, 5000)
            )
            
            metrics_series.append(metrics)
        
        return metrics_series
    
    @staticmethod
    def generate_benchmark_dataset(size: int = 100) -> List[Dict[str, Any]]:
        """Generate benchmark dataset for model validation."""
        np.random.seed(42)
        
        positive_texts = [
            "Amazing touchdown pass!",
            "Incredible performance by the team",
            "Outstanding defensive play",
            "Championship winning performance",
            "Great coaching decision",
            "Perfect execution on offense",
            "Dominant rushing attack",
            "Excellent field goal kick",
            "Spectacular catch by receiver",
            "Brilliant quarterback play"
        ]
        
        negative_texts = [
            "Terrible fumble by the quarterback",
            "Disappointing loss in overtime",
            "Injury concerns for key players",
            "Poor coaching decision",
            "Missed field goal attempt",
            "Costly interception thrown",
            "Weak defensive performance",
            "Penalties hurt the team",
            "Offensive line struggles",
            "Turnover in red zone"
        ]
        
        neutral_texts = [
            "The game is scheduled for Sunday",
            "Player statistics for the season",
            "Game postponed due to weather",
            "Halftime show performance",
            "Team practice scheduled",
            "Press conference at 3 PM",
            "Stadium capacity is 70000",
            "Ticket sales begin Monday",
            "Weather forecast for game day",
            "Team roster updated"
        ]
        
        dataset = []
        
        for i in range(size):
            category = np.random.choice(["positive", "negative", "neutral"], p=[0.4, 0.4, 0.2])
            
            if category == "positive":
                text = np.random.choice(positive_texts)
                label = "POSITIVE"
                expected_score = np.random.uniform(0.6, 0.95)
            elif category == "negative":
                text = np.random.choice(negative_texts)
                label = "NEGATIVE"
                expected_score = np.random.uniform(-0.95, -0.6)
            else:
                text = np.random.choice(neutral_texts)
                label = "NEUTRAL"
                expected_score = np.random.uniform(-0.1, 0.1)
            
            dataset.append({
                "text": text,
                "label": label,
                "expected_score": expected_score,
                "category": category
            })
        
        return dataset


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


# Pytest markers for MLOps tests
pytest_plugins = []

def pytest_configure(config):
    """Configure pytest markers for MLOps tests."""
    config.addinivalue_line("markers", "mlops: MLOps related tests")
    config.addinivalue_line("markers", "model_validation: Model validation tests")
    config.addinivalue_line("markers", "deployment: Model deployment tests")
    config.addinivalue_line("markers", "data_drift: Data drift detection tests")
    config.addinivalue_line("markers", "integration: MLOps integration tests")
    config.addinivalue_line("markers", "performance: Performance monitoring tests")
    config.addinivalue_line("markers", "retraining: Model retraining tests")
    config.addinivalue_line("markers", "ab_testing: A/B testing functionality")
    config.addinivalue_line("markers", "feature_store: Feature store integration")
    config.addinivalue_line("markers", "experiment_tracking: Experiment tracking tests")