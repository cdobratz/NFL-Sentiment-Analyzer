"""
Data drift detection and monitoring tests for MLOps pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

from app.services.mlops.model_deployment_service import ModelDeploymentService
from app.services.mlops.model_retraining_service import ModelRetrainingService
from app.models.mlops import (
    ModelPerformanceMetrics, ModelAlert, ModelDeployment,
    ModelStatus, DeploymentStrategy
)


class TestDataDriftDetection:
    """Test data drift detection algorithms and monitoring."""
    
    @pytest.fixture
    def drift_detector(self):
        """Create a data drift detector instance."""
        return DataDriftDetector()
    
    @pytest.fixture
    def baseline_data(self):
        """Create baseline training data distribution."""
        np.random.seed(42)
        return {
            "sentiment_scores": np.random.normal(0.1, 0.3, 1000),
            "confidence_scores": np.random.beta(2, 2, 1000),
            "text_lengths": np.random.lognormal(3, 0.5, 1000),
            "keyword_counts": np.random.poisson(3, 1000),
            "team_mentions": np.random.choice(['patriots', 'cowboys', 'packers'], 1000),
            "source_distribution": np.random.choice(['twitter', 'espn', 'news'], 1000, p=[0.6, 0.3, 0.1])
        }
    
    @pytest.fixture
    def current_data_normal(self):
        """Create current data with normal distribution (no drift)."""
        np.random.seed(43)
        return {
            "sentiment_scores": np.random.normal(0.12, 0.31, 500),
            "confidence_scores": np.random.beta(2.1, 2.1, 500),
            "text_lengths": np.random.lognormal(3.05, 0.52, 500),
            "keyword_counts": np.random.poisson(3.1, 500),
            "team_mentions": np.random.choice(['patriots', 'cowboys', 'packers'], 500),
            "source_distribution": np.random.choice(['twitter', 'espn', 'news'], 500, p=[0.58, 0.32, 0.1])
        }
    
    @pytest.fixture
    def current_data_drift(self):
        """Create current data with significant drift."""
        np.random.seed(44)
        return {
            "sentiment_scores": np.random.normal(0.3, 0.4, 500),  # Shifted mean and variance
            "confidence_scores": np.random.beta(3, 1.5, 500),     # Different beta parameters
            "text_lengths": np.random.lognormal(2.5, 0.8, 500),   # Different log-normal parameters
            "keyword_counts": np.random.poisson(5, 500),          # Higher Poisson rate
            "team_mentions": np.random.choice(['patriots', 'cowboys', 'packers'], 500),
            "source_distribution": np.random.choice(['twitter', 'espn', 'news'], 500, p=[0.3, 0.5, 0.2])  # Different distribution
        }
    
    def test_kolmogorov_smirnov_drift_detection_no_drift(self, drift_detector, baseline_data, current_data_normal):
        """Test KS drift detection with no drift."""
        drift_score = drift_detector.detect_drift_ks(
            baseline_data["sentiment_scores"],
            current_data_normal["sentiment_scores"]
        )
        
        assert 0 <= drift_score <= 1
        assert drift_score < 0.1  # Should indicate no significant drift
    
    def test_kolmogorov_smirnov_drift_detection_with_drift(self, drift_detector, baseline_data, current_data_drift):
        """Test KS drift detection with significant drift."""
        drift_score = drift_detector.detect_drift_ks(
            baseline_data["sentiment_scores"],
            current_data_drift["sentiment_scores"]
        )
        
        assert 0 <= drift_score <= 1
        assert drift_score > 0.2  # Should indicate significant drift
    
    def test_population_stability_index_no_drift(self, drift_detector, baseline_data, current_data_normal):
        """Test PSI drift detection with no drift."""
        psi_score = drift_detector.calculate_psi(
            baseline_data["confidence_scores"],
            current_data_normal["confidence_scores"],
            bins=10
        )
        
        assert psi_score >= 0
        assert psi_score < 0.1  # PSI < 0.1 indicates no significant drift
    
    def test_population_stability_index_with_drift(self, drift_detector, baseline_data, current_data_drift):
        """Test PSI drift detection with significant drift."""
        psi_score = drift_detector.calculate_psi(
            baseline_data["confidence_scores"],
            current_data_drift["confidence_scores"],
            bins=10
        )
        
        assert psi_score >= 0
        assert psi_score > 0.2  # PSI > 0.2 indicates significant drift
    
    def test_jensen_shannon_divergence_no_drift(self, drift_detector, baseline_data, current_data_normal):
        """Test JS divergence drift detection with no drift."""
        js_score = drift_detector.calculate_js_divergence(
            baseline_data["text_lengths"],
            current_data_normal["text_lengths"],
            bins=20
        )
        
        assert 0 <= js_score <= 1
        assert js_score < 0.1  # Should indicate no significant drift
    
    def test_jensen_shannon_divergence_with_drift(self, drift_detector, baseline_data, current_data_drift):
        """Test JS divergence drift detection with significant drift."""
        js_score = drift_detector.calculate_js_divergence(
            baseline_data["text_lengths"],
            current_data_drift["text_lengths"],
            bins=20
        )
        
        assert 0 <= js_score <= 1
        assert js_score > 0.2  # Should indicate significant drift
    
    def test_categorical_drift_detection_no_drift(self, drift_detector, baseline_data, current_data_normal):
        """Test categorical drift detection with no drift."""
        drift_score = drift_detector.detect_categorical_drift(
            baseline_data["team_mentions"],
            current_data_normal["team_mentions"]
        )
        
        assert 0 <= drift_score <= 1
        assert drift_score < 0.1  # Should indicate no significant drift
    
    def test_categorical_drift_detection_with_drift(self, drift_detector, baseline_data, current_data_drift):
        """Test categorical drift detection with significant drift."""
        # Create data with different categorical distribution
        current_data_drift["team_mentions"] = np.random.choice(
            ['patriots', 'cowboys', 'packers'], 500, p=[0.8, 0.1, 0.1]  # Heavy bias towards patriots
        )
        
        drift_score = drift_detector.detect_categorical_drift(
            baseline_data["team_mentions"],
            current_data_drift["team_mentions"]
        )
        
        assert 0 <= drift_score <= 1
        assert drift_score > 0.2  # Should indicate significant drift
    
    def test_multivariate_drift_detection(self, drift_detector, baseline_data, current_data_normal, current_data_drift):
        """Test multivariate drift detection."""
        # Create feature matrices
        baseline_features = np.column_stack([
            baseline_data["sentiment_scores"],
            baseline_data["confidence_scores"],
            baseline_data["keyword_counts"]
        ])
        
        current_features_normal = np.column_stack([
            current_data_normal["sentiment_scores"],
            current_data_normal["confidence_scores"],
            current_data_normal["keyword_counts"]
        ])
        
        current_features_drift = np.column_stack([
            current_data_drift["sentiment_scores"],
            current_data_drift["confidence_scores"],
            current_data_drift["keyword_counts"]
        ])
        
        # Test no drift
        drift_score_normal = drift_detector.detect_multivariate_drift(
            baseline_features,
            current_features_normal
        )
        
        assert 0 <= drift_score_normal <= 1
        assert drift_score_normal < 0.15
        
        # Test with drift
        drift_score_drift = drift_detector.detect_multivariate_drift(
            baseline_features,
            current_features_drift
        )
        
        assert 0 <= drift_score_drift <= 1
        assert drift_score_drift > 0.2
    
    def test_comprehensive_drift_analysis(self, drift_detector, baseline_data, current_data_drift):
        """Test comprehensive drift analysis across all features."""
        drift_report = drift_detector.analyze_comprehensive_drift(
            baseline_data,
            current_data_drift,
            feature_types={
                "sentiment_scores": "continuous",
                "confidence_scores": "continuous",
                "text_lengths": "continuous",
                "keyword_counts": "discrete",
                "team_mentions": "categorical",
                "source_distribution": "categorical"
            }
        )
        
        assert "overall_drift_score" in drift_report
        assert "feature_drift_scores" in drift_report
        assert "drift_severity" in drift_report
        assert "recommendations" in drift_report
        
        # Check individual feature scores
        assert "sentiment_scores" in drift_report["feature_drift_scores"]
        assert "team_mentions" in drift_report["feature_drift_scores"]
        
        # Overall drift should be significant
        assert drift_report["overall_drift_score"] > 0.2
        assert drift_report["drift_severity"] in ["moderate", "high", "severe"]


class TestPerformanceMonitoring:
    """Test model performance monitoring and alerting."""
    
    @pytest.fixture
    def deployment_service(self):
        """Create deployment service for testing."""
        service = ModelDeploymentService()
        service.db = MagicMock()
        return service
    
    @pytest.fixture
    def deployment(self):
        """Create test deployment."""
        return ModelDeployment(
            deployment_id="test_deployment_monitoring",
            model_id="test_model",
            model_version="1.0",
            environment="production",
            strategy=DeploymentStrategy.IMMEDIATE,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    
    def test_performance_threshold_monitoring_normal(self, deployment_service, deployment):
        """Test performance monitoring with normal metrics."""
        metrics = ModelPerformanceMetrics(
            model_id="test_model",
            model_version="1.0",
            timestamp=datetime.utcnow(),
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            prediction_count=1000,
            avg_prediction_time_ms=45.0,
            error_rate=0.02,
            throughput_per_second=120.0,
            data_drift_score=0.05,
            period_start=datetime.utcnow() - timedelta(hours=1),
            period_end=datetime.utcnow(),
            sample_size=1000
        )
        
        deployment_service.active_deployments[deployment.deployment_id] = deployment
        
        # Should not generate alerts for normal performance
        alerts = asyncio.run(deployment_service.monitor_model_performance(
            deployment.deployment_id, metrics
        ))
        
        assert len(alerts) == 0
    
    def test_performance_threshold_monitoring_degraded(self, deployment_service, deployment):
        """Test performance monitoring with degraded metrics."""
        metrics = ModelPerformanceMetrics(
            model_id="test_model",
            model_version="1.0",
            timestamp=datetime.utcnow(),
            accuracy=0.70,  # Below threshold
            precision=0.68,  # Below threshold
            recall=0.72,
            f1_score=0.70,  # Below threshold
            prediction_count=1000,
            avg_prediction_time_ms=150.0,  # Above threshold
            error_rate=0.08,  # Above threshold
            throughput_per_second=30.0,  # Below threshold
            data_drift_score=0.15,  # Above threshold
            period_start=datetime.utcnow() - timedelta(hours=1),
            period_end=datetime.utcnow(),
            sample_size=1000
        )
        
        deployment_service.active_deployments[deployment.deployment_id] = deployment
        deployment_service._store_alert = AsyncMock()
        
        # Should generate multiple alerts
        alerts = asyncio.run(deployment_service.monitor_model_performance(
            deployment.deployment_id, metrics
        ))
        
        assert len(alerts) > 0
        
        # Check for different types of alerts
        alert_types = [alert.alert_type for alert in alerts]
        assert "performance_degradation" in alert_types
        assert "data_drift" in alert_types
        assert "error_spike" in alert_types
    
    def test_alert_severity_classification(self, deployment_service, deployment):
        """Test alert severity classification."""
        # Critical performance degradation
        critical_metrics = ModelPerformanceMetrics(
            model_id="test_model",
            model_version="1.0",
            timestamp=datetime.utcnow(),
            accuracy=0.50,  # Very low
            error_rate=0.15,  # Very high
            period_start=datetime.utcnow() - timedelta(hours=1),
            period_end=datetime.utcnow(),
            sample_size=1000
        )
        
        deployment_service.active_deployments[deployment.deployment_id] = deployment
        deployment_service._store_alert = AsyncMock()
        
        alerts = asyncio.run(deployment_service.monitor_model_performance(
            deployment.deployment_id, critical_metrics
        ))
        
        # Should have critical/high severity alerts
        high_severity_alerts = [alert for alert in alerts if alert.severity in ["critical", "high"]]
        assert len(high_severity_alerts) > 0
    
    def test_performance_trend_analysis(self):
        """Test performance trend analysis over time."""
        # Create historical performance data
        historical_metrics = []
        base_time = datetime.utcnow() - timedelta(days=7)
        
        for i in range(7):
            # Simulate gradual performance degradation
            accuracy = 0.90 - (i * 0.02)  # Decreasing accuracy
            error_rate = 0.01 + (i * 0.005)  # Increasing error rate
            
            metrics = ModelPerformanceMetrics(
                model_id="test_model",
                model_version="1.0",
                timestamp=base_time + timedelta(days=i),
                accuracy=accuracy,
                error_rate=error_rate,
                period_start=base_time + timedelta(days=i) - timedelta(hours=1),
                period_end=base_time + timedelta(days=i),
                sample_size=1000
            )
            historical_metrics.append(metrics)
        
        trend_analyzer = PerformanceTrendAnalyzer()
        trend_analysis = trend_analyzer.analyze_trends(historical_metrics)
        
        assert "accuracy_trend" in trend_analysis
        assert "error_rate_trend" in trend_analysis
        assert trend_analysis["accuracy_trend"]["direction"] == "decreasing"
        assert trend_analysis["error_rate_trend"]["direction"] == "increasing"
        assert trend_analysis["overall_trend"] == "degrading"
    
    def test_anomaly_detection_in_metrics(self):
        """Test anomaly detection in performance metrics."""
        # Create normal metrics with one anomaly
        normal_accuracies = [0.85, 0.86, 0.84, 0.87, 0.85, 0.86, 0.84]
        anomaly_accuracy = 0.65  # Significant drop
        
        all_accuracies = normal_accuracies + [anomaly_accuracy]
        
        anomaly_detector = MetricAnomalyDetector()
        anomalies = anomaly_detector.detect_anomalies(
            all_accuracies,
            method="isolation_forest"
        )
        
        assert len(anomalies) > 0
        assert anomalies[-1] == True  # Last value should be detected as anomaly
    
    def test_performance_forecasting(self):
        """Test performance forecasting based on historical data."""
        # Create time series data
        timestamps = [datetime.utcnow() - timedelta(days=i) for i in range(30, 0, -1)]
        accuracies = [0.90 - (i * 0.001) for i in range(30)]  # Gradual decline
        
        forecaster = PerformanceForecaster()
        forecast = forecaster.forecast_performance(
            timestamps=timestamps,
            values=accuracies,
            forecast_days=7
        )
        
        assert "forecasted_values" in forecast
        assert "confidence_intervals" in forecast
        assert "trend_direction" in forecast
        assert len(forecast["forecasted_values"]) == 7
        assert forecast["trend_direction"] == "declining"


class TestRetrainingTriggers:
    """Test automated retraining trigger mechanisms."""
    
    @pytest.fixture
    def retraining_service(self):
        """Create retraining service for testing."""
        service = ModelRetrainingService()
        service.db = MagicMock()
        return service
    
    @pytest.mark.asyncio
    async def test_performance_based_trigger(self, retraining_service):
        """Test performance-based retraining trigger."""
        # Setup performance history with degradation
        model_name = "test_model"
        retraining_service.performance_history[model_name] = []
        
        # Add degrading performance metrics
        for i in range(5):
            metrics = ModelPerformanceMetrics(
                model_id=model_name,
                model_version="1.0",
                timestamp=datetime.utcnow() - timedelta(hours=i),
                accuracy=0.85 - (i * 0.02),  # Degrading accuracy
                f1_score=0.83 - (i * 0.02),  # Degrading F1
                period_start=datetime.utcnow() - timedelta(hours=i+1),
                period_end=datetime.utcnow() - timedelta(hours=i),
                sample_size=1000
            )
            retraining_service.performance_history[model_name].append(metrics)
        
        # Create retraining config
        config = ModelRetrainingConfig(
            config_id="test_config",
            model_name=model_name,
            performance_threshold=0.8,
            enabled=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        retraining_service.retraining_configs["test_config"] = config
        
        # Check triggers
        trigger_result = await retraining_service.check_retraining_triggers(model_name)
        
        assert trigger_result["should_retrain"] is True
        assert "performance_degradation" in trigger_result["triggers_activated"]
    
    @pytest.mark.asyncio
    async def test_data_drift_based_trigger(self, retraining_service):
        """Test data drift-based retraining trigger."""
        model_name = "test_model"
        
        # Add metrics with high data drift
        metrics = ModelPerformanceMetrics(
            model_id=model_name,
            model_version="1.0",
            timestamp=datetime.utcnow(),
            accuracy=0.85,
            f1_score=0.83,
            data_drift_score=0.15,  # High drift
            period_start=datetime.utcnow() - timedelta(hours=1),
            period_end=datetime.utcnow(),
            sample_size=1000
        )
        retraining_service.performance_history[model_name] = [metrics]
        
        # Create retraining config
        config = ModelRetrainingConfig(
            config_id="test_config",
            model_name=model_name,
            data_drift_threshold=0.1,
            enabled=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        retraining_service.retraining_configs["test_config"] = config
        
        # Check triggers
        trigger_result = await retraining_service.check_retraining_triggers(model_name)
        
        assert trigger_result["should_retrain"] is True
        assert "data_drift" in trigger_result["triggers_activated"]
    
    @pytest.mark.asyncio
    async def test_time_based_trigger(self, retraining_service):
        """Test time-based retraining trigger."""
        model_name = "test_model"
        
        # Create config with old last trigger time
        config = ModelRetrainingConfig(
            config_id="test_config",
            model_name=model_name,
            time_based_trigger="0 2 * * 0",  # Weekly
            last_triggered=datetime.utcnow() - timedelta(days=8),  # 8 days ago
            enabled=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        retraining_service.retraining_configs["test_config"] = config
        
        # Check triggers
        trigger_result = await retraining_service.check_retraining_triggers(model_name)
        
        assert trigger_result["should_retrain"] is True
        assert "scheduled_retraining" in trigger_result["triggers_activated"]
    
    @pytest.mark.asyncio
    async def test_data_volume_trigger(self, retraining_service):
        """Test data volume-based retraining trigger."""
        model_name = "test_model"
        
        # Mock new data count
        retraining_service._get_new_data_count = AsyncMock(return_value=15000)
        
        # Create config with data volume trigger
        config = ModelRetrainingConfig(
            config_id="test_config",
            model_name=model_name,
            data_volume_trigger=10000,  # Trigger after 10k new samples
            last_triggered=datetime.utcnow() - timedelta(days=1),
            enabled=True,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        retraining_service.retraining_configs["test_config"] = config
        
        # Check triggers
        trigger_result = await retraining_service.check_retraining_triggers(model_name)
        
        assert trigger_result["should_retrain"] is True
        assert "new_data_available" in trigger_result["triggers_activated"]


# Helper classes for testing
class DataDriftDetector:
    """Data drift detection utility class."""
    
    def detect_drift_ks(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Detect drift using Kolmogorov-Smirnov test."""
        from scipy import stats
        statistic, p_value = stats.ks_2samp(baseline, current)
        return statistic
    
    def calculate_psi(self, baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        # Create bins based on baseline data
        bin_edges = np.histogram_bin_edges(baseline, bins=bins)
        
        # Calculate distributions
        baseline_dist, _ = np.histogram(baseline, bins=bin_edges, density=True)
        current_dist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        # Normalize to probabilities
        baseline_dist = baseline_dist / np.sum(baseline_dist)
        current_dist = current_dist / np.sum(current_dist)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        baseline_dist = baseline_dist + epsilon
        current_dist = current_dist + epsilon
        
        # Calculate PSI
        psi = np.sum((current_dist - baseline_dist) * np.log(current_dist / baseline_dist))
        return psi
    
    def calculate_js_divergence(self, baseline: np.ndarray, current: np.ndarray, bins: int = 20) -> float:
        """Calculate Jensen-Shannon divergence."""
        from scipy.spatial.distance import jensenshannon
        
        # Create histograms
        bin_edges = np.histogram_bin_edges(np.concatenate([baseline, current]), bins=bins)
        baseline_hist, _ = np.histogram(baseline, bins=bin_edges, density=True)
        current_hist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        # Normalize
        baseline_hist = baseline_hist / np.sum(baseline_hist)
        current_hist = current_hist / np.sum(current_hist)
        
        # Calculate JS divergence
        js_distance = jensenshannon(baseline_hist, current_hist)
        return js_distance ** 2  # Return squared distance
    
    def detect_categorical_drift(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Detect drift in categorical variables using chi-square test."""
        from scipy import stats
        
        # Get unique categories
        all_categories = np.unique(np.concatenate([baseline, current]))
        
        # Calculate distributions
        baseline_counts = np.array([np.sum(baseline == cat) for cat in all_categories])
        current_counts = np.array([np.sum(current == cat) for cat in all_categories])
        
        # Normalize to probabilities
        baseline_probs = baseline_counts / np.sum(baseline_counts)
        current_probs = current_counts / np.sum(current_counts)
        
        # Calculate chi-square statistic
        expected_current = baseline_probs * np.sum(current_counts)
        chi2_stat = np.sum((current_counts - expected_current) ** 2 / (expected_current + 1e-10))
        
        # Normalize to 0-1 range (approximate)
        return min(chi2_stat / len(all_categories), 1.0)
    
    def detect_multivariate_drift(self, baseline: np.ndarray, current: np.ndarray) -> float:
        """Detect multivariate drift using Maximum Mean Discrepancy."""
        # Simplified MMD implementation
        def rbf_kernel(X, Y, gamma=1.0):
            """RBF kernel for MMD calculation."""
            XX = np.sum(X**2, axis=1)[:, np.newaxis]
            YY = np.sum(Y**2, axis=1)[np.newaxis, :]
            XY = np.dot(X, Y.T)
            distances = XX + YY - 2 * XY
            return np.exp(-gamma * distances)
        
        # Calculate MMD
        n_baseline = baseline.shape[0]
        n_current = current.shape[0]
        
        K_XX = rbf_kernel(baseline, baseline)
        K_YY = rbf_kernel(current, current)
        K_XY = rbf_kernel(baseline, current)
        
        mmd = (np.sum(K_XX) / (n_baseline * n_baseline) + 
               np.sum(K_YY) / (n_current * n_current) - 
               2 * np.sum(K_XY) / (n_baseline * n_current))
        
        return max(0, mmd)  # Ensure non-negative
    
    def analyze_comprehensive_drift(self, baseline_data: Dict, current_data: Dict, feature_types: Dict) -> Dict:
        """Analyze drift across all features comprehensively."""
        feature_scores = {}
        
        for feature, feature_type in feature_types.items():
            if feature not in baseline_data or feature not in current_data:
                continue
            
            baseline_values = baseline_data[feature]
            current_values = current_data[feature]
            
            if feature_type == "continuous":
                score = self.detect_drift_ks(baseline_values, current_values)
            elif feature_type == "discrete":
                score = self.calculate_psi(baseline_values, current_values)
            elif feature_type == "categorical":
                score = self.detect_categorical_drift(baseline_values, current_values)
            else:
                score = 0.0
            
            feature_scores[feature] = score
        
        # Calculate overall drift score
        overall_score = np.mean(list(feature_scores.values()))
        
        # Determine severity
        if overall_score < 0.1:
            severity = "low"
        elif overall_score < 0.2:
            severity = "moderate"
        elif overall_score < 0.3:
            severity = "high"
        else:
            severity = "severe"
        
        # Generate recommendations
        recommendations = []
        if overall_score > 0.2:
            recommendations.append("Consider model retraining")
        if overall_score > 0.3:
            recommendations.append("Investigate data pipeline changes")
            recommendations.append("Review feature engineering")
        
        return {
            "overall_drift_score": overall_score,
            "feature_drift_scores": feature_scores,
            "drift_severity": severity,
            "recommendations": recommendations
        }


class PerformanceTrendAnalyzer:
    """Analyze performance trends over time."""
    
    def analyze_trends(self, metrics_history: List[ModelPerformanceMetrics]) -> Dict:
        """Analyze performance trends."""
        if len(metrics_history) < 3:
            return {"error": "Insufficient data for trend analysis"}
        
        # Extract time series data
        timestamps = [m.timestamp for m in metrics_history]
        accuracies = [m.accuracy for m in metrics_history if m.accuracy is not None]
        error_rates = [m.error_rate for m in metrics_history if m.error_rate is not None]
        
        # Analyze trends
        accuracy_trend = self._analyze_single_trend(accuracies)
        error_rate_trend = self._analyze_single_trend(error_rates)
        
        # Determine overall trend
        if accuracy_trend["direction"] == "decreasing" or error_rate_trend["direction"] == "increasing":
            overall_trend = "degrading"
        elif accuracy_trend["direction"] == "increasing" and error_rate_trend["direction"] == "decreasing":
            overall_trend = "improving"
        else:
            overall_trend = "stable"
        
        return {
            "accuracy_trend": accuracy_trend,
            "error_rate_trend": error_rate_trend,
            "overall_trend": overall_trend,
            "analysis_period": {
                "start": min(timestamps),
                "end": max(timestamps),
                "data_points": len(metrics_history)
            }
        }
    
    def _analyze_single_trend(self, values: List[float]) -> Dict:
        """Analyze trend for a single metric."""
        if len(values) < 2:
            return {"direction": "unknown", "slope": 0, "confidence": 0}
        
        # Simple linear regression
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine direction
        if abs(slope) < 0.001:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Calculate confidence (simplified)
        confidence = min(abs(slope) * 100, 1.0)
        
        return {
            "direction": direction,
            "slope": slope,
            "confidence": confidence
        }


class MetricAnomalyDetector:
    """Detect anomalies in performance metrics."""
    
    def detect_anomalies(self, values: List[float], method: str = "isolation_forest") -> List[bool]:
        """Detect anomalies in metric values."""
        if len(values) < 3:
            return [False] * len(values)
        
        values_array = np.array(values).reshape(-1, 1)
        
        if method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            detector = IsolationForest(contamination=0.1, random_state=42)
            anomalies = detector.fit_predict(values_array)
            return [a == -1 for a in anomalies]
        
        elif method == "statistical":
            # Use statistical method (z-score)
            mean_val = np.mean(values)
            std_val = np.std(values)
            z_scores = np.abs((values_array.flatten() - mean_val) / (std_val + 1e-10))
            return [z > 2.5 for z in z_scores]  # 2.5 sigma threshold
        
        else:
            return [False] * len(values)


class PerformanceForecaster:
    """Forecast future performance based on historical data."""
    
    def forecast_performance(self, timestamps: List[datetime], values: List[float], forecast_days: int = 7) -> Dict:
        """Forecast performance metrics."""
        if len(values) < 5:
            return {"error": "Insufficient data for forecasting"}
        
        # Simple linear trend forecasting
        x = np.arange(len(values))
        y = np.array(values)
        
        # Fit linear model
        coeffs = np.polyfit(x, y, 1)
        slope, intercept = coeffs
        
        # Generate forecasts
        forecast_x = np.arange(len(values), len(values) + forecast_days)
        forecasted_values = slope * forecast_x + intercept
        
        # Calculate confidence intervals (simplified)
        residuals = y - (slope * x + intercept)
        std_error = np.std(residuals)
        confidence_intervals = [(val - 1.96 * std_error, val + 1.96 * std_error) 
                               for val in forecasted_values]
        
        # Determine trend direction
        if abs(slope) < 0.001:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"
        
        return {
            "forecasted_values": forecasted_values.tolist(),
            "confidence_intervals": confidence_intervals,
            "trend_direction": trend_direction,
            "forecast_period_days": forecast_days,
            "model_accuracy": 1.0 - (std_error / np.mean(y))  # Simplified accuracy measure
        }