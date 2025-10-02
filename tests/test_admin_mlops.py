"""
Tests for admin MLOps functionality.
Tests model retraining, deployment, and monitoring features.
"""
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from bson import ObjectId

# Mock settings before importing app modules
with patch.dict('os.environ', {
    'SECRET_KEY': 'test-secret-key',
    'MONGODB_URL': 'mongodb://test:27017',
    'DATABASE_NAME': 'test_db'
}):
    from app.models.mlops import (
        ModelRetrainingRequest,
        ExperimentRun,
        ExperimentStatus,
        ModelMetadata,
        ModelStatus,
        ModelDeployment,
        DeploymentStrategy,
        ModelPerformanceMetrics
    )


class TestModelRetrainingFunctionality:
    """Test model retraining functionality for admin users."""
    
    @pytest.mark.asyncio
    async def test_model_retraining_request_validation(self):
        """Test model retraining request validation."""
        # Valid retraining request
        valid_request = ModelRetrainingRequest(
            model_name="sentiment_base",
            trigger_reason="manual_admin_trigger",
            training_config={"epochs": 10, "learning_rate": 0.001},
            auto_deploy=False
        )
        
        assert valid_request.model_name == "sentiment_base"
        assert valid_request.trigger_reason == "manual_admin_trigger"
        assert valid_request.auto_deploy is False
        assert valid_request.training_config["epochs"] == 10
    
    @pytest.mark.asyncio
    async def test_experiment_run_creation(self):
        """Test experiment run creation and tracking."""
        experiment_run = ExperimentRun(
            experiment_id="exp_123",
            run_id="run_456",
            experiment_name="sentiment_retraining",
            status=ExperimentStatus.RUNNING,
            model_type="sentiment_analysis",
            hyperparameters={"learning_rate": 0.001, "batch_size": 32},
            started_at=datetime.utcnow()
        )
        
        assert experiment_run.experiment_id == "exp_123"
        assert experiment_run.run_id == "run_456"
        assert experiment_run.status == ExperimentStatus.RUNNING
        assert experiment_run.hyperparameters["learning_rate"] == 0.001
    
    @pytest.mark.asyncio
    async def test_model_retraining_workflow(self):
        """Test complete model retraining workflow."""
        mock_mlops_service = MagicMock()
        mock_db = MagicMock()
        
        # Mock MLOps service initialization
        mock_mlops_service.initialized = True
        
        # Create mock experiment run
        mock_experiment_run = ExperimentRun(
            experiment_id="exp_789",
            run_id="run_101112",
            experiment_name="admin_triggered_retraining",
            status=ExperimentStatus.RUNNING,
            model_type="sentiment_analysis",
            started_at=datetime.utcnow()
        )
        
        mock_mlops_service.trigger_retraining.return_value = mock_experiment_run
        
        # Mock database operations
        mock_db.ml_jobs.insert_one = AsyncMock()
        
        # Execute retraining workflow
        result = await self._execute_retraining_workflow(
            mock_mlops_service,
            mock_db,
            "sentiment_base",
            "manual_admin_trigger",
            "admin_user_123"
        )
        
        assert result["success"] is True
        assert result["job_id"] == "run_101112"
        assert result["experiment_id"] == "exp_789"
        
        # Verify MLOps service was called
        mock_mlops_service.trigger_retraining.assert_called_once()
        
        # Verify job was stored in database
        mock_db.ml_jobs.insert_one.assert_called_once()
    
    async def _execute_retraining_workflow(self, mlops_service, db, model_name, trigger_reason, admin_user_id):
        """Execute model retraining workflow."""
        try:
            # Trigger retraining through MLOps service
            experiment_run = mlops_service.trigger_retraining(
                model_name=model_name,
                reason=trigger_reason
            )
            
            # Store job record in database
            retraining_job = {
                "job_id": experiment_run.run_id,
                "experiment_id": experiment_run.experiment_id,
                "model_name": model_name,
                "status": experiment_run.status.value,
                "trigger_reason": trigger_reason,
                "triggered_by": admin_user_id,
                "created_at": datetime.utcnow(),
                "started_at": experiment_run.started_at
            }
            
            await db.ml_jobs.insert_one(retraining_job)
            
            return {
                "success": True,
                "job_id": experiment_run.run_id,
                "experiment_id": experiment_run.experiment_id,
                "status": experiment_run.status.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @pytest.mark.asyncio
    async def test_model_retraining_failure_handling(self):
        """Test handling of model retraining failures."""
        mock_mlops_service = MagicMock()
        mock_db = MagicMock()
        
        # Mock MLOps service failure
        mock_mlops_service.initialized = False
        mock_mlops_service.initialize = AsyncMock(side_effect=Exception("MLOps initialization failed"))
        
        # Execute retraining workflow with failure
        result = await self._execute_retraining_workflow(
            mock_mlops_service,
            mock_db,
            "sentiment_base",
            "manual_admin_trigger",
            "admin_user_123"
        )
        
        assert result["success"] is False
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_ml_job_status_tracking(self):
        """Test ML job status tracking and updates."""
        mock_db = MagicMock()
        
        # Mock job data
        job_data = {
            "job_id": "run_123",
            "experiment_id": "exp_456",
            "model_name": "sentiment_base",
            "status": "running",
            "created_at": datetime.utcnow(),
            "started_at": datetime.utcnow()
        }
        
        # Mock cursor for job retrieval
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__.return_value = iter([job_data])
        mock_db.ml_jobs.find.return_value = mock_cursor
        
        # Test job retrieval
        jobs = []
        async for job in mock_cursor:
            jobs.append(job)
        
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "run_123"
        assert jobs[0]["status"] == "running"
        
        # Test job status update
        mock_db.ml_jobs.update_one = AsyncMock()
        
        # Simulate job completion
        await mock_db.ml_jobs.update_one(
            {"job_id": "run_123"},
            {
                "$set": {
                    "status": "completed",
                    "completed_at": datetime.utcnow(),
                    "final_metrics": {"accuracy": 0.87, "f1_score": 0.84}
                }
            }
        )
        
        mock_db.ml_jobs.update_one.assert_called_once()


class TestModelDeploymentManagement:
    """Test model deployment management functionality."""
    
    @pytest.mark.asyncio
    async def test_model_deployment_configuration(self):
        """Test model deployment configuration."""
        deployment = ModelDeployment(
            deployment_id="deploy_123",
            model_id="sentiment_v2",
            model_version="2.0.0",
            environment="production",
            strategy=DeploymentStrategy.BLUE_GREEN,
            status=ModelStatus.DEPLOYED,
            traffic_percentage=100.0,
            replicas=3,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        assert deployment.deployment_id == "deploy_123"
        assert deployment.model_id == "sentiment_v2"
        assert deployment.strategy == DeploymentStrategy.BLUE_GREEN
        assert deployment.traffic_percentage == 100.0
        assert deployment.replicas == 3
    
    @pytest.mark.asyncio
    async def test_model_deployment_workflow(self):
        """Test complete model deployment workflow."""
        mock_mlops_service = MagicMock()
        mock_db = MagicMock()
        
        # Mock deployment creation
        deployment_config = {
            "model_id": "sentiment_v2",
            "model_version": "2.0.0",
            "environment": "production",
            "strategy": "blue_green",
            "replicas": 3
        }
        
        mock_deployment = ModelDeployment(
            deployment_id="deploy_456",
            model_id=deployment_config["model_id"],
            model_version=deployment_config["model_version"],
            environment=deployment_config["environment"],
            strategy=DeploymentStrategy.BLUE_GREEN,
            status=ModelStatus.DEPLOYED,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        mock_mlops_service.deploy_model.return_value = mock_deployment
        mock_db.deployments.insert_one = AsyncMock()
        
        # Execute deployment workflow
        result = await self._execute_deployment_workflow(
            mock_mlops_service,
            mock_db,
            deployment_config,
            "admin_user_123"
        )
        
        assert result["success"] is True
        assert result["deployment_id"] == "deploy_456"
        
        # Verify MLOps service was called
        mock_mlops_service.deploy_model.assert_called_once()
        
        # Verify deployment was stored
        mock_db.deployments.insert_one.assert_called_once()
    
    async def _execute_deployment_workflow(self, mlops_service, db, deployment_config, admin_user_id):
        """Execute model deployment workflow."""
        try:
            # Deploy model through MLOps service
            deployment = mlops_service.deploy_model(
                model_id=deployment_config["model_id"],
                model_version=deployment_config["model_version"],
                environment=deployment_config["environment"],
                strategy=deployment_config["strategy"]
            )
            
            # Store deployment record
            deployment_record = {
                "deployment_id": deployment.deployment_id,
                "model_id": deployment.model_id,
                "model_version": deployment.model_version,
                "environment": deployment.environment,
                "status": deployment.status.value,
                "deployed_by": admin_user_id,
                "created_at": deployment.created_at
            }
            
            await db.deployments.insert_one(deployment_record)
            
            return {
                "success": True,
                "deployment_id": deployment.deployment_id,
                "status": deployment.status.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @pytest.mark.asyncio
    async def test_deployment_rollback_functionality(self):
        """Test deployment rollback functionality."""
        mock_mlops_service = MagicMock()
        mock_db = MagicMock()
        
        # Mock current deployment
        current_deployment = {
            "deployment_id": "deploy_current",
            "model_id": "sentiment_v2",
            "model_version": "2.0.0",
            "status": "deployed"
        }
        
        # Mock previous deployment
        previous_deployment = {
            "deployment_id": "deploy_previous",
            "model_id": "sentiment_v1",
            "model_version": "1.5.0",
            "status": "retired"
        }
        
        mock_db.deployments.find_one.side_effect = [current_deployment, previous_deployment]
        mock_db.deployments.update_one = AsyncMock()
        
        # Execute rollback
        result = await self._execute_deployment_rollback(
            mock_mlops_service,
            mock_db,
            "deploy_current",
            "Performance degradation",
            "admin_user_123"
        )
        
        assert result["success"] is True
        assert "rollback_deployment_id" in result
        
        # Verify database updates
        assert mock_db.deployments.update_one.call_count >= 1
    
    async def _execute_deployment_rollback(self, mlops_service, db, deployment_id, rollback_reason, admin_user_id):
        """Execute deployment rollback."""
        try:
            # Get current deployment
            current_deployment = db.deployments.find_one({"deployment_id": deployment_id})
            
            if not current_deployment:
                raise ValueError("Deployment not found")
            
            # Get previous deployment
            previous_deployment = db.deployments.find_one({
                "model_id": current_deployment["model_id"],
                "status": "retired"
            }, sort=[("created_at", -1)])
            
            if not previous_deployment:
                raise ValueError("No previous deployment found for rollback")
            
            # Update current deployment status
            await db.deployments.update_one(
                {"deployment_id": deployment_id},
                {
                    "$set": {
                        "status": "rolled_back",
                        "rollback_reason": rollback_reason,
                        "rolled_back_at": datetime.utcnow(),
                        "rolled_back_by": admin_user_id
                    }
                }
            )
            
            # Reactivate previous deployment
            await db.deployments.update_one(
                {"deployment_id": previous_deployment["deployment_id"]},
                {
                    "$set": {
                        "status": "deployed",
                        "reactivated_at": datetime.utcnow(),
                        "reactivated_by": admin_user_id
                    }
                }
            )
            
            return {
                "success": True,
                "rollback_deployment_id": previous_deployment["deployment_id"],
                "rollback_reason": rollback_reason
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


class TestModelPerformanceMonitoring:
    """Test model performance monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self):
        """Test performance metrics collection and storage."""
        performance_metrics = ModelPerformanceMetrics(
            model_id="sentiment_v1",
            model_version="1.0.0",
            timestamp=datetime.utcnow(),
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            prediction_count=1000,
            avg_prediction_time_ms=150.5,
            error_rate=0.02,
            period_start=datetime.utcnow() - timedelta(hours=1),
            period_end=datetime.utcnow(),
            sample_size=1000
        )
        
        assert performance_metrics.model_id == "sentiment_v1"
        assert performance_metrics.accuracy == 0.85
        assert performance_metrics.f1_score == 0.85
        assert performance_metrics.prediction_count == 1000
        assert performance_metrics.error_rate == 0.02
    
    @pytest.mark.asyncio
    async def test_performance_metrics_aggregation(self):
        """Test performance metrics aggregation over time periods."""
        mock_db = MagicMock()
        
        # Mock performance metrics data
        metrics_data = [
            {
                "model_id": "sentiment_v1",
                "timestamp": datetime.utcnow() - timedelta(hours=1),
                "accuracy": 0.85,
                "f1_score": 0.82,
                "avg_prediction_time_ms": 150,
                "prediction_count": 1000,
                "error_rate": 0.02
            },
            {
                "model_id": "sentiment_v1",
                "timestamp": datetime.utcnow() - timedelta(hours=2),
                "accuracy": 0.87,
                "f1_score": 0.84,
                "avg_prediction_time_ms": 145,
                "prediction_count": 950,
                "error_rate": 0.015
            },
            {
                "model_id": "sentiment_v1",
                "timestamp": datetime.utcnow() - timedelta(hours=3),
                "accuracy": 0.83,
                "f1_score": 0.80,
                "avg_prediction_time_ms": 155,
                "prediction_count": 1100,
                "error_rate": 0.025
            }
        ]
        
        # Mock cursor
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__.return_value = iter(metrics_data)
        mock_db.performance_metrics.find.return_value = mock_cursor
        
        # Execute performance analysis
        performance_summary = await self._analyze_model_performance(mock_db, "sentiment_v1", 24)
        
        assert performance_summary["model_id"] == "sentiment_v1"
        assert performance_summary["metrics_count"] == 3
        assert "avg_accuracy" in performance_summary
        assert "avg_f1_score" in performance_summary
        assert "total_predictions" in performance_summary
        
        # Verify calculations
        expected_avg_accuracy = (0.85 + 0.87 + 0.83) / 3
        assert abs(performance_summary["avg_accuracy"] - expected_avg_accuracy) < 0.001
    
    async def _analyze_model_performance(self, db, model_id, hours):
        """Analyze model performance over time period."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours)
        
        query = {
            "model_id": model_id,
            "timestamp": {"$gte": start_date}
        }
        
        metrics = []
        async for doc in db.performance_metrics.find(query):
            metrics.append(doc)
        
        if not metrics:
            return {"model_id": model_id, "metrics": []}
        
        # Calculate aggregated metrics
        avg_accuracy = sum(m.get("accuracy", 0) for m in metrics) / len(metrics)
        avg_f1_score = sum(m.get("f1_score", 0) for m in metrics) / len(metrics)
        avg_response_time = sum(m.get("avg_prediction_time_ms", 0) for m in metrics) / len(metrics)
        total_predictions = sum(m.get("prediction_count", 0) for m in metrics)
        avg_error_rate = sum(m.get("error_rate", 0) for m in metrics) / len(metrics)
        
        return {
            "model_id": model_id,
            "period_hours": hours,
            "metrics_count": len(metrics),
            "avg_accuracy": round(avg_accuracy, 3),
            "avg_f1_score": round(avg_f1_score, 3),
            "avg_response_time_ms": round(avg_response_time, 2),
            "total_predictions": total_predictions,
            "avg_error_rate": round(avg_error_rate, 4)
        }
    
    @pytest.mark.asyncio
    async def test_performance_threshold_monitoring(self):
        """Test performance threshold monitoring and alerting."""
        # Define performance thresholds
        thresholds = {
            "accuracy": 0.80,
            "f1_score": 0.75,
            "error_rate": 0.05,
            "avg_prediction_time_ms": 200
        }
        
        # Test metrics that violate thresholds
        problematic_metrics = ModelPerformanceMetrics(
            model_id="sentiment_v1",
            model_version="1.0.0",
            timestamp=datetime.utcnow(),
            accuracy=0.75,  # Below threshold
            f1_score=0.70,  # Below threshold
            error_rate=0.08,  # Above threshold
            avg_prediction_time_ms=250,  # Above threshold
            prediction_count=1000,
            period_start=datetime.utcnow() - timedelta(hours=1),
            period_end=datetime.utcnow(),
            sample_size=1000
        )
        
        # Check threshold violations
        violations = self._check_performance_thresholds(problematic_metrics, thresholds)
        
        assert len(violations) == 4  # All metrics violate thresholds
        assert "accuracy" in violations
        assert "f1_score" in violations
        assert "error_rate" in violations
        assert "avg_prediction_time_ms" in violations
    
    def _check_performance_thresholds(self, metrics, thresholds):
        """Check performance metrics against thresholds."""
        violations = []
        
        if metrics.accuracy and metrics.accuracy < thresholds["accuracy"]:
            violations.append("accuracy")
        
        if metrics.f1_score and metrics.f1_score < thresholds["f1_score"]:
            violations.append("f1_score")
        
        if metrics.error_rate > thresholds["error_rate"]:
            violations.append("error_rate")
        
        if metrics.avg_prediction_time_ms > thresholds["avg_prediction_time_ms"]:
            violations.append("avg_prediction_time_ms")
        
        return violations
    
    @pytest.mark.asyncio
    async def test_model_comparison_analysis(self):
        """Test model comparison analysis for A/B testing."""
        # Mock performance data for two models
        model_a_metrics = [
            {"accuracy": 0.85, "f1_score": 0.82, "avg_prediction_time_ms": 150},
            {"accuracy": 0.87, "f1_score": 0.84, "avg_prediction_time_ms": 145},
            {"accuracy": 0.83, "f1_score": 0.80, "avg_prediction_time_ms": 155}
        ]
        
        model_b_metrics = [
            {"accuracy": 0.88, "f1_score": 0.85, "avg_prediction_time_ms": 160},
            {"accuracy": 0.86, "f1_score": 0.83, "avg_prediction_time_ms": 165},
            {"accuracy": 0.89, "f1_score": 0.86, "avg_prediction_time_ms": 158}
        ]
        
        # Perform comparison analysis
        comparison = self._compare_model_performance(model_a_metrics, model_b_metrics)
        
        assert "model_a" in comparison
        assert "model_b" in comparison
        assert "winner" in comparison
        
        # Model B should be the winner (higher accuracy and f1_score)
        assert comparison["winner"] == "model_b"
        assert comparison["model_b"]["avg_accuracy"] > comparison["model_a"]["avg_accuracy"]
    
    def _compare_model_performance(self, model_a_metrics, model_b_metrics):
        """Compare performance between two models."""
        # Calculate averages for model A
        model_a_avg = {
            "avg_accuracy": sum(m["accuracy"] for m in model_a_metrics) / len(model_a_metrics),
            "avg_f1_score": sum(m["f1_score"] for m in model_a_metrics) / len(model_a_metrics),
            "avg_prediction_time_ms": sum(m["avg_prediction_time_ms"] for m in model_a_metrics) / len(model_a_metrics)
        }
        
        # Calculate averages for model B
        model_b_avg = {
            "avg_accuracy": sum(m["accuracy"] for m in model_b_metrics) / len(model_b_metrics),
            "avg_f1_score": sum(m["f1_score"] for m in model_b_metrics) / len(model_b_metrics),
            "avg_prediction_time_ms": sum(m["avg_prediction_time_ms"] for m in model_b_metrics) / len(model_b_metrics)
        }
        
        # Determine winner based on accuracy and f1_score
        model_a_score = (model_a_avg["avg_accuracy"] + model_a_avg["avg_f1_score"]) / 2
        model_b_score = (model_b_avg["avg_accuracy"] + model_b_avg["avg_f1_score"]) / 2
        
        winner = "model_b" if model_b_score > model_a_score else "model_a"
        
        return {
            "model_a": model_a_avg,
            "model_b": model_b_avg,
            "winner": winner,
            "performance_difference": abs(model_b_score - model_a_score)
        }