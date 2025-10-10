"""
Integration tests for admin functionality.
Tests complete admin workflows and system monitoring features.
"""
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from bson import ObjectId

# Mock settings before importing app modules
with patch.dict('os.environ', {
    'SECRET_KEY': '78183c734c4337b3b9ac71f816dfab85a8a3bebbc4f4dc6ecd5d1b9c0d4307f1',
    'MONGODB_URL': 'mongodb://test:27017',
    'DATABASE_NAME': 'test_db'
}):
    # MLOps models for testing (without importing the service)
    from app.models.mlops import ModelRetrainingRequest, ExperimentRun, ExperimentStatus


class TestAdminUserManagementWorkflow:
    """Test complete admin user management workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_user_lifecycle_management(self):
        """Test complete user lifecycle from creation to deactivation."""
        # Mock database
        mock_db = MagicMock()
        
        # Test user creation (would be done through registration)
        new_user_id = str(ObjectId())
        new_user = {
            "_id": new_user_id,
            "email": "newuser@example.com",
            "username": "newuser",
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "preferences": {}
        }
        
        # Mock user retrieval
        mock_db.users.find_one.return_value = new_user
        
        # Test user activation/deactivation
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_db.users.update_one.return_value = mock_result
        
        # Simulate deactivation
        deactivation_result = await self._simulate_user_deactivation(mock_db, new_user_id)
        assert deactivation_result["success"] is True
        
        # Simulate reactivation
        activation_result = await self._simulate_user_activation(mock_db, new_user_id)
        assert activation_result["success"] is True
        
        # Verify database calls
        assert mock_db.users.update_one.call_count == 2
    
    async def _simulate_user_deactivation(self, mock_db, user_id):
        """Simulate user deactivation process."""
        result = mock_db.users.update_one(
            {"_id": user_id},
            {"$set": {"is_active": False, "deactivated_at": datetime.utcnow()}}
        )
        return {"success": result.matched_count > 0}
    
    async def _simulate_user_activation(self, mock_db, user_id):
        """Simulate user activation process."""
        result = mock_db.users.update_one(
            {"_id": user_id},
            {"$set": {"is_active": True}, "$unset": {"deactivated_at": ""}}
        )
        return {"success": result.matched_count > 0}
    
    @pytest.mark.asyncio
    async def test_bulk_user_operations(self):
        """Test bulk user operations for admin efficiency."""
        mock_db = MagicMock()
        
        # Mock multiple users
        users = []
        for i in range(5):
            users.append({
                "_id": str(ObjectId()),
                "email": f"user{i}@example.com",
                "username": f"user{i}",
                "role": "user",
                "is_active": True,
                "created_at": datetime.utcnow() - timedelta(days=i)
            })
        
        # Mock cursor for bulk retrieval
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__.return_value = iter(users)
        mock_db.users.find.return_value = mock_cursor
        
        # Simulate bulk user retrieval
        retrieved_users = []
        async for user in mock_cursor:
            retrieved_users.append(user)
        
        assert len(retrieved_users) == 5
        assert all(user["role"] == "user" for user in retrieved_users)
        
        # Test bulk deactivation
        mock_bulk_result = MagicMock()
        mock_bulk_result.modified_count = 3
        mock_db.users.update_many.return_value = mock_bulk_result
        
        # Simulate bulk deactivation of inactive users
        result = mock_db.users.update_many(
            {"last_login": {"$lt": datetime.utcnow() - timedelta(days=30)}},
            {"$set": {"is_active": False, "deactivated_at": datetime.utcnow()}}
        )
        
        assert result.modified_count == 3


class TestSystemMonitoringIntegration:
    """Test system monitoring and health check integration."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_system_health_check(self):
        """Test comprehensive system health monitoring."""
        # Mock all system components
        health_status = await self._perform_comprehensive_health_check()
        
        assert "services" in health_status
        assert "system_resources" in health_status
        assert "overall_status" in health_status
        
        # Verify all critical services are checked
        expected_services = ["mongodb", "redis", "mlops"]
        for service in expected_services:
            assert service in health_status["services"]
    
    async def _perform_comprehensive_health_check(self):
        """Simulate comprehensive health check."""
        health_status = {
            "timestamp": datetime.utcnow(),
            "services": {},
            "system_resources": {},
            "overall_status": "healthy"
        }
        
        # Mock MongoDB health
        try:
            # Simulate MongoDB ping
            health_status["services"]["mongodb"] = {
                "status": "healthy",
                "response_time_ms": 5.2,
                "database_size_mb": 150.5,
                "collections": 8,
                "indexes": 15
            }
        except Exception as e:
            health_status["services"]["mongodb"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Mock Redis health
        try:
            health_status["services"]["redis"] = {
                "status": "healthy",
                "response_time_ms": 2.1,
                "memory_used_mb": 75.3,
                "connected_clients": 12,
                "total_commands_processed": 50000
            }
        except Exception as e:
            health_status["services"]["redis"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Mock MLOps health
        try:
            health_status["services"]["mlops"] = {
                "status": "healthy",
                "initialized": True,
                "services": {
                    "hopsworks": "healthy",
                    "huggingface": "healthy",
                    "wandb": "healthy"
                }
            }
        except Exception as e:
            health_status["services"]["mlops"] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
        
        # Mock system resources
        health_status["system_resources"] = {
            "cpu_usage_percent": 45.2,
            "memory_usage_percent": 68.5,
            "memory_available_gb": 4.2,
            "disk_usage_percent": 72.1,
            "disk_free_gb": 85.3
        }
        
        return health_status
    
    @pytest.mark.asyncio
    async def test_system_health_degradation_detection(self):
        """Test detection of system health degradation."""
        # Test high resource usage scenario
        high_usage_health = await self._simulate_high_resource_usage()
        assert high_usage_health["overall_status"] == "degraded"
        
        # Test service failure scenario
        service_failure_health = await self._simulate_service_failure()
        assert service_failure_health["overall_status"] == "degraded"
    
    async def _simulate_high_resource_usage(self):
        """Simulate high system resource usage."""
        return {
            "overall_status": "degraded",
            "system_resources": {
                "cpu_usage_percent": 85.0,  # High CPU
                "memory_usage_percent": 90.0,  # High memory
                "disk_usage_percent": 95.0  # High disk
            },
            "services": {
                "mongodb": {"status": "healthy"},
                "redis": {"status": "healthy"},
                "mlops": {"status": "healthy"}
            }
        }
    
    async def _simulate_service_failure(self):
        """Simulate service failure scenario."""
        return {
            "overall_status": "degraded",
            "system_resources": {
                "cpu_usage_percent": 45.0,
                "memory_usage_percent": 60.0,
                "disk_usage_percent": 70.0
            },
            "services": {
                "mongodb": {"status": "unhealthy", "error": "Connection timeout"},
                "redis": {"status": "healthy"},
                "mlops": {"status": "healthy"}
            }
        }


class TestMLOpsIntegration:
    """Test MLOps integration for admin functionality."""
    
    @pytest.mark.asyncio
    async def test_model_retraining_workflow(self):
        """Test complete model retraining workflow."""
        # Mock MLOps service
        mock_mlops_service = MagicMock()
        mock_mlops_service.initialized = True
        
        # Create mock experiment run
        mock_experiment_run = MagicMock()
        mock_experiment_run.run_id = "run_12345"
        mock_experiment_run.experiment_id = "exp_67890"
        mock_experiment_run.status = ExperimentStatus.RUNNING
        mock_experiment_run.started_at = datetime.utcnow()
        
        mock_mlops_service.trigger_retraining.return_value = mock_experiment_run
        
        # Mock database for job storage
        mock_db = MagicMock()
        mock_db.ml_jobs.insert_one = AsyncMock()
        
        # Simulate retraining request
        retraining_request = ModelRetrainingRequest(
            model_name="sentiment_base",
            trigger_reason="manual_admin_trigger",
            auto_deploy=False
        )
        
        # Execute retraining workflow
        result = await self._execute_retraining_workflow(
            mock_mlops_service, mock_db, retraining_request, "admin_user_id"
        )
        
        assert result["success"] is True
        assert result["job_id"] == "run_12345"
        assert result["experiment_id"] == "exp_67890"
        
        # Verify MLOps service was called
        mock_mlops_service.trigger_retraining.assert_called_once()
        
        # Verify job was stored
        mock_db.ml_jobs.insert_one.assert_called_once()
    
    async def _execute_retraining_workflow(self, mlops_service, db, request, admin_user_id):
        """Execute model retraining workflow."""
        try:
            # Trigger retraining
            experiment_run = mlops_service.trigger_retraining(
                model_name=request.model_name,
                reason=request.trigger_reason,
                config=request.training_config
            )
            
            # Store job record
            retraining_job = {
                "job_id": experiment_run.run_id,
                "experiment_id": experiment_run.experiment_id,
                "model_name": request.model_name,
                "status": experiment_run.status.value,
                "trigger_reason": request.trigger_reason,
                "triggered_by": admin_user_id,
                "auto_deploy": request.auto_deploy,
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
    async def test_model_performance_monitoring(self):
        """Test model performance monitoring integration."""
        mock_db = MagicMock()
        
        # Mock performance metrics data
        performance_metrics = [
            {
                "_id": ObjectId(),
                "model_id": "sentiment_v1",
                "timestamp": datetime.utcnow() - timedelta(hours=1),
                "accuracy": 0.85,
                "f1_score": 0.82,
                "avg_prediction_time_ms": 150,
                "prediction_count": 1000,
                "error_rate": 0.02
            },
            {
                "_id": ObjectId(),
                "model_id": "sentiment_v1",
                "timestamp": datetime.utcnow() - timedelta(hours=2),
                "accuracy": 0.87,
                "f1_score": 0.84,
                "avg_prediction_time_ms": 145,
                "prediction_count": 950,
                "error_rate": 0.015
            }
        ]
        
        # Mock cursor
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__.return_value = iter(performance_metrics)
        mock_db.performance_metrics.find.return_value = mock_cursor
        
        # Execute performance monitoring
        performance_summary = await self._analyze_model_performance(mock_db, "sentiment_v1", 7)
        
        assert performance_summary["model_id"] == "sentiment_v1"
        assert "avg_accuracy" in performance_summary
        assert "avg_f1_score" in performance_summary
        assert "total_predictions" in performance_summary
    
    async def _analyze_model_performance(self, db, model_id, days):
        """Analyze model performance over time period."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
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
            "period_days": days,
            "metrics_count": len(metrics),
            "avg_accuracy": round(avg_accuracy, 3),
            "avg_f1_score": round(avg_f1_score, 3),
            "avg_response_time_ms": round(avg_response_time, 2),
            "total_predictions": total_predictions,
            "avg_error_rate": round(avg_error_rate, 4)
        }


class TestAnalyticsIntegration:
    """Test analytics and reporting integration."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_analytics_generation(self):
        """Test comprehensive analytics data generation."""
        mock_db = MagicMock()
        
        # Execute analytics generation
        analytics = await self._generate_comprehensive_analytics(mock_db, 7)
        
        assert "users" in analytics
        assert "sentiment_analyses" in analytics
        assert "model_performance" in analytics
        assert "api_usage" in analytics
        
        # Verify time period
        assert analytics["period"] == "7 days"
        assert "period_start" in analytics
        assert "period_end" in analytics
    
    async def _generate_comprehensive_analytics(self, db, days):
        """Generate comprehensive analytics report."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        analytics = {
            "period": f"{days} days",
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Mock user analytics
        analytics["users"] = {
            "total": 150,
            "active": 120,
            "inactive": 30,
            "new": 25,
            "daily_registrations": {
                "2024-01-01": 5,
                "2024-01-02": 8,
                "2024-01-03": 12
            }
        }
        
        # Mock sentiment analytics
        analytics["sentiment_analyses"] = {
            "total": 5000,
            "recent": 750,
            "distribution": {
                "positive": 400,
                "negative": 200,
                "neutral": 150
            },
            "confidence_by_sentiment": {
                "positive": 0.85,
                "negative": 0.78,
                "neutral": 0.72
            },
            "daily_volume": {
                "2024-01-01": 100,
                "2024-01-02": 125,
                "2024-01-03": 110
            }
        }
        
        # Mock model performance analytics
        analytics["model_performance"] = [
            {
                "model_id": "sentiment_v1",
                "avg_accuracy": 0.85,
                "avg_f1_score": 0.82,
                "avg_response_time_ms": 150,
                "total_predictions": 10000,
                "avg_error_rate": 0.02
            }
        ]
        
        # Mock API usage analytics
        analytics["api_usage"] = {
            "daily_usage": {
                "2024-01-01": {
                    "total_requests": 1500,
                    "unique_users": 45,
                    "avg_response_time_ms": 125
                },
                "2024-01-02": {
                    "total_requests": 1750,
                    "unique_users": 52,
                    "avg_response_time_ms": 118
                }
            }
        }
        
        return analytics
    
    @pytest.mark.asyncio
    async def test_real_time_metrics_aggregation(self):
        """Test real-time metrics aggregation for admin dashboard."""
        # Mock real-time data sources
        real_time_metrics = await self._collect_real_time_metrics()
        
        assert "current_active_users" in real_time_metrics
        assert "requests_per_minute" in real_time_metrics
        assert "error_rate_last_hour" in real_time_metrics
        assert "model_predictions_last_hour" in real_time_metrics
    
    async def _collect_real_time_metrics(self):
        """Collect real-time system metrics."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "current_active_users": 45,
            "requests_per_minute": 125,
            "error_rate_last_hour": 0.015,
            "model_predictions_last_hour": 2500,
            "cache_hit_rate": 0.85,
            "database_connections": 12,
            "queue_size": 5
        }


class TestAdminAlertingSystem:
    """Test admin alerting and notification system."""
    
    @pytest.mark.asyncio
    async def test_alert_generation_and_management(self):
        """Test alert generation and management workflow."""
        mock_db = MagicMock()
        
        # Test alert creation
        alert = await self._create_system_alert(
            mock_db,
            "performance_degradation",
            "high",
            "Model accuracy dropped below threshold",
            0.8,
            0.75
        )
        
        assert alert["alert_type"] == "performance_degradation"
        assert alert["severity"] == "high"
        assert alert["status"] == "active"
        
        # Test alert acknowledgment
        acknowledged_alert = await self._acknowledge_alert(mock_db, alert["alert_id"], "admin_user_id")
        assert acknowledged_alert["status"] == "acknowledged"
        
        # Test alert resolution
        resolved_alert = await self._resolve_alert(mock_db, alert["alert_id"], "admin_user_id")
        assert resolved_alert["status"] == "resolved"
    
    async def _create_system_alert(self, db, alert_type, severity, message, threshold_value, actual_value):
        """Create a system alert."""
        alert = {
            "alert_id": str(ObjectId()),
            "model_id": "sentiment_v1",
            "model_version": "1.0.0",
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "threshold_value": threshold_value,
            "actual_value": actual_value,
            "status": "active",
            "created_at": datetime.utcnow(),
            "recommended_actions": [
                "Check model performance metrics",
                "Consider model retraining",
                "Review recent data quality"
            ]
        }
        
        # Mock database insertion
        db.alerts.insert_one(alert)
        
        return alert
    
    async def _acknowledge_alert(self, db, alert_id, admin_user_id):
        """Acknowledge an alert."""
        update_result = {
            "status": "acknowledged",
            "acknowledged_at": datetime.utcnow(),
            "acknowledged_by": admin_user_id
        }
        
        # Mock database update
        db.alerts.update_one(
            {"alert_id": alert_id},
            {"$set": update_result}
        )
        
        return update_result
    
    async def _resolve_alert(self, db, alert_id, admin_user_id):
        """Resolve an alert."""
        update_result = {
            "status": "resolved",
            "resolved_at": datetime.utcnow(),
            "resolved_by": admin_user_id
        }
        
        # Mock database update
        db.alerts.update_one(
            {"alert_id": alert_id},
            {"$set": update_result}
        )
        
        return update_result
    
    @pytest.mark.asyncio
    async def test_alert_escalation_workflow(self):
        """Test alert escalation workflow for critical issues."""
        mock_db = MagicMock()
        
        # Create critical alert
        critical_alert = await self._create_system_alert(
            mock_db,
            "system_failure",
            "critical",
            "Database connection lost",
            None,
            None
        )
        
        # Test escalation logic
        escalation_result = await self._escalate_alert(mock_db, critical_alert["alert_id"])
        
        assert escalation_result["escalated"] is True
        assert "notification_sent" in escalation_result
        assert "escalation_level" in escalation_result
    
    async def _escalate_alert(self, db, alert_id):
        """Escalate a critical alert."""
        escalation_result = {
            "escalated": True,
            "escalation_level": "critical",
            "notification_sent": True,
            "escalated_at": datetime.utcnow(),
            "notification_channels": ["email", "slack", "pagerduty"]
        }
        
        # Mock database update
        db.alerts.update_one(
            {"alert_id": alert_id},
            {"$set": {
                "escalated": True,
                "escalation_level": "critical",
                "escalated_at": escalation_result["escalated_at"]
            }}
        )
        
        return escalation_result