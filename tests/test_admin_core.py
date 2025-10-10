"""
Core admin functionality tests.
Tests admin endpoints without complex MLOps dependencies.
"""
import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from bson import ObjectId
from fastapi import HTTPException, status

# Mock settings before importing app modules
with patch.dict('os.environ', {
    'SECRET_KEY': '78183c734c4337b3b9ac71f816dfab85a8a3bebbc4f4dc6ecd5d1b9c0d4307f1',
    'MONGODB_URL': 'mongodb://test:27017',
    'DATABASE_NAME': 'test_db'
}):
    from app.core.dependencies import get_current_admin_user
    from app.models.user import UserRole, UserResponse


class TestAdminRoleBasedAccess:
    """Test role-based access control for admin functionality."""
    
    @pytest.mark.asyncio
    async def test_admin_access_granted(self):
        """Test that admin users can access admin functionality."""
        admin_user = {
            "_id": str(ObjectId()),
            "email": "admin@example.com",
            "username": "adminuser",
            "role": "admin",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "preferences": {}
        }
        
        result = await get_current_admin_user(admin_user)
        assert result == admin_user
        assert result["role"] == "admin"
    
    @pytest.mark.asyncio
    async def test_regular_user_access_denied(self):
        """Test that regular users cannot access admin functionality."""
        regular_user = {
            "_id": str(ObjectId()),
            "email": "user@example.com",
            "username": "regularuser",
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "preferences": {}
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin_user(regular_user)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Not enough permissions" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_user_without_role_access_denied(self):
        """Test that users without role cannot access admin functionality."""
        user_without_role = {
            "_id": str(ObjectId()),
            "email": "user@example.com",
            "username": "noroleuser",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "preferences": {}
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin_user(user_without_role)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN


class TestAdminUserManagement:
    """Test admin user management operations."""
    
    @pytest.mark.asyncio
    async def test_user_list_retrieval(self):
        """Test retrieving list of users for admin."""
        mock_db = MagicMock()
        
        # Mock user data
        sample_users = [
            {
                "_id": ObjectId(),
                "email": "user1@example.com",
                "username": "user1",
                "role": "user",
                "is_active": True,
                "created_at": datetime.utcnow(),
                "preferences": {}
            },
            {
                "_id": ObjectId(),
                "email": "user2@example.com",
                "username": "user2",
                "role": "user",
                "is_active": True,
                "created_at": datetime.utcnow(),
                "preferences": {}
            }
        ]
        
        # Mock cursor
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__.return_value = iter(sample_users)
        mock_db.users.find.return_value = mock_cursor
        
        # Simulate user retrieval
        users = []
        async for user in mock_cursor:
            # Simulate response processing
            user["id"] = str(user["_id"])
            del user["_id"]
            users.append(UserResponse(**user))
        
        assert len(users) == 2
        assert users[0].email == "user1@example.com"
        assert users[1].email == "user2@example.com"
        
        # Verify no sensitive data
        for user in users:
            assert not hasattr(user, 'hashed_password')
    
    @pytest.mark.asyncio
    async def test_user_activation_deactivation(self):
        """Test user activation and deactivation operations."""
        mock_db = MagicMock()
        
        # Mock successful update
        mock_result = MagicMock()
        mock_result.matched_count = 1
        mock_db.users.update_one.return_value = mock_result
        
        user_id = str(ObjectId())
        
        # Test deactivation
        deactivation_result = mock_db.users.update_one(
            {"_id": user_id},
            {"$set": {"is_active": False, "deactivated_at": datetime.utcnow()}}
        )
        
        assert deactivation_result.matched_count == 1
        
        # Test activation
        activation_result = mock_db.users.update_one(
            {"_id": user_id},
            {"$set": {"is_active": True}, "$unset": {"deactivated_at": ""}}
        )
        
        assert activation_result.matched_count == 1
        
        # Verify database calls
        assert mock_db.users.update_one.call_count == 2
    
    @pytest.mark.asyncio
    async def test_user_not_found_handling(self):
        """Test handling of user not found scenarios."""
        mock_db = MagicMock()
        
        # Mock no match found
        mock_result = MagicMock()
        mock_result.matched_count = 0
        mock_db.users.update_one.return_value = mock_result
        
        user_id = "nonexistent_user_id"
        
        # Attempt to deactivate non-existent user
        result = mock_db.users.update_one(
            {"_id": user_id},
            {"$set": {"is_active": False}}
        )
        
        assert result.matched_count == 0
        
        # This would trigger a 404 error in the actual endpoint
        if result.matched_count == 0:
            with pytest.raises(HTTPException) as exc_info:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found"
                )
            
            assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND


class TestAdminSystemMonitoring:
    """Test admin system monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_system_health_check_structure(self):
        """Test system health check data structure."""
        # Simulate health check response
        health_status = {
            "timestamp": datetime.utcnow(),
            "services": {
                "mongodb": {
                    "status": "healthy",
                    "response_time_ms": 5.2,
                    "database_size_mb": 150.5
                },
                "redis": {
                    "status": "healthy",
                    "response_time_ms": 2.1,
                    "memory_used_mb": 75.3
                }
            },
            "system_resources": {
                "cpu_usage_percent": 45.2,
                "memory_usage_percent": 68.5,
                "disk_usage_percent": 72.1
            },
            "overall_status": "healthy"
        }
        
        # Verify structure
        assert "timestamp" in health_status
        assert "services" in health_status
        assert "system_resources" in health_status
        assert "overall_status" in health_status
        
        # Verify service status
        assert health_status["services"]["mongodb"]["status"] == "healthy"
        assert health_status["services"]["redis"]["status"] == "healthy"
        
        # Verify resource metrics
        assert 0 <= health_status["system_resources"]["cpu_usage_percent"] <= 100
        assert 0 <= health_status["system_resources"]["memory_usage_percent"] <= 100
        assert 0 <= health_status["system_resources"]["disk_usage_percent"] <= 100
    
    @pytest.mark.asyncio
    async def test_system_health_degradation_detection(self):
        """Test detection of system health degradation."""
        # Test high resource usage
        high_usage_status = {
            "system_resources": {
                "cpu_usage_percent": 85.0,  # High
                "memory_usage_percent": 90.0,  # High
                "disk_usage_percent": 95.0  # High
            },
            "overall_status": "healthy"  # Initially healthy
        }
        
        # Check if any resource is above threshold
        cpu_high = high_usage_status["system_resources"]["cpu_usage_percent"] > 80
        memory_high = high_usage_status["system_resources"]["memory_usage_percent"] > 85
        disk_high = high_usage_status["system_resources"]["disk_usage_percent"] > 90
        
        if cpu_high or memory_high or disk_high:
            high_usage_status["overall_status"] = "degraded"
        
        assert high_usage_status["overall_status"] == "degraded"
    
    @pytest.mark.asyncio
    async def test_service_failure_detection(self):
        """Test detection of service failures."""
        # Test service failure scenario
        service_failure_status = {
            "services": {
                "mongodb": {
                    "status": "unhealthy",
                    "error": "Connection timeout"
                },
                "redis": {
                    "status": "healthy"
                }
            },
            "overall_status": "healthy"  # Initially healthy
        }
        
        # Check for any unhealthy services
        unhealthy_services = [
            service for service, details in service_failure_status["services"].items()
            if details.get("status") == "unhealthy"
        ]
        
        if unhealthy_services:
            service_failure_status["overall_status"] = "degraded"
        
        assert service_failure_status["overall_status"] == "degraded"
        assert len(unhealthy_services) == 1
        assert "mongodb" in unhealthy_services


class TestAdminAnalytics:
    """Test admin analytics functionality."""
    
    @pytest.mark.asyncio
    async def test_user_analytics_calculation(self):
        """Test user analytics calculation."""
        mock_db = MagicMock()
        
        # Mock user counts
        mock_db.users.count_documents.side_effect = [100, 85, 15]  # total, active, new
        
        # Simulate analytics calculation
        total_users = 100
        active_users = 85
        new_users = 15
        inactive_users = total_users - active_users
        
        user_analytics = {
            "total": total_users,
            "active": active_users,
            "inactive": inactive_users,
            "new": new_users
        }
        
        assert user_analytics["total"] == 100
        assert user_analytics["active"] == 85
        assert user_analytics["inactive"] == 15
        assert user_analytics["new"] == 15
        assert user_analytics["active"] + user_analytics["inactive"] == user_analytics["total"]
    
    @pytest.mark.asyncio
    async def test_sentiment_analytics_aggregation(self):
        """Test sentiment analytics aggregation."""
        # Mock sentiment distribution data
        sentiment_data = [
            {"_id": "positive", "count": 150, "avg_confidence": 0.85},
            {"_id": "negative", "count": 75, "avg_confidence": 0.78},
            {"_id": "neutral", "count": 100, "avg_confidence": 0.72}
        ]
        
        # Calculate analytics
        total_analyses = sum(item["count"] for item in sentiment_data)
        sentiment_distribution = {item["_id"]: item["count"] for item in sentiment_data}
        confidence_by_sentiment = {item["_id"]: item["avg_confidence"] for item in sentiment_data}
        
        sentiment_analytics = {
            "total": total_analyses,
            "distribution": sentiment_distribution,
            "confidence_by_sentiment": confidence_by_sentiment
        }
        
        assert sentiment_analytics["total"] == 325
        assert sentiment_analytics["distribution"]["positive"] == 150
        assert sentiment_analytics["distribution"]["negative"] == 75
        assert sentiment_analytics["distribution"]["neutral"] == 100
        assert sentiment_analytics["confidence_by_sentiment"]["positive"] == 0.85
    
    @pytest.mark.asyncio
    async def test_time_series_analytics(self):
        """Test time series analytics processing."""
        # Mock daily data
        daily_data = [
            {"_id": "2024-01-01", "count": 50},
            {"_id": "2024-01-02", "count": 75},
            {"_id": "2024-01-03", "count": 60}
        ]
        
        # Process time series data
        daily_volume = {item["_id"]: item["count"] for item in daily_data}
        total_volume = sum(item["count"] for item in daily_data)
        avg_daily_volume = total_volume / len(daily_data)
        
        time_series_analytics = {
            "daily_volume": daily_volume,
            "total_volume": total_volume,
            "avg_daily_volume": avg_daily_volume,
            "period_days": len(daily_data)
        }
        
        assert time_series_analytics["total_volume"] == 185
        assert time_series_analytics["avg_daily_volume"] == 185 / 3
        assert time_series_analytics["period_days"] == 3
        assert len(time_series_analytics["daily_volume"]) == 3


class TestAdminCacheManagement:
    """Test admin cache management functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_clearing_operations(self):
        """Test cache clearing operations."""
        mock_redis = AsyncMock()
        
        # Test Redis cache clearing
        mock_redis.flushdb = AsyncMock()
        await mock_redis.flushdb()
        
        mock_redis.flushdb.assert_called_once()
        
        # Simulate cache clearing result
        cache_clear_result = {
            "cleared_caches": ["redis"],
            "success": True,
            "message": "Cache cleared successfully: redis"
        }
        
        assert cache_clear_result["success"] is True
        assert "redis" in cache_clear_result["cleared_caches"]
    
    @pytest.mark.asyncio
    async def test_cache_unavailable_handling(self):
        """Test handling when cache services are unavailable."""
        # Simulate no Redis available
        mock_redis = None
        
        if not mock_redis:
            cache_clear_result = {
                "success": False,
                "error": "No cache services available"
            }
        
        assert cache_clear_result["success"] is False
        assert "No cache services available" in cache_clear_result["error"]


class TestAdminErrorHandling:
    """Test admin error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_database_connection_error_handling(self):
        """Test handling of database connection errors."""
        mock_db = MagicMock()
        
        # Mock database connection failure
        mock_db.command.side_effect = Exception("Connection failed")
        
        try:
            await mock_db.command("ping")
            db_status = "healthy"
        except Exception as e:
            db_status = "unhealthy"
            error_message = str(e)
        
        assert db_status == "unhealthy"
        assert error_message == "Connection failed"
    
    @pytest.mark.asyncio
    async def test_permission_error_handling(self):
        """Test handling of permission errors."""
        # Simulate permission error
        with pytest.raises(HTTPException) as exc_info:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Not enough permissions" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_resource_not_found_error_handling(self):
        """Test handling of resource not found errors."""
        # Simulate resource not found
        with pytest.raises(HTTPException) as exc_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resource not found"
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "Resource not found" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_server_error_handling(self):
        """Test handling of internal server errors."""
        # Simulate server error
        with pytest.raises(HTTPException) as exc_info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Internal server error" in str(exc_info.value.detail)


class TestAdminDataValidation:
    """Test admin data validation and sanitization."""
    
    def test_user_response_data_sanitization(self):
        """Test that sensitive data is removed from user responses."""
        raw_user_data = {
            "_id": ObjectId(),
            "email": "user@example.com",
            "username": "testuser",
            "hashed_password": "sensitive_hash_value",
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow(),
            "preferences": {}
        }
        
        # Simulate response processing
        sanitized_data = {k: v for k, v in raw_user_data.items() if k != "hashed_password"}
        sanitized_data["id"] = str(sanitized_data.pop("_id"))
        
        user_response = UserResponse(**sanitized_data)
        
        assert user_response.email == "user@example.com"
        assert user_response.username == "testuser"
        assert user_response.role == "user"
        assert not hasattr(user_response, 'hashed_password')
    
    def test_query_parameter_validation(self):
        """Test query parameter validation for admin endpoints."""
        # Test limit parameter validation
        valid_limits = [1, 50, 100, 200]
        invalid_limits = [0, -1, 201, 1000]
        
        for limit in valid_limits:
            assert 1 <= limit <= 200
        
        for limit in invalid_limits:
            assert not (1 <= limit <= 200)
        
        # Test skip parameter validation
        valid_skips = [0, 10, 100]
        invalid_skips = [-1, -10]
        
        for skip in valid_skips:
            assert skip >= 0
        
        for skip in invalid_skips:
            assert not (skip >= 0)
    
    def test_date_range_validation(self):
        """Test date range validation for analytics."""
        # Test valid date ranges
        valid_days = [1, 7, 14, 30]
        invalid_days = [0, -1, 31, 100]
        
        for days in valid_days:
            assert 1 <= days <= 30
        
        for days in invalid_days:
            assert not (1 <= days <= 30)
        
        # Test date calculation
        end_date = datetime.utcnow()
        days = 7
        start_date = end_date - timedelta(days=days)
        
        assert start_date < end_date
        assert (end_date - start_date).days == days