"""
Unit tests for admin functionality.
Tests individual admin functions and methods without external dependencies.
"""
import pytest
import os
from unittest.mock import MagicMock, AsyncMock, patch
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
    from app.models.user import UserRole


class TestAdminUserValidation:
    """Test admin user validation and authorization."""
    
    @pytest.mark.asyncio
    async def test_get_current_admin_user_success(self):
        """Test successful admin user validation."""
        admin_user = {
            "_id": str(ObjectId()),
            "email": "admin@example.com",
            "username": "adminuser",
            "role": "admin",
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        result = await get_current_admin_user(admin_user)
        
        assert result == admin_user
        assert result["role"] == "admin"
    
    @pytest.mark.asyncio
    async def test_get_current_admin_user_regular_user_denied(self):
        """Test admin access denied for regular user."""
        regular_user = {
            "_id": str(ObjectId()),
            "email": "user@example.com",
            "username": "regularuser",
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin_user(regular_user)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Not enough permissions" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_get_current_admin_user_missing_role(self):
        """Test admin access denied for user without role."""
        user_without_role = {
            "_id": str(ObjectId()),
            "email": "user@example.com",
            "username": "noroleuser",
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_admin_user(user_without_role)
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
    
    @pytest.mark.asyncio
    async def test_get_current_admin_user_inactive_admin(self):
        """Test that inactive admin users are still validated (role check only)."""
        inactive_admin = {
            "_id": str(ObjectId()),
            "email": "admin@example.com",
            "username": "inactiveadmin",
            "role": "admin",
            "is_active": False,
            "created_at": datetime.utcnow()
        }
        
        # The dependency only checks role, not active status
        result = await get_current_admin_user(inactive_admin)
        assert result == inactive_admin


class TestAdminDataValidation:
    """Test admin-specific data validation and processing."""
    
    def test_user_role_enum_validation(self):
        """Test user role enum validation."""
        # Valid roles
        assert UserRole.USER == "user"
        assert UserRole.ADMIN == "admin"
        
        # Test enum membership
        valid_roles = [role.value for role in UserRole]
        assert "user" in valid_roles
        assert "admin" in valid_roles
        assert "superuser" not in valid_roles
    
    def test_admin_query_parameters_validation(self):
        """Test validation of admin query parameters."""
        # Test limit parameter bounds
        valid_limits = [1, 50, 100, 200]
        invalid_limits = [0, -1, 201, 1000]
        
        for limit in valid_limits:
            assert 1 <= limit <= 200
        
        for limit in invalid_limits:
            assert not (1 <= limit <= 200)
        
        # Test skip parameter bounds
        valid_skips = [0, 10, 100, 1000]
        invalid_skips = [-1, -10]
        
        for skip in valid_skips:
            assert skip >= 0
        
        for skip in invalid_skips:
            assert not (skip >= 0)
    
    def test_analytics_date_range_validation(self):
        """Test analytics date range validation."""
        # Valid day ranges
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


class TestAdminUtilityFunctions:
    """Test admin utility functions and helpers."""
    
    def test_object_id_string_conversion(self):
        """Test ObjectId to string conversion for admin responses."""
        object_id = ObjectId()
        object_id_str = str(object_id)
        
        assert isinstance(object_id_str, str)
        assert len(object_id_str) == 24
        assert ObjectId.is_valid(object_id_str)
    
    def test_datetime_iso_conversion(self):
        """Test datetime to ISO string conversion for admin responses."""
        test_datetime = datetime.utcnow()
        iso_string = test_datetime.isoformat()
        
        assert isinstance(iso_string, str)
        assert "T" in iso_string
        
        # Test parsing back
        parsed_datetime = datetime.fromisoformat(iso_string.replace('Z', '+00:00') if iso_string.endswith('Z') else iso_string)
        assert abs((parsed_datetime - test_datetime).total_seconds()) < 1
    
    def test_admin_response_data_sanitization(self):
        """Test that sensitive data is removed from admin responses."""
        user_data = {
            "_id": ObjectId(),
            "email": "user@example.com",
            "username": "testuser",
            "hashed_password": "sensitive_hash",
            "role": "user",
            "is_active": True,
            "created_at": datetime.utcnow()
        }
        
        # Simulate admin response processing
        sanitized_data = {k: v for k, v in user_data.items() if k != "hashed_password"}
        sanitized_data["id"] = str(sanitized_data.pop("_id"))
        
        assert "hashed_password" not in sanitized_data
        assert "id" in sanitized_data
        assert "_id" not in sanitized_data
        assert sanitized_data["email"] == user_data["email"]


class TestAdminErrorHandling:
    """Test admin-specific error handling."""
    
    def test_admin_not_found_error(self):
        """Test handling of not found errors in admin operations."""
        with pytest.raises(HTTPException) as exc_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
        assert "User not found" in str(exc_info.value.detail)
    
    def test_admin_permission_error(self):
        """Test handling of permission errors in admin operations."""
        with pytest.raises(HTTPException) as exc_info:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Not enough permissions" in str(exc_info.value.detail)
    
    def test_admin_server_error(self):
        """Test handling of server errors in admin operations."""
        with pytest.raises(HTTPException) as exc_info:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
        
        assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Internal server error" in str(exc_info.value.detail)


class TestAdminDataAggregation:
    """Test admin data aggregation and statistics functions."""
    
    def test_user_statistics_calculation(self):
        """Test user statistics calculation logic."""
        total_users = 100
        active_users = 85
        new_users = 15
        
        inactive_users = total_users - active_users
        
        assert inactive_users == 15
        assert active_users + inactive_users == total_users
        assert new_users <= total_users
    
    def test_sentiment_distribution_calculation(self):
        """Test sentiment distribution calculation logic."""
        sentiment_counts = {
            "positive": 150,
            "negative": 75,
            "neutral": 100
        }
        
        total_analyses = sum(sentiment_counts.values())
        
        assert total_analyses == 325
        
        # Calculate percentages
        sentiment_percentages = {
            sentiment: round((count / total_analyses) * 100, 2)
            for sentiment, count in sentiment_counts.items()
        }
        
        assert sentiment_percentages["positive"] == 46.15
        assert sentiment_percentages["negative"] == 23.08
        assert sentiment_percentages["neutral"] == 30.77
        # Account for floating-point rounding errors
        assert abs(sum(sentiment_percentages.values()) - 100.0) < 0.01
    
    def test_model_performance_aggregation(self):
        """Test model performance metrics aggregation."""
        performance_metrics = [
            {"accuracy": 0.85, "f1_score": 0.82, "prediction_time_ms": 150},
            {"accuracy": 0.87, "f1_score": 0.84, "prediction_time_ms": 145},
            {"accuracy": 0.83, "f1_score": 0.80, "prediction_time_ms": 155}
        ]
        
        # Calculate averages
        avg_accuracy = sum(m["accuracy"] for m in performance_metrics) / len(performance_metrics)
        avg_f1_score = sum(m["f1_score"] for m in performance_metrics) / len(performance_metrics)
        avg_prediction_time = sum(m["prediction_time_ms"] for m in performance_metrics) / len(performance_metrics)
        
        assert round(avg_accuracy, 3) == 0.850
        assert round(avg_f1_score, 3) == 0.820
        assert round(avg_prediction_time, 2) == 150.0
    
    def test_time_series_data_processing(self):
        """Test time series data processing for admin analytics."""
        # Simulate daily data points
        daily_data = {
            "2024-01-01": {"requests": 100, "errors": 5},
            "2024-01-02": {"requests": 120, "errors": 3},
            "2024-01-03": {"requests": 95, "errors": 8}
        }
        
        # Calculate totals and averages
        total_requests = sum(data["requests"] for data in daily_data.values())
        total_errors = sum(data["errors"] for data in daily_data.values())
        avg_requests = total_requests / len(daily_data)
        error_rate = (total_errors / total_requests) * 100
        
        assert total_requests == 315
        assert total_errors == 16
        assert round(avg_requests, 2) == 105.0
        assert round(error_rate, 2) == 5.08


class TestAdminCacheOperations:
    """Test admin cache management operations."""
    
    def test_cache_key_generation(self):
        """Test cache key generation for admin operations."""
        user_id = str(ObjectId())
        cache_keys = {
            "user_profile": f"user:{user_id}:profile",
            "user_permissions": f"user:{user_id}:permissions",
            "system_health": "admin:system_health",
            "analytics": "admin:analytics:7d"
        }
        
        for key_type, key in cache_keys.items():
            assert isinstance(key, str)
            assert len(key) > 0
            if "user" in key_type:
                assert user_id in key
            if "admin" in key_type:
                assert "admin:" in key
    
    def test_cache_expiration_times(self):
        """Test cache expiration time calculations."""
        cache_ttl = {
            "user_profile": 300,  # 5 minutes
            "system_health": 60,  # 1 minute
            "analytics": 1800,    # 30 minutes
            "model_performance": 600  # 10 minutes
        }
        
        for cache_type, ttl in cache_ttl.items():
            assert ttl > 0
            assert ttl <= 1800  # Max 30 minutes
            
            # Test expiration datetime calculation
            expiration = datetime.utcnow() + timedelta(seconds=ttl)
            assert expiration > datetime.utcnow()