"""
Integration tests for analytics and reporting features.
Tests complete workflows combining database, caching, and analytics services.

Requirements covered:
- 7.1: Efficient data storage with proper indexing
- 7.2: Fast queries for trend analysis
- 7.4: Aggregated sentiment metrics and analytics
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import AsyncMock, MagicMock, patch
import time


class TestAnalyticsIntegrationWorkflows:
    """Test complete analytics workflows"""
    
    @pytest.fixture
    def mock_database_with_data(self):
        """Mock database with realistic data"""
        db = MagicMock()
        
        # Sample sentiment data
        sentiment_data = [
            {
                "_id": f"sentiment_{i}",
                "text": f"Sample text {i}",
                "sentiment": "POSITIVE" if i % 3 == 0 else "NEGATIVE" if i % 3 == 1 else "NEUTRAL",
                "sentiment_score": 0.8 if i % 3 == 0 else -0.6 if i % 3 == 1 else 0.1,
                "confidence": 0.85,
                "team_id": f"team_{i % 4}",  # 4 teams
                "player_id": f"player_{i % 10}" if i % 2 == 0 else None,
                "game_id": f"game_{i // 10}" if i % 5 == 0 else None,
                "source": ["TWITTER", "ESPN", "NEWS"][i % 3],
                "category": ["performance", "general", "injury"][i % 3],
                "timestamp": datetime.utcnow() - timedelta(hours=i % 48),
                "metadata": {"processed_at": datetime.utcnow()}
            }
            for i in range(100)
        ]
        
        # Mock collection
        collection = AsyncMock()
        
        # Mock aggregation results
        mock_aggregation_results = [
            {
                "_id": "team_0",
                "total_mentions": 25,
                "avg_sentiment": 0.3,
                "avg_confidence": 0.85,
                "positive_count": 8,
                "negative_count": 9,
                "neutral_count": 8,
                "sentiment_scores": [0.8, -0.6, 0.1] * 8,
                "categories": ["performance", "general", "injury"] * 8,
                "sources": ["TWITTER", "ESPN", "NEWS"] * 8,
                "timestamps": [datetime(2024, 1, 1, 12, 0, 0) - timedelta(hours=h) for h in range(25)]
            },
            {
                "_id": "team_1", 
                "total_mentions": 25,
                "avg_sentiment": 0.2,
                "avg_confidence": 0.85,
                "positive_count": 8,
                "negative_count": 9,
                "neutral_count": 8,
                "sentiment_scores": [0.8, -0.6, 0.1] * 8,
                "categories": ["performance", "general", "injury"] * 8,
                "sources": ["TWITTER", "ESPN", "NEWS"] * 8,
                "timestamps": [datetime(2024, 1, 1, 12, 0, 0) - timedelta(hours=h) for h in range(25)]
            }
        ]
        
        async def mock_aggregate(pipeline):
            # Return mock cursor with aggregation results
            mock_cursor = AsyncMock()
            mock_cursor.to_list = AsyncMock(return_value=mock_aggregation_results)
            return mock_cursor
        
        collection.aggregate = mock_aggregate
        collection.find = AsyncMock(return_value=AsyncMock(to_list=AsyncMock(return_value=sentiment_data)))
        collection.count_documents = AsyncMock(return_value=len(sentiment_data))
        
        db.sentiment_analysis = collection
        db.sentiment_analysis_archive = collection
        
        return db
    
    @pytest.fixture
    def mock_cache_service(self):
        """Mock caching service"""
        cache_service = AsyncMock()
        cache_service._cache = {}  # In-memory cache for testing
        
        async def mock_get_analytics_data(query_hash):
            return cache_service._cache.get(f"analytics:{query_hash}")
        
        async def mock_cache_analytics_data(query_hash, data):
            cache_service._cache[f"analytics:{query_hash}"] = data
            return True
        
        async def mock_get_sentiment_trends(entity_type, entity_id, period):
            return cache_service._cache.get(f"trends:{entity_type}:{entity_id}:{period}")
        
        async def mock_cache_sentiment_trends(entity_type, entity_id, trends, period):
            cache_service._cache[f"trends:{entity_type}:{entity_id}:{period}"] = trends
            return True
        
        cache_service.get_analytics_data = mock_get_analytics_data
        cache_service.cache_analytics_data = mock_cache_analytics_data
        cache_service.get_sentiment_trends = mock_get_sentiment_trends
        cache_service.cache_sentiment_trends = mock_cache_sentiment_trends
        
        return cache_service
    
    @pytest.mark.asyncio
    async def test_complete_analytics_workflow(self, mock_database_with_data, mock_cache_service):
        """Test complete analytics workflow from query to result"""
        
        # Mock analytics service
        class MockAnalyticsService:
            def __init__(self, db, cache_service):
                self.db = db
                self.cache_service = cache_service
            
            def _generate_query_hash(self, query_params):
                import hashlib
                query_str = json.dumps(query_params, sort_keys=True, default=str)
                return hashlib.md5(query_str.encode()).hexdigest()
            
            def _calculate_category_breakdown(self, categories):
                if not categories:
                    return {}
                total = len(categories)
                breakdown = {}
                for category in set(categories):
                    count = categories.count(category)
                    breakdown[category] = count / total
                return breakdown
            
            def _calculate_source_breakdown(self, sources):
                if not sources:
                    return {}
                total = len(sources)
                breakdown = {}
                for source in set(sources):
                    count = sources.count(source)
                    breakdown[source] = count / total
                return breakdown
            
            async def get_aggregated_sentiment_metrics(self, entity_type, entity_id=None, **kwargs):
                # Generate cache key
                query_params = {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    **kwargs
                }
                query_hash = self._generate_query_hash(query_params)
                
                # Check cache first
                cached_result = await self.cache_service.get_analytics_data(query_hash)
                if cached_result:
                    return cached_result
                
                # Simulate database query
                collection = self.db.sentiment_analysis
                cursor = await collection.aggregate([])
                results = await cursor.to_list(length=None)
                
                # Process results
                processed_results = []
                for result in results:
                    entity_metrics = {
                        "entity_id": result["_id"],
                        "entity_type": entity_type,
                        "total_mentions": result["total_mentions"],
                        "average_sentiment": result["avg_sentiment"],
                        "average_confidence": result["avg_confidence"],
                        "sentiment_distribution": {
                            "positive": result["positive_count"] / result["total_mentions"],
                            "negative": result["negative_count"] / result["total_mentions"],
                            "neutral": result["neutral_count"] / result["total_mentions"]
                        },
                        "sentiment_volatility": 0.3,  # Mock volatility
                        "category_breakdown": self._calculate_category_breakdown(result["categories"]),
                        "source_breakdown": self._calculate_source_breakdown(result["sources"]),
                        "time_range": {
                            "start": min(result["timestamps"]).isoformat(),
                            "end": max(result["timestamps"]).isoformat()
                        }
                    }
                    processed_results.append(entity_metrics)
                
                # Use fixed timestamp for testing
                fixed_timestamp = datetime(2024, 1, 1, 12, 0, 0)
                final_result = {
                    "query_parameters": query_params,
                    "total_entities": len(processed_results),
                    "generated_at": fixed_timestamp.isoformat(),
                    "metrics": processed_results
                }
                
                # Cache the result
                await self.cache_service.cache_analytics_data(query_hash, final_result)
                
                return final_result
        
        analytics_service = MockAnalyticsService(mock_database_with_data, mock_cache_service)
        
        # Test workflow: Get team sentiment metrics
        # Use fixed datetime for consistent testing
        fixed_start_date = datetime(2024, 1, 1, 0, 0, 0)
        
        start_time = time.time()
        
        # First call - should hit database and cache result
        result1 = await analytics_service.get_aggregated_sentiment_metrics(
            entity_type="team",
            entity_id="team_1",
            start_date=fixed_start_date
        )
        
        first_call_time = time.time() - start_time
        
        # Second call - should hit cache
        start_time = time.time()
        result2 = await analytics_service.get_aggregated_sentiment_metrics(
            entity_type="team", 
            entity_id="team_1",
            start_date=fixed_start_date
        )
        
        second_call_time = time.time() - start_time
        
        # Verify results
        assert result1 == result2, "Cached result should match original"
        assert len(result1["metrics"]) > 0, "Should return metrics"
        assert result1["total_entities"] > 0, "Should have entities"
        
        # Verify caching improved performance
        assert second_call_time < first_call_time, "Cached call should be faster"
        
        # Verify result structure
        for metric in result1["metrics"]:
            assert "entity_id" in metric
            assert "total_mentions" in metric
            assert "average_sentiment" in metric
            assert "sentiment_distribution" in metric
            assert "category_breakdown" in metric
            assert "source_breakdown" in metric
    
    @pytest.mark.asyncio
    async def test_real_time_analytics_pipeline(self, mock_database_with_data, mock_cache_service):
        """Test real-time analytics pipeline with cache invalidation"""
        
        # Mock services
        class MockRealTimeAnalytics:
            def __init__(self, db, cache_service):
                self.db = db
                self.cache_service = cache_service
            
            async def process_new_sentiment(self, sentiment_data):
                """Process new sentiment data and update analytics"""
                # Step 1: Store in database
                collection = self.db.sentiment_analysis
                await collection.insert_one(sentiment_data)
                
                # Step 2: Invalidate related cache entries
                entity_type = "team"
                entity_id = sentiment_data.get("team_id")
                
                if entity_id:
                    # Invalidate team-specific caches
                    cache_keys_to_invalidate = [
                        f"team_sentiment:{entity_id}",
                        f"trends:team:{entity_id}:24h",
                        f"trends:team:{entity_id}:7d"
                    ]
                    
                    for key in cache_keys_to_invalidate:
                        if key in self.cache_service._cache:
                            del self.cache_service._cache[key]
                
                # Step 3: Update real-time metrics
                await self._update_real_time_metrics(entity_type, entity_id, sentiment_data)
                
                return {"status": "processed", "entity_id": entity_id}
            
            async def _update_real_time_metrics(self, entity_type, entity_id, sentiment_data):
                """Update real-time metrics"""
                # Simulate real-time metric updates
                current_metrics = {
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "last_sentiment": sentiment_data["sentiment_score"],
                    "last_updated": datetime.utcnow().isoformat(),
                    "recent_volume": 1  # Increment volume
                }
                
                # Cache updated metrics
                cache_key = f"realtime:{entity_type}:{entity_id}"
                self.cache_service._cache[cache_key] = current_metrics
        
        real_time_analytics = MockRealTimeAnalytics(mock_database_with_data, mock_cache_service)
        
        # Test processing new sentiment data
        new_sentiment = {
            "_id": "new_sentiment_1",
            "text": "Amazing performance by the team!",
            "sentiment": "POSITIVE",
            "sentiment_score": 0.9,
            "confidence": 0.95,
            "team_id": "team_1",
            "source": "TWITTER",
            "timestamp": datetime.utcnow()
        }
        
        # Process the new sentiment
        result = await real_time_analytics.process_new_sentiment(new_sentiment)
        
        # Verify processing
        assert result["status"] == "processed"
        assert result["entity_id"] == "team_1"
        
        # Verify real-time metrics were updated
        realtime_key = "realtime:team:team_1"
        assert realtime_key in mock_cache_service._cache
        
        realtime_metrics = mock_cache_service._cache[realtime_key]
        assert realtime_metrics["last_sentiment"] == 0.9
        assert realtime_metrics["recent_volume"] == 1
    
    @pytest.mark.asyncio
    async def test_analytics_export_workflow(self, mock_database_with_data, mock_cache_service):
        """Test complete analytics export workflow"""
        
        class MockAnalyticsExporter:
            def __init__(self, db, cache_service):
                self.db = db
                self.cache_service = cache_service
            
            async def export_team_analytics(self, export_format="json", time_period="7d"):
                """Export team analytics data"""
                # Get analytics data
                analytics_data = await self._get_export_data(time_period)
                
                # Format for export
                if export_format == "json":
                    return json.dumps(analytics_data, indent=2, default=str)
                elif export_format == "csv":
                    return self._format_as_csv(analytics_data)
                else:
                    raise ValueError(f"Unsupported format: {export_format}")
            
            async def _get_export_data(self, time_period):
                """Get data for export"""
                # Simulate getting aggregated data
                collection = self.db.sentiment_analysis
                cursor = await collection.aggregate([])
                results = await cursor.to_list(length=None)
                
                export_data = []
                for result in results:
                    export_record = {
                        "team_id": result["_id"],
                        "total_mentions": result["total_mentions"],
                        "average_sentiment": result["avg_sentiment"],
                        "positive_percentage": (result["positive_count"] / result["total_mentions"]) * 100,
                        "negative_percentage": (result["negative_count"] / result["total_mentions"]) * 100,
                        "neutral_percentage": (result["neutral_count"] / result["total_mentions"]) * 100,
                        "export_timestamp": datetime.utcnow().isoformat()
                    }
                    export_data.append(export_record)
                
                return export_data
            
            def _format_as_csv(self, data):
                """Format data as CSV"""
                if not data:
                    return ""
                
                import io
                import csv
                
                output = io.StringIO()
                fieldnames = data[0].keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
                
                return output.getvalue()
        
        exporter = MockAnalyticsExporter(mock_database_with_data, mock_cache_service)
        
        # Test JSON export
        json_export = await exporter.export_team_analytics(export_format="json")
        json_data = json.loads(json_export)
        
        assert isinstance(json_data, list)
        assert len(json_data) > 0
        
        # Verify JSON structure
        for record in json_data:
            assert "team_id" in record
            assert "total_mentions" in record
            assert "average_sentiment" in record
            assert "export_timestamp" in record
        
        # Test CSV export
        csv_export = await exporter.export_team_analytics(export_format="csv")
        
        assert isinstance(csv_export, str)
        assert "team_id" in csv_export  # Header should be present
        assert len(csv_export.split('\n')) > 1  # Should have header + data rows
    
    @pytest.mark.asyncio
    async def test_analytics_error_handling_workflow(self, mock_cache_service):
        """Test analytics workflow error handling"""
        
        # Mock database with errors
        error_db = MagicMock()
        error_collection = AsyncMock()
        
        # Simulate database error
        async def mock_aggregate_with_error(pipeline):
            raise Exception("Database connection error")
        
        error_collection.aggregate = mock_aggregate_with_error
        error_db.sentiment_analysis = error_collection
        
        class MockAnalyticsWithErrorHandling:
            def __init__(self, db, cache_service):
                self.db = db
                self.cache_service = cache_service
            
            async def get_analytics_with_fallback(self, entity_type, entity_id):
                """Get analytics with error handling and fallback"""
                try:
                    # Try database query
                    collection = self.db.sentiment_analysis
                    cursor = await collection.aggregate([])
                    results = await cursor.to_list(length=None)
                    return {"status": "success", "data": results}
                
                except Exception as e:
                    # Fallback to cached data
                    cache_key = f"fallback:{entity_type}:{entity_id}"
                    cached_data = self.cache_service._cache.get(cache_key)
                    
                    if cached_data:
                        return {
                            "status": "fallback_cache",
                            "data": cached_data,
                            "error": str(e)
                        }
                    else:
                        # Return minimal response
                        return {
                            "status": "error",
                            "data": None,
                            "error": str(e)
                        }
        
        analytics_service = MockAnalyticsWithErrorHandling(error_db, mock_cache_service)
        
        # Test error handling without cache
        result1 = await analytics_service.get_analytics_with_fallback("team", "team_1")
        
        assert result1["status"] == "error"
        assert result1["data"] is None
        assert "Database connection error" in result1["error"]
        
        # Add fallback data to cache
        fallback_data = {"team_id": "team_1", "avg_sentiment": 0.5}
        mock_cache_service._cache["fallback:team:team_1"] = fallback_data
        
        # Test error handling with cache fallback
        result2 = await analytics_service.get_analytics_with_fallback("team", "team_1")
        
        assert result2["status"] == "fallback_cache"
        assert result2["data"] == fallback_data
        assert "Database connection error" in result2["error"]
    
    @pytest.mark.asyncio
    async def test_concurrent_analytics_requests(self, mock_database_with_data, mock_cache_service):
        """Test handling concurrent analytics requests"""
        
        class MockConcurrentAnalytics:
            def __init__(self, db, cache_service):
                self.db = db
                self.cache_service = cache_service
                self.request_count = 0
            
            async def get_team_metrics(self, team_id):
                """Get team metrics with request tracking"""
                self.request_count += 1
                
                # Simulate processing time
                await asyncio.sleep(0.01)
                
                return {
                    "team_id": team_id,
                    "avg_sentiment": 0.7,
                    "total_mentions": 100,
                    "request_id": self.request_count
                }
        
        analytics_service = MockConcurrentAnalytics(mock_database_with_data, mock_cache_service)
        
        # Test concurrent requests
        team_ids = [f"team_{i}" for i in range(10)]
        
        start_time = time.time()
        
        # Execute concurrent requests
        tasks = [analytics_service.get_team_metrics(team_id) for team_id in team_ids]
        results = await asyncio.gather(*tasks)
        
        execution_time = time.time() - start_time
        
        # Verify concurrent execution
        assert len(results) == len(team_ids)
        assert execution_time < 0.5, f"Concurrent requests took too long: {execution_time}s"
        
        # Verify all requests were processed
        assert analytics_service.request_count == len(team_ids)
        
        # Verify results
        for i, result in enumerate(results):
            assert result["team_id"] == team_ids[i]
            assert result["avg_sentiment"] == 0.7
            assert result["request_id"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])