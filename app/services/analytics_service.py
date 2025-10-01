"""
Advanced analytics and reporting service for NFL sentiment analysis.
Provides aggregated metrics, trend analysis, and export functionality.
"""

import asyncio
import hashlib
import json
import csv
import io
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
import logging
from app.core.database import get_database
from app.services.caching_service import get_caching_service
from app.models.sentiment import (
    SentimentLabel, SentimentCategory, DataSource,
    SentimentTrend, TeamSentiment, PlayerSentiment, GameSentiment
)

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Advanced analytics service for sentiment data"""
    
    def __init__(self, db: Optional[AsyncIOMotorDatabase] = None):
        self.db = db
        self.cache_service = None
    
    async def get_database(self) -> AsyncIOMotorDatabase:
        """Get database instance"""
        if not self.db:
            self.db = await get_database()
        return self.db
    
    async def get_cache_service(self):
        """Get caching service instance"""
        if not self.cache_service:
            self.cache_service = await get_caching_service()
        return self.cache_service
    
    def _generate_query_hash(self, query_params: Dict[str, Any]) -> str:
        """Generate hash for caching query results"""
        query_str = json.dumps(query_params, sort_keys=True, default=str)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    async def get_aggregated_sentiment_metrics(
        self,
        entity_type: str,  # "team", "player", "game"
        entity_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        sources: Optional[List[DataSource]] = None,
        categories: Optional[List[SentimentCategory]] = None
    ) -> Dict[str, Any]:
        """Get aggregated sentiment metrics for entities"""
        
        # Generate cache key
        query_params = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "start_date": start_date,
            "end_date": end_date,
            "sources": sources,
            "categories": categories
        }
        query_hash = self._generate_query_hash(query_params)
        
        # Check cache first
        cache_service = await self.get_cache_service()
        cached_result = await cache_service.get_analytics_data(query_hash)
        if cached_result:
            return cached_result
        
        db = await self.get_database()
        collection = db.sentiment_analysis
        
        # Build aggregation pipeline
        pipeline = []
        
        # Match stage
        match_conditions = {}
        
        if entity_id:
            match_conditions[f"{entity_type}_id"] = entity_id
        
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter["$gte"] = start_date
            if end_date:
                date_filter["$lte"] = end_date
            match_conditions["timestamp"] = date_filter
        
        if sources:
            match_conditions["source"] = {"$in": sources}
        
        if categories:
            match_conditions["category"] = {"$in": categories}
        
        if match_conditions:
            pipeline.append({"$match": match_conditions})
        
        # Group by entity and calculate metrics
        group_stage = {
            "$group": {
                "_id": f"${entity_type}_id",
                "total_mentions": {"$sum": 1},
                "avg_sentiment": {"$avg": "$sentiment_score"},
                "avg_confidence": {"$avg": "$confidence"},
                "positive_count": {
                    "$sum": {"$cond": [{"$eq": ["$sentiment", "POSITIVE"]}, 1, 0]}
                },
                "negative_count": {
                    "$sum": {"$cond": [{"$eq": ["$sentiment", "NEGATIVE"]}, 1, 0]}
                },
                "neutral_count": {
                    "$sum": {"$cond": [{"$eq": ["$sentiment", "NEUTRAL"]}, 1, 0]}
                },
                "sentiment_scores": {"$push": "$sentiment_score"},
                "categories": {"$push": "$category"},
                "sources": {"$push": "$source"},
                "timestamps": {"$push": "$timestamp"}
            }
        }
        
        pipeline.append(group_stage)
        
        # Add calculated fields
        pipeline.append({
            "$addFields": {
                "sentiment_distribution": {
                    "positive": {"$divide": ["$positive_count", "$total_mentions"]},
                    "negative": {"$divide": ["$negative_count", "$total_mentions"]},
                    "neutral": {"$divide": ["$neutral_count", "$total_mentions"]}
                },
                "sentiment_volatility": {
                    "$stdDevPop": "$sentiment_scores"
                }
            }
        })
        
        # Execute aggregation
        cursor = collection.aggregate(pipeline)
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
                "sentiment_distribution": result["sentiment_distribution"],
                "sentiment_volatility": result.get("sentiment_volatility", 0),
                "category_breakdown": self._calculate_category_breakdown(result["categories"]),
                "source_breakdown": self._calculate_source_breakdown(result["sources"]),
                "time_range": {
                    "start": min(result["timestamps"]).isoformat(),
                    "end": max(result["timestamps"]).isoformat()
                }
            }
            processed_results.append(entity_metrics)
        
        # Sort by total mentions
        processed_results.sort(key=lambda x: x["total_mentions"], reverse=True)
        
        final_result = {
            "query_parameters": query_params,
            "total_entities": len(processed_results),
            "generated_at": datetime.utcnow().isoformat(),
            "metrics": processed_results
        }
        
        # Cache the result
        await cache_service.cache_analytics_data(query_hash, final_result)
        
        return final_result
    
    def _calculate_category_breakdown(self, categories: List[str]) -> Dict[str, float]:
        """Calculate category distribution"""
        if not categories:
            return {}
        
        total = len(categories)
        breakdown = {}
        
        for category in set(categories):
            count = categories.count(category)
            breakdown[category] = count / total
        
        return breakdown
    
    def _calculate_source_breakdown(self, sources: List[str]) -> Dict[str, float]:
        """Calculate source distribution"""
        if not sources:
            return {}
        
        total = len(sources)
        breakdown = {}
        
        for source in set(sources):
            count = sources.count(source)
            breakdown[source] = count / total
        
        return breakdown
    
    async def get_sentiment_trends(
        self,
        entity_type: str,
        entity_id: str,
        period: str = "24h",  # "1h", "24h", "7d", "30d"
        interval: str = "hour"  # "minute", "hour", "day"
    ) -> List[SentimentTrend]:
        """Get sentiment trends over time"""
        
        # Check cache first
        cache_service = await self.get_cache_service()
        cached_trends = await cache_service.get_sentiment_trends(
            entity_type, entity_id, period
        )
        if cached_trends:
            return [SentimentTrend(**trend) for trend in cached_trends]
        
        db = await self.get_database()
        collection = db.sentiment_analysis
        
        # Calculate time range
        end_time = datetime.utcnow()
        if period == "1h":
            start_time = end_time - timedelta(hours=1)
        elif period == "24h":
            start_time = end_time - timedelta(hours=24)
        elif period == "7d":
            start_time = end_time - timedelta(days=7)
        elif period == "30d":
            start_time = end_time - timedelta(days=30)
        else:
            start_time = end_time - timedelta(hours=24)
        
        # Determine grouping interval
        if interval == "minute":
            date_format = "%Y-%m-%d %H:%M"
            group_interval = {"$dateToString": {"format": date_format, "date": "$timestamp"}}
        elif interval == "hour":
            date_format = "%Y-%m-%d %H"
            group_interval = {"$dateToString": {"format": date_format, "date": "$timestamp"}}
        else:  # day
            date_format = "%Y-%m-%d"
            group_interval = {"$dateToString": {"format": date_format, "date": "$timestamp"}}
        
        # Build aggregation pipeline
        pipeline = [
            {
                "$match": {
                    f"{entity_type}_id": entity_id,
                    "timestamp": {"$gte": start_time, "$lte": end_time}
                }
            },
            {
                "$group": {
                    "_id": group_interval,
                    "avg_sentiment": {"$avg": "$sentiment_score"},
                    "volume": {"$sum": 1},
                    "categories": {"$push": "$category"},
                    "sources": {"$push": "$source"}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]
        
        cursor = collection.aggregate(pipeline)
        results = await cursor.to_list(length=None)
        
        # Convert to SentimentTrend objects
        trends = []
        for result in results:
            trend = SentimentTrend(
                timestamp=datetime.strptime(result["_id"], date_format),
                sentiment_score=result["avg_sentiment"],
                volume=result["volume"],
                category_breakdown=self._calculate_category_breakdown(result["categories"]),
                source_breakdown=self._calculate_source_breakdown(result["sources"])
            )
            trends.append(trend)
        
        # Cache the results
        trends_dict = [trend.dict() for trend in trends]
        await cache_service.cache_sentiment_trends(
            entity_type, entity_id, trends_dict, period
        )
        
        return trends
    
    async def get_sentiment_leaderboards(
        self,
        leaderboard_type: str,  # "most_positive", "most_negative", "most_volatile", "most_mentioned"
        entity_type: str = "team",  # "team", "player"
        limit: int = 10,
        time_period: str = "24h"
    ) -> List[Dict[str, Any]]:
        """Get sentiment leaderboards"""
        
        # Check cache first
        cache_key = f"{leaderboard_type}_{entity_type}_{time_period}_{limit}"
        cache_service = await self.get_cache_service()
        cached_leaderboard = await cache_service.get_leaderboard(cache_key)
        if cached_leaderboard:
            return cached_leaderboard
        
        # Get aggregated metrics for all entities
        end_time = datetime.utcnow()
        if time_period == "1h":
            start_time = end_time - timedelta(hours=1)
        elif time_period == "24h":
            start_time = end_time - timedelta(hours=24)
        elif time_period == "7d":
            start_time = end_time - timedelta(days=7)
        else:
            start_time = end_time - timedelta(hours=24)
        
        metrics = await self.get_aggregated_sentiment_metrics(
            entity_type=entity_type,
            start_date=start_time,
            end_date=end_time
        )
        
        # Sort based on leaderboard type
        if leaderboard_type == "most_positive":
            sorted_metrics = sorted(
                metrics["metrics"], 
                key=lambda x: x["average_sentiment"], 
                reverse=True
            )
        elif leaderboard_type == "most_negative":
            sorted_metrics = sorted(
                metrics["metrics"], 
                key=lambda x: x["average_sentiment"]
            )
        elif leaderboard_type == "most_volatile":
            sorted_metrics = sorted(
                metrics["metrics"], 
                key=lambda x: x["sentiment_volatility"], 
                reverse=True
            )
        elif leaderboard_type == "most_mentioned":
            sorted_metrics = sorted(
                metrics["metrics"], 
                key=lambda x: x["total_mentions"], 
                reverse=True
            )
        else:
            sorted_metrics = metrics["metrics"]
        
        # Limit results
        leaderboard = sorted_metrics[:limit]
        
        # Add ranking
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1
        
        # Cache the results
        await cache_service.cache_leaderboard(cache_key, leaderboard)
        
        return leaderboard
    
    async def get_historical_comparison(
        self,
        entity_type: str,
        entity_id: str,
        comparison_periods: List[str] = ["7d", "30d", "90d"]
    ) -> Dict[str, Any]:
        """Get historical sentiment comparison"""
        
        current_time = datetime.utcnow()
        comparisons = {}
        
        for period in comparison_periods:
            if period == "7d":
                start_time = current_time - timedelta(days=7)
            elif period == "30d":
                start_time = current_time - timedelta(days=30)
            elif period == "90d":
                start_time = current_time - timedelta(days=90)
            else:
                continue
            
            metrics = await self.get_aggregated_sentiment_metrics(
                entity_type=entity_type,
                entity_id=entity_id,
                start_date=start_time,
                end_date=current_time
            )
            
            if metrics["metrics"]:
                entity_metrics = metrics["metrics"][0]
                comparisons[period] = {
                    "average_sentiment": entity_metrics["average_sentiment"],
                    "total_mentions": entity_metrics["total_mentions"],
                    "sentiment_distribution": entity_metrics["sentiment_distribution"],
                    "sentiment_volatility": entity_metrics["sentiment_volatility"]
                }
        
        # Calculate period-over-period changes
        changes = {}
        periods_list = list(comparisons.keys())
        
        for i in range(len(periods_list) - 1):
            current_period = periods_list[i]
            previous_period = periods_list[i + 1]
            
            if current_period in comparisons and previous_period in comparisons:
                current_sentiment = comparisons[current_period]["average_sentiment"]
                previous_sentiment = comparisons[previous_period]["average_sentiment"]
                
                change = current_sentiment - previous_sentiment
                percent_change = (change / abs(previous_sentiment)) * 100 if previous_sentiment != 0 else 0
                
                changes[f"{current_period}_vs_{previous_period}"] = {
                    "absolute_change": change,
                    "percent_change": percent_change,
                    "direction": "up" if change > 0 else "down" if change < 0 else "stable"
                }
        
        return {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "comparisons": comparisons,
            "changes": changes,
            "generated_at": current_time.isoformat()
        }
    
    async def export_analytics_data(
        self,
        export_format: str,  # "csv", "json"
        data_type: str,  # "metrics", "trends", "leaderboard"
        **kwargs
    ) -> Union[str, bytes]:
        """Export analytics data in specified format"""
        
        # Get the data based on type
        if data_type == "metrics":
            data = await self.get_aggregated_sentiment_metrics(**kwargs)
            export_data = data["metrics"]
        elif data_type == "trends":
            trends = await self.get_sentiment_trends(**kwargs)
            export_data = [trend.dict() for trend in trends]
        elif data_type == "leaderboard":
            export_data = await self.get_sentiment_leaderboards(**kwargs)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        if export_format == "json":
            return json.dumps(export_data, indent=2, default=str)
        
        elif export_format == "csv":
            if not export_data:
                return ""
            
            # Flatten nested dictionaries for CSV
            flattened_data = []
            for item in export_data:
                flattened_item = self._flatten_dict(item)
                flattened_data.append(flattened_item)
            
            # Create CSV
            output = io.StringIO()
            if flattened_data:
                fieldnames = flattened_data[0].keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flattened_data)
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
        """Flatten nested dictionary for CSV export"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to comma-separated strings
                items.append((new_key, ", ".join(map(str, v))))
            else:
                items.append((new_key, v))
        return dict(items)
    
    async def get_sentiment_insights(
        self,
        entity_type: str,
        entity_id: str,
        analysis_period: str = "7d"
    ) -> Dict[str, Any]:
        """Generate advanced sentiment insights and recommendations"""
        
        # Get historical data
        historical_comparison = await self.get_historical_comparison(
            entity_type, entity_id, ["7d", "30d"]
        )
        
        # Get trends
        trends = await self.get_sentiment_trends(
            entity_type, entity_id, analysis_period, "hour"
        )
        
        # Get current metrics
        current_metrics = await self.get_aggregated_sentiment_metrics(
            entity_type=entity_type,
            entity_id=entity_id,
            start_date=datetime.utcnow() - timedelta(days=1)
        )
        
        insights = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "analysis_period": analysis_period,
            "key_findings": [],
            "sentiment_drivers": {},
            "anomalies": [],
            "predictions": {},
            "recommendations": [],
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Analyze trends for key findings
        if trends:
            recent_trend = trends[-5:]  # Last 5 data points
            if len(recent_trend) >= 2:
                trend_direction = "increasing" if recent_trend[-1].sentiment_score > recent_trend[0].sentiment_score else "decreasing"
                insights["key_findings"].append(f"Sentiment is {trend_direction} over the last few hours")
        
        # Analyze volatility
        if current_metrics["metrics"]:
            volatility = current_metrics["metrics"][0]["sentiment_volatility"]
            if volatility > 0.5:
                insights["key_findings"].append("High sentiment volatility detected")
                insights["recommendations"].append("Monitor for potential sentiment shifts")
        
        # Analyze historical changes
        if "changes" in historical_comparison:
            for change_key, change_data in historical_comparison["changes"].items():
                if abs(change_data["percent_change"]) > 20:
                    insights["key_findings"].append(
                        f"Significant sentiment change: {change_data['percent_change']:.1f}% {change_data['direction']}"
                    )
        
        return insights


# Global analytics service instance
analytics_service = AnalyticsService()


async def get_analytics_service() -> AnalyticsService:
    """Dependency to get analytics service"""
    return analytics_service