"""
Hopsworks feature store integration service for ML features management.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import json

try:
    import hopsworks
    import hsfs
    HOPSWORKS_AVAILABLE = True
except ImportError:
    HOPSWORKS_AVAILABLE = False
    hopsworks = None
    hsfs = None

from ...models.mlops import FeatureStore
from ...models.sentiment import SentimentResult, TeamSentiment, PlayerSentiment
from ...core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class HopsworksService:
    """Service for integrating with Hopsworks feature store"""
    
    def __init__(self):
        self.project = None
        self.feature_store = None
        self.feature_groups = {}
        self.feature_views = {}
        
        # Configuration
        self.project_name = getattr(settings, 'HOPSWORKS_PROJECT', 'nfl-sentiment-analyzer')
        self.api_key = getattr(settings, 'HOPSWORKS_API_KEY', None)
        self.host = getattr(settings, 'HOPSWORKS_HOST', 'https://c.app.hopsworks.ai')
        
        if not HOPSWORKS_AVAILABLE:
            logger.warning("Hopsworks not available. Install with: pip install hopsworks")
            return
        
        # Initialize connection if API key is provided
        if self.api_key:
            try:
                self._connect()
            except Exception as e:
                logger.warning(f"Failed to connect to Hopsworks: {e}")
    
    def _connect(self):
        """Connect to Hopsworks"""
        if not HOPSWORKS_AVAILABLE:
            return
        
        try:
            self.project = hopsworks.login(
                api_key_value=self.api_key,
                project=self.project_name,
                host=self.host
            )
            
            self.feature_store = self.project.get_feature_store()
            logger.info(f"Connected to Hopsworks project: {self.project_name}")
            
        except Exception as e:
            logger.error(f"Error connecting to Hopsworks: {e}")
            raise
    
    async def create_sentiment_feature_group(
        self,
        name: str = "sentiment_features",
        version: int = 1,
        description: str = "NFL sentiment analysis features"
    ) -> Optional[Any]:
        """
        Create a feature group for sentiment analysis features
        
        Args:
            name: Name of the feature group
            version: Version of the feature group
            description: Description of the feature group
            
        Returns:
            Feature group object
        """
        if not HOPSWORKS_AVAILABLE or not self.feature_store:
            logger.warning("Hopsworks not available, returning None")
            return None
        
        try:
            # Define feature schema for sentiment analysis
            features = [
                hsfs.feature.Feature("entity_id", "string", description="Team, player, or game ID"),
                hsfs.feature.Feature("entity_type", "string", description="Type of entity (team/player/game)"),
                hsfs.feature.Feature("sentiment_score", "double", description="Sentiment score (-1 to 1)"),
                hsfs.feature.Feature("confidence", "double", description="Confidence score (0 to 1)"),
                hsfs.feature.Feature("positive_mentions", "int", description="Number of positive mentions"),
                hsfs.feature.Feature("negative_mentions", "int", description="Number of negative mentions"),
                hsfs.feature.Feature("neutral_mentions", "int", description="Number of neutral mentions"),
                hsfs.feature.Feature("total_mentions", "int", description="Total number of mentions"),
                hsfs.feature.Feature("avg_sentiment_24h", "double", description="Average sentiment in last 24h"),
                hsfs.feature.Feature("sentiment_trend_7d", "double", description="Sentiment trend over 7 days"),
                hsfs.feature.Feature("injury_sentiment", "double", description="Injury-related sentiment"),
                hsfs.feature.Feature("performance_sentiment", "double", description="Performance-related sentiment"),
                hsfs.feature.Feature("trade_sentiment", "double", description="Trade-related sentiment"),
                hsfs.feature.Feature("betting_sentiment", "double", description="Betting-related sentiment"),
                hsfs.feature.Feature("social_volume", "int", description="Social media volume"),
                hsfs.feature.Feature("news_volume", "int", description="News article volume"),
                hsfs.feature.Feature("data_sources", "string", description="Comma-separated data sources"),
                hsfs.feature.Feature("last_updated", "timestamp", description="Last update timestamp"),
                hsfs.feature.Feature("event_time", "timestamp", description="Event timestamp")
            ]
            
            # Create feature group
            feature_group = self.feature_store.create_feature_group(
                name=name,
                version=version,
                description=description,
                primary_key=["entity_id", "entity_type"],
                event_time="event_time",
                online_enabled=True,
                features=features
            )
            
            # Cache the feature group
            self.feature_groups[f"{name}_v{version}"] = feature_group
            
            logger.info(f"Created sentiment feature group: {name} v{version}")
            return feature_group
            
        except Exception as e:
            logger.error(f"Error creating sentiment feature group: {e}")
            return None
    
    async def create_nfl_context_feature_group(
        self,
        name: str = "nfl_context_features",
        version: int = 1,
        description: str = "NFL contextual features"
    ) -> Optional[Any]:
        """
        Create a feature group for NFL contextual features
        
        Args:
            name: Name of the feature group
            version: Version of the feature group
            description: Description of the feature group
            
        Returns:
            Feature group object
        """
        if not HOPSWORKS_AVAILABLE or not self.feature_store:
            logger.warning("Hopsworks not available, returning None")
            return None
        
        try:
            # Define NFL context features
            features = [
                hsfs.feature.Feature("entity_id", "string", description="Team, player, or game ID"),
                hsfs.feature.Feature("entity_type", "string", description="Type of entity"),
                hsfs.feature.Feature("team_name", "string", description="Team name"),
                hsfs.feature.Feature("conference", "string", description="Team conference (AFC/NFC)"),
                hsfs.feature.Feature("division", "string", description="Team division"),
                hsfs.feature.Feature("current_season", "int", description="Current NFL season"),
                hsfs.feature.Feature("current_week", "int", description="Current NFL week"),
                hsfs.feature.Feature("wins", "int", description="Team wins"),
                hsfs.feature.Feature("losses", "int", description="Team losses"),
                hsfs.feature.Feature("win_percentage", "double", description="Win percentage"),
                hsfs.feature.Feature("points_for", "int", description="Points scored"),
                hsfs.feature.Feature("points_against", "int", description="Points allowed"),
                hsfs.feature.Feature("point_differential", "int", description="Point differential"),
                hsfs.feature.Feature("playoff_probability", "double", description="Playoff probability"),
                hsfs.feature.Feature("strength_of_schedule", "double", description="Strength of schedule"),
                hsfs.feature.Feature("injury_count", "int", description="Number of injured players"),
                hsfs.feature.Feature("key_player_injured", "boolean", description="Key player injured"),
                hsfs.feature.Feature("recent_trades", "int", description="Recent trades count"),
                hsfs.feature.Feature("coaching_changes", "boolean", description="Recent coaching changes"),
                hsfs.feature.Feature("home_field_advantage", "double", description="Home field advantage score"),
                hsfs.feature.Feature("event_time", "timestamp", description="Event timestamp")
            ]
            
            # Create feature group
            feature_group = self.feature_store.create_feature_group(
                name=name,
                version=version,
                description=description,
                primary_key=["entity_id", "entity_type"],
                event_time="event_time",
                online_enabled=True,
                features=features
            )
            
            # Cache the feature group
            self.feature_groups[f"{name}_v{version}"] = feature_group
            
            logger.info(f"Created NFL context feature group: {name} v{version}")
            return feature_group
            
        except Exception as e:
            logger.error(f"Error creating NFL context feature group: {e}")
            return None
    
    async def insert_sentiment_features(
        self,
        sentiment_data: List[Dict[str, Any]],
        feature_group_name: str = "sentiment_features",
        version: int = 1
    ) -> bool:
        """
        Insert sentiment features into the feature store
        
        Args:
            sentiment_data: List of sentiment feature dictionaries
            feature_group_name: Name of the feature group
            version: Version of the feature group
            
        Returns:
            Success status
        """
        if not HOPSWORKS_AVAILABLE or not self.feature_store:
            logger.warning("Hopsworks not available, skipping feature insertion")
            return False
        
        try:
            # Get or create feature group
            fg_key = f"{feature_group_name}_v{version}"
            if fg_key not in self.feature_groups:
                feature_group = await self.create_sentiment_feature_group(
                    feature_group_name, version
                )
                if not feature_group:
                    return False
            else:
                feature_group = self.feature_groups[fg_key]
            
            # Convert to DataFrame
            df = pd.DataFrame(sentiment_data)
            
            # Ensure required columns exist
            required_columns = [
                "entity_id", "entity_type", "sentiment_score", "confidence",
                "event_time"
            ]
            
            for col in required_columns:
                if col not in df.columns:
                    if col == "event_time":
                        df[col] = datetime.utcnow()
                    else:
                        df[col] = None
            
            # Insert features
            feature_group.insert(df)
            
            logger.info(f"Inserted {len(sentiment_data)} sentiment features")
            return True
            
        except Exception as e:
            logger.error(f"Error inserting sentiment features: {e}")
            return False
    
    async def get_sentiment_features(
        self,
        entity_id: str,
        entity_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        feature_group_name: str = "sentiment_features",
        version: int = 1
    ) -> Optional[pd.DataFrame]:
        """
        Get sentiment features for a specific entity
        
        Args:
            entity_id: ID of the entity
            entity_type: Type of entity (team/player/game)
            start_time: Start time for filtering
            end_time: End time for filtering
            feature_group_name: Name of the feature group
            version: Version of the feature group
            
        Returns:
            DataFrame with sentiment features
        """
        if not HOPSWORKS_AVAILABLE or not self.feature_store:
            logger.warning("Hopsworks not available, returning None")
            return None
        
        try:
            # Get feature group
            fg_key = f"{feature_group_name}_v{version}"
            if fg_key not in self.feature_groups:
                feature_group = self.feature_store.get_feature_group(
                    feature_group_name, version
                )
                self.feature_groups[fg_key] = feature_group
            else:
                feature_group = self.feature_groups[fg_key]
            
            # Build query
            query = feature_group.select_all().filter(
                (feature_group.entity_id == entity_id) &
                (feature_group.entity_type == entity_type)
            )
            
            # Add time filters if provided
            if start_time:
                query = query.filter(feature_group.event_time >= start_time)
            if end_time:
                query = query.filter(feature_group.event_time <= end_time)
            
            # Execute query
            df = query.read()
            
            logger.info(f"Retrieved {len(df)} sentiment features for {entity_type}:{entity_id}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting sentiment features: {e}")
            return None
    
    async def create_feature_view(
        self,
        name: str,
        version: int = 1,
        description: str = "NFL sentiment feature view",
        feature_groups: Optional[List[str]] = None
    ) -> Optional[Any]:
        """
        Create a feature view for ML training
        
        Args:
            name: Name of the feature view
            version: Version of the feature view
            description: Description of the feature view
            feature_groups: List of feature group names to include
            
        Returns:
            Feature view object
        """
        if not HOPSWORKS_AVAILABLE or not self.feature_store:
            logger.warning("Hopsworks not available, returning None")
            return None
        
        try:
            # Default feature groups
            if not feature_groups:
                feature_groups = ["sentiment_features", "nfl_context_features"]
            
            # Get feature groups
            query_objects = []
            for fg_name in feature_groups:
                try:
                    fg = self.feature_store.get_feature_group(fg_name, 1)
                    query_objects.append(fg.select_all())
                except Exception as e:
                    logger.warning(f"Could not get feature group {fg_name}: {e}")
            
            if not query_objects:
                logger.error("No valid feature groups found for feature view")
                return None
            
            # Join feature groups if multiple
            if len(query_objects) == 1:
                query = query_objects[0]
            else:
                query = query_objects[0]
                for q in query_objects[1:]:
                    query = query.join(q, on=["entity_id", "entity_type"])
            
            # Create feature view
            feature_view = self.feature_store.create_feature_view(
                name=name,
                version=version,
                description=description,
                query=query
            )
            
            # Cache the feature view
            self.feature_views[f"{name}_v{version}"] = feature_view
            
            logger.info(f"Created feature view: {name} v{version}")
            return feature_view
            
        except Exception as e:
            logger.error(f"Error creating feature view: {e}")
            return None
    
    async def get_training_data(
        self,
        feature_view_name: str,
        version: int = 1,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        training_dataset_version: Optional[int] = None
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Get training data from a feature view
        
        Args:
            feature_view_name: Name of the feature view
            version: Version of the feature view
            start_time: Start time for training data
            end_time: End time for training data
            training_dataset_version: Version of training dataset
            
        Returns:
            Tuple of (features, labels) DataFrames
        """
        if not HOPSWORKS_AVAILABLE or not self.feature_store:
            logger.warning("Hopsworks not available, returning None")
            return None
        
        try:
            # Get feature view
            fv_key = f"{feature_view_name}_v{version}"
            if fv_key not in self.feature_views:
                feature_view = self.feature_store.get_feature_view(
                    feature_view_name, version
                )
                self.feature_views[fv_key] = feature_view
            else:
                feature_view = self.feature_views[fv_key]
            
            # Create training dataset
            td_version = training_dataset_version or 1
            
            X_train, X_test, y_train, y_test = feature_view.train_test_split(
                test_size=0.2,
                random_state=42
            )
            
            logger.info(f"Retrieved training data: {len(X_train)} train, {len(X_test)} test samples")
            return (X_train, y_train), (X_test, y_test)
            
        except Exception as e:
            logger.error(f"Error getting training data: {e}")
            return None
    
    async def compute_feature_statistics(
        self,
        feature_group_name: str,
        version: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        Compute statistics for a feature group
        
        Args:
            feature_group_name: Name of the feature group
            version: Version of the feature group
            
        Returns:
            Feature statistics
        """
        if not HOPSWORKS_AVAILABLE or not self.feature_store:
            logger.warning("Hopsworks not available, returning None")
            return None
        
        try:
            # Get feature group
            feature_group = self.feature_store.get_feature_group(
                feature_group_name, version
            )
            
            # Compute statistics
            statistics = feature_group.compute_statistics()
            
            logger.info(f"Computed statistics for {feature_group_name}")
            return statistics
            
        except Exception as e:
            logger.error(f"Error computing feature statistics: {e}")
            return None
    
    async def update_sentiment_features_from_results(
        self,
        sentiment_results: List[SentimentResult]
    ) -> bool:
        """
        Update feature store with new sentiment analysis results
        
        Args:
            sentiment_results: List of sentiment analysis results
            
        Returns:
            Success status
        """
        try:
            feature_data = []
            
            for result in sentiment_results:
                # Determine entity info
                entity_id = result.team_id or result.player_id or result.game_id
                if not entity_id:
                    continue
                
                entity_type = "team" if result.team_id else ("player" if result.player_id else "game")
                
                # Create feature record
                feature_record = {
                    "entity_id": entity_id,
                    "entity_type": entity_type,
                    "sentiment_score": result.sentiment_score,
                    "confidence": result.confidence,
                    "positive_mentions": 1 if result.sentiment.value == "POSITIVE" else 0,
                    "negative_mentions": 1 if result.sentiment.value == "NEGATIVE" else 0,
                    "neutral_mentions": 1 if result.sentiment.value == "NEUTRAL" else 0,
                    "total_mentions": 1,
                    "injury_sentiment": result.aspect_sentiments.get("injury", 0.0),
                    "performance_sentiment": result.aspect_sentiments.get("performance", 0.0),
                    "trade_sentiment": result.aspect_sentiments.get("trade", 0.0),
                    "betting_sentiment": result.aspect_sentiments.get("betting", 0.0),
                    "social_volume": 1 if result.source.value in ["twitter", "reddit"] else 0,
                    "news_volume": 1 if result.source.value in ["espn", "news"] else 0,
                    "data_sources": result.source.value,
                    "last_updated": datetime.utcnow(),
                    "event_time": result.timestamp
                }
                
                feature_data.append(feature_record)
            
            # Insert features
            if feature_data:
                success = await self.insert_sentiment_features(feature_data)
                return success
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating sentiment features: {e}")
            return False
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of the Hopsworks service"""
        return {
            "hopsworks_available": HOPSWORKS_AVAILABLE,
            "connected": self.project is not None,
            "project_name": self.project_name,
            "feature_store_available": self.feature_store is not None,
            "cached_feature_groups": len(self.feature_groups),
            "cached_feature_views": len(self.feature_views)
        }