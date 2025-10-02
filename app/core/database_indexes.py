"""
Database indexing configuration for efficient sentiment analysis queries.
"""

from typing import List, Dict, Any
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from motor.motor_asyncio import AsyncIOMotorDatabase


class DatabaseIndexes:
    """Database indexing configuration for MongoDB collections"""

    @staticmethod
    def get_sentiment_analysis_indexes() -> List[IndexModel]:
        """Get indexes for sentiment_analysis collection"""
        return [
            # Primary queries
            IndexModel([("team_id", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("player_id", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("game_id", ASCENDING), ("timestamp", DESCENDING)]),
            # Source and category filtering
            IndexModel([("source", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("category", ASCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("sentiment", ASCENDING), ("timestamp", DESCENDING)]),
            # Compound indexes for complex queries
            IndexModel(
                [
                    ("team_id", ASCENDING),
                    ("category", ASCENDING),
                    ("timestamp", DESCENDING),
                ]
            ),
            IndexModel(
                [
                    ("player_id", ASCENDING),
                    ("source", ASCENDING),
                    ("timestamp", DESCENDING),
                ]
            ),
            IndexModel(
                [
                    ("game_id", ASCENDING),
                    ("sentiment", ASCENDING),
                    ("timestamp", DESCENDING),
                ]
            ),
            # Confidence and score filtering
            IndexModel([("confidence", DESCENDING), ("timestamp", DESCENDING)]),
            IndexModel([("sentiment_score", DESCENDING), ("timestamp", DESCENDING)]),
            # Text search index
            IndexModel([("text", TEXT)]),
            # Processing metadata
            IndexModel([("processed_at", DESCENDING)]),
            IndexModel([("model_version", ASCENDING)]),
            # Time-based partitioning helpers
            IndexModel([("timestamp", DESCENDING)]),  # Most recent first
            IndexModel([("timestamp", ASCENDING)]),  # Oldest first for cleanup
        ]

    @staticmethod
    def get_team_sentiment_indexes() -> List[IndexModel]:
        """Get indexes for aggregated team sentiment collection"""
        return [
            # Primary key
            IndexModel([("team_id", ASCENDING)], unique=True),
            # Sorting and filtering
            IndexModel([("current_sentiment", DESCENDING)]),
            IndexModel([("last_updated", DESCENDING)]),
            IndexModel([("total_mentions", DESCENDING)]),
            IndexModel([("confidence_score", DESCENDING)]),
            # Category-based queries
            IndexModel([("sentiment_by_category.performance", DESCENDING)]),
            IndexModel([("sentiment_by_category.injury", DESCENDING)]),
            IndexModel([("sentiment_by_category.trade", DESCENDING)]),
        ]

    @staticmethod
    def get_player_sentiment_indexes() -> List[IndexModel]:
        """Get indexes for aggregated player sentiment collection"""
        return [
            # Primary key
            IndexModel([("player_id", ASCENDING)], unique=True),
            # Team-based queries
            IndexModel([("team_id", ASCENDING), ("current_sentiment", DESCENDING)]),
            IndexModel([("position", ASCENDING), ("current_sentiment", DESCENDING)]),
            # Sorting and filtering
            IndexModel([("current_sentiment", DESCENDING)]),
            IndexModel([("last_updated", DESCENDING)]),
            IndexModel([("total_mentions", DESCENDING)]),
            IndexModel([("fantasy_sentiment", DESCENDING)]),
            IndexModel([("performance_sentiment", DESCENDING)]),
        ]

    @staticmethod
    def get_game_sentiment_indexes() -> List[IndexModel]:
        """Get indexes for aggregated game sentiment collection"""
        return [
            # Primary key
            IndexModel([("game_id", ASCENDING)], unique=True),
            # Time-based queries
            IndexModel([("game_date", DESCENDING)]),
            IndexModel([("week", ASCENDING), ("season", ASCENDING)]),
            # Team-based queries
            IndexModel([("home_team_id", ASCENDING), ("game_date", DESCENDING)]),
            IndexModel([("away_team_id", ASCENDING), ("game_date", DESCENDING)]),
            # Sentiment-based queries
            IndexModel([("overall_sentiment", DESCENDING)]),
            IndexModel([("prediction_confidence", DESCENDING)]),
            IndexModel([("betting_sentiment", DESCENDING)]),
        ]

    @staticmethod
    def get_teams_indexes() -> List[IndexModel]:
        """Get indexes for teams collection"""
        return [
            # Primary key
            IndexModel([("_id", ASCENDING)], unique=True),
            # Common queries
            IndexModel([("abbreviation", ASCENDING)], unique=True),
            IndexModel([("name", ASCENDING)]),
            IndexModel([("conference", ASCENDING), ("division", ASCENDING)]),
            # Text search for aliases and keywords
            IndexModel([("aliases", TEXT), ("keywords", TEXT)]),
        ]

    @staticmethod
    def get_players_indexes() -> List[IndexModel]:
        """Get indexes for players collection"""
        return [
            # Primary key
            IndexModel([("_id", ASCENDING)], unique=True),
            # Team and position queries
            IndexModel([("team_id", ASCENDING), ("position", ASCENDING)]),
            IndexModel([("position", ASCENDING)]),
            IndexModel([("status", ASCENDING)]),
            # Fantasy relevance
            IndexModel([("fantasy_relevant", ASCENDING), ("position", ASCENDING)]),
            # Text search for names and aliases
            IndexModel([("name", TEXT), ("aliases", TEXT)]),
        ]

    @staticmethod
    def get_games_indexes() -> List[IndexModel]:
        """Get indexes for games collection"""
        return [
            # Primary key
            IndexModel([("_id", ASCENDING)], unique=True),
            # Time-based queries
            IndexModel([("game_date", DESCENDING)]),
            IndexModel([("season", ASCENDING), ("week", ASCENDING)]),
            # Team queries
            IndexModel([("home_team_id", ASCENDING), ("game_date", DESCENDING)]),
            IndexModel([("away_team_id", ASCENDING), ("game_date", DESCENDING)]),
            # Status and type
            IndexModel([("status", ASCENDING), ("game_date", DESCENDING)]),
            IndexModel([("game_type", ASCENDING), ("game_date", DESCENDING)]),
            # Special games
            IndexModel([("primetime", ASCENDING), ("game_date", DESCENDING)]),
            IndexModel([("rivalry_game", ASCENDING), ("game_date", DESCENDING)]),
        ]


async def create_all_indexes(db: AsyncIOMotorDatabase) -> Dict[str, List[str]]:
    """Create all indexes for the application"""
    results = {}

    # Sentiment analysis indexes
    collection = db.sentiment_analysis
    indexes = DatabaseIndexes.get_sentiment_analysis_indexes()
    result = await collection.create_indexes(indexes)
    results["sentiment_analysis"] = result

    # Team sentiment indexes
    collection = db.team_sentiment
    indexes = DatabaseIndexes.get_team_sentiment_indexes()
    result = await collection.create_indexes(indexes)
    results["team_sentiment"] = result

    # Player sentiment indexes
    collection = db.player_sentiment
    indexes = DatabaseIndexes.get_player_sentiment_indexes()
    result = await collection.create_indexes(indexes)
    results["player_sentiment"] = result

    # Game sentiment indexes
    collection = db.game_sentiment
    indexes = DatabaseIndexes.get_game_sentiment_indexes()
    result = await collection.create_indexes(indexes)
    results["game_sentiment"] = result

    # Teams indexes
    collection = db.teams
    indexes = DatabaseIndexes.get_teams_indexes()
    result = await collection.create_indexes(indexes)
    results["teams"] = result

    # Players indexes
    collection = db.players
    indexes = DatabaseIndexes.get_players_indexes()
    result = await collection.create_indexes(indexes)
    results["players"] = result

    # Games indexes
    collection = db.games
    indexes = DatabaseIndexes.get_games_indexes()
    result = await collection.create_indexes(indexes)
    results["games"] = result

    return results


async def drop_all_indexes(db: AsyncIOMotorDatabase) -> Dict[str, bool]:
    """Drop all custom indexes (keep only _id indexes)"""
    results = {}
    collections = [
        "sentiment_analysis",
        "team_sentiment",
        "player_sentiment",
        "game_sentiment",
        "teams",
        "players",
        "games",
    ]

    for collection_name in collections:
        try:
            collection = db[collection_name]
            await collection.drop_indexes()
            results[collection_name] = True
        except Exception as e:
            results[collection_name] = False
            print(f"Error dropping indexes for {collection_name}: {e}")

    return results
