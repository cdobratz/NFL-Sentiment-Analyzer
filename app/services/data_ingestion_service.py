"""
Data Ingestion Service for NFL Sentiment Analyzer

This service handles real-time data collection from multiple sources:
- Twitter/X API for social media sentiment
- ESPN API for NFL news and game data
- Betting APIs (DraftKings, MGM) for odds and lines
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import aiohttp
import json
from dataclasses import dataclass
from enum import Enum

from ..core.config import settings
from ..core.database import get_database
from ..models.nfl import Team, Player, Game, BettingLine

logger = logging.getLogger(__name__)


class DataSource(str, Enum):
    TWITTER = "twitter"
    ESPN = "espn"
    DRAFTKINGS = "draftkings"
    MGM = "mgm"
    NEWS = "news"


@dataclass
class RawDataItem:
    """Raw data item from external sources"""

    source: DataSource
    data_type: str  # tweet, news, game, betting_line
    content: Dict[str, Any]
    timestamp: datetime
    metadata: Dict[str, Any]


class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    async def acquire(self):
        """Wait if necessary to respect rate limits"""
        now = datetime.now()
        # Remove old requests outside the time window
        self.requests = [
            req_time
            for req_time in self.requests
            if (now - req_time).seconds < self.time_window
        ]

        if len(self.requests) >= self.max_requests:
            # Wait until we can make another request
            oldest_request = min(self.requests)
            wait_time = self.time_window - (now - oldest_request).seconds
            if wait_time > 0:
                await asyncio.sleep(wait_time)

        self.requests.append(now)


class DataIngestionService:
    """Main data ingestion service for collecting real-time NFL data"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiters = {
            DataSource.TWITTER: RateLimiter(300, 900),  # 300 requests per 15 minutes
            DataSource.ESPN: RateLimiter(100, 3600),  # 100 requests per hour
            DataSource.DRAFTKINGS: RateLimiter(60, 60),  # 60 requests per minute
            DataSource.MGM: RateLimiter(60, 60),  # 60 requests per minute
        }
        self.is_running = False

    async def start(self):
        """Initialize the data ingestion service"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        self.is_running = True
        logger.info("Data ingestion service started")

    async def stop(self):
        """Stop the data ingestion service"""
        self.is_running = False
        if self.session:
            await self.session.close()
        logger.info("Data ingestion service stopped")

    async def collect_twitter_data(
        self, keywords: List[str], max_results: int = 100
    ) -> List[RawDataItem]:
        """
        Collect NFL-related tweets using Twitter API v2

        Args:
            keywords: List of keywords to search for
            max_results: Maximum number of tweets to collect

        Returns:
            List of raw data items from Twitter
        """
        if not settings.twitter_bearer_token:
            logger.warning("Twitter Bearer Token not configured")
            return []

        await self.rate_limiters[DataSource.TWITTER].acquire()

        # Build search query
        query = " OR ".join([f'"{keyword}"' for keyword in keywords])
        query += " -is:retweet lang:en"  # Exclude retweets, English only

        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {
            "Authorization": f"Bearer {settings.twitter_bearer_token}",
            "Content-Type": "application/json",
        }

        params = {
            "query": query,
            "max_results": min(max_results, 100),  # API limit
            "tweet.fields": "created_at,author_id,public_metrics,context_annotations,lang",
            "user.fields": "name,username,verified,public_metrics",
            "expansions": "author_id",
        }

        try:
            async with self.session.get(
                url, headers=headers, params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_twitter_data(data)
                else:
                    logger.error(
                        f"Twitter API error: {response.status} - {await response.text()}"
                    )
                    return []
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {e}")
            return []

    def _process_twitter_data(self, data: Dict) -> List[RawDataItem]:
        """Process raw Twitter API response into RawDataItem objects"""
        items = []
        tweets = data.get("data", [])
        users = {user["id"]: user for user in data.get("includes", {}).get("users", [])}

        for tweet in tweets:
            user = users.get(tweet["author_id"], {})

            raw_item = RawDataItem(
                source=DataSource.TWITTER,
                data_type="tweet",
                content={
                    "id": tweet["id"],
                    "text": tweet["text"],
                    "created_at": tweet["created_at"],
                    "author": {
                        "id": tweet["author_id"],
                        "name": user.get("name", ""),
                        "username": user.get("username", ""),
                        "verified": user.get("verified", False),
                        "followers_count": user.get("public_metrics", {}).get(
                            "followers_count", 0
                        ),
                    },
                    "metrics": tweet.get("public_metrics", {}),
                    "context_annotations": tweet.get("context_annotations", []),
                },
                timestamp=datetime.now(),
                metadata={"lang": tweet.get("lang", "en"), "source": "twitter_api_v2"},
            )
            items.append(raw_item)

        logger.info(f"Processed {len(items)} tweets from Twitter API")
        return items

    async def fetch_espn_data(
        self, game_ids: Optional[List[str]] = None
    ) -> List[RawDataItem]:
        """
        Fetch NFL data from ESPN API

        Args:
            game_ids: Optional list of specific game IDs to fetch

        Returns:
            List of raw data items from ESPN
        """
        await self.rate_limiters[DataSource.ESPN].acquire()

        items = []

        # Fetch current week's games if no specific games requested
        if not game_ids:
            items.extend(await self._fetch_espn_scoreboard())

        # Fetch news
        items.extend(await self._fetch_espn_news())

        return items

    async def _fetch_espn_scoreboard(self) -> List[RawDataItem]:
        """Fetch current NFL scoreboard from ESPN"""
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_espn_scoreboard(data)
                else:
                    logger.error(f"ESPN Scoreboard API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching ESPN scoreboard: {e}")
            return []

    def _process_espn_scoreboard(self, data: Dict) -> List[RawDataItem]:
        """Process ESPN scoreboard data"""
        items = []
        events = data.get("events", [])

        for event in events:
            raw_item = RawDataItem(
                source=DataSource.ESPN,
                data_type="game",
                content={
                    "id": event["id"],
                    "name": event["name"],
                    "short_name": event["shortName"],
                    "date": event["date"],
                    "status": event["status"],
                    "competitions": event["competitions"],
                    "season": event.get("season", {}),
                    "week": event.get("week", {}),
                },
                timestamp=datetime.now(),
                metadata={
                    "source": "espn_scoreboard",
                    "season_type": event.get("season", {}).get("type", ""),
                    "week_number": event.get("week", {}).get("number", 0),
                },
            )
            items.append(raw_item)

        logger.info(f"Processed {len(items)} games from ESPN scoreboard")
        return items

    async def _fetch_espn_news(self) -> List[RawDataItem]:
        """Fetch NFL news from ESPN"""
        url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._process_espn_news(data)
                else:
                    logger.error(f"ESPN News API error: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching ESPN news: {e}")
            return []

    def _process_espn_news(self, data: Dict) -> List[RawDataItem]:
        """Process ESPN news data"""
        items = []
        articles = data.get("articles", [])

        for article in articles:
            raw_item = RawDataItem(
                source=DataSource.ESPN,
                data_type="news",
                content={
                    "id": article.get("id"),
                    "headline": article.get("headline"),
                    "description": article.get("description"),
                    "story": article.get("story"),
                    "published": article.get("published"),
                    "categories": article.get("categories", []),
                    "links": article.get("links", {}),
                    "images": article.get("images", []),
                },
                timestamp=datetime.now(),
                metadata={"source": "espn_news", "type": article.get("type", "story")},
            )
            items.append(raw_item)

        logger.info(f"Processed {len(items)} news articles from ESPN")
        return items

    async def get_betting_lines(
        self, sportsbooks: List[str] = None
    ) -> List[RawDataItem]:
        """
        Fetch betting lines from multiple sportsbooks

        Args:
            sportsbooks: List of sportsbooks to fetch from (draftkings, mgm)

        Returns:
            List of raw betting line data
        """
        if not sportsbooks:
            sportsbooks = ["draftkings", "mgm"]

        items = []

        for sportsbook in sportsbooks:
            if sportsbook.lower() == "draftkings":
                items.extend(await self._fetch_draftkings_lines())
            elif sportsbook.lower() == "mgm":
                items.extend(await self._fetch_mgm_lines())

        return items

    async def _fetch_draftkings_lines(self) -> List[RawDataItem]:
        """Fetch betting lines from DraftKings (mock implementation)"""
        # Note: This is a mock implementation as DraftKings doesn't have a public API
        # In a real implementation, you would need to use their official API or
        # a third-party odds provider like The Odds API

        await self.rate_limiters[DataSource.DRAFTKINGS].acquire()

        logger.info("Mock DraftKings betting lines fetch")

        # Mock data structure
        mock_lines = [
            {
                "game_id": "mock_game_1",
                "home_team": "Kansas City Chiefs",
                "away_team": "Buffalo Bills",
                "spread": -3.5,
                "over_under": 47.5,
                "home_moneyline": -180,
                "away_moneyline": +150,
                "last_updated": datetime.now().isoformat(),
            }
        ]

        items = []
        for line in mock_lines:
            raw_item = RawDataItem(
                source=DataSource.DRAFTKINGS,
                data_type="betting_line",
                content=line,
                timestamp=datetime.now(),
                metadata={"source": "draftkings_api", "sportsbook": "DraftKings"},
            )
            items.append(raw_item)

        return items

    async def _fetch_mgm_lines(self) -> List[RawDataItem]:
        """Fetch betting lines from MGM (mock implementation)"""
        # Note: This is a mock implementation as MGM doesn't have a public API

        await self.rate_limiters[DataSource.MGM].acquire()

        logger.info("Mock MGM betting lines fetch")

        # Mock data structure
        mock_lines = [
            {
                "game_id": "mock_game_1",
                "home_team": "Kansas City Chiefs",
                "away_team": "Buffalo Bills",
                "spread": -3.0,
                "over_under": 48.0,
                "home_moneyline": -175,
                "away_moneyline": +145,
                "last_updated": datetime.now().isoformat(),
            }
        ]

        items = []
        for line in mock_lines:
            raw_item = RawDataItem(
                source=DataSource.MGM,
                data_type="betting_line",
                content=line,
                timestamp=datetime.now(),
                metadata={"source": "mgm_api", "sportsbook": "MGM"},
            )
            items.append(raw_item)

        return items

    async def process_raw_data(self, raw_items: List[RawDataItem]) -> Dict[str, int]:
        """
        Process raw data items and store them in the database

        Args:
            raw_items: List of raw data items to process

        Returns:
            Dictionary with processing statistics
        """
        stats = {
            "processed": 0,
            "duplicates": 0,
            "errors": 0,
            "tweets": 0,
            "news": 0,
            "games": 0,
            "betting_lines": 0,
        }

        db = await get_database()

        for item in raw_items:
            try:
                # Check for duplicates
                if await self._is_duplicate(db, item):
                    stats["duplicates"] += 1
                    continue

                # Process based on data type
                if item.data_type == "tweet":
                    await self._store_tweet(db, item)
                    stats["tweets"] += 1
                elif item.data_type == "news":
                    await self._store_news(db, item)
                    stats["news"] += 1
                elif item.data_type == "game":
                    await self._store_game_data(db, item)
                    stats["games"] += 1
                elif item.data_type == "betting_line":
                    await self._store_betting_line(db, item)
                    stats["betting_lines"] += 1

                stats["processed"] += 1

            except Exception as e:
                logger.error(f"Error processing raw data item: {e}")
                stats["errors"] += 1

        logger.info(f"Data processing complete: {stats}")
        return stats

    async def _is_duplicate(self, db, item: RawDataItem) -> bool:
        """Check if data item already exists in database"""
        collection_name = f"raw_{item.data_type}s"

        # Create a unique identifier based on source and content
        if item.data_type == "tweet":
            unique_id = item.content.get("id")
        elif item.data_type == "news":
            unique_id = item.content.get("id")
        elif item.data_type == "game":
            unique_id = item.content.get("id")
        elif item.data_type == "betting_line":
            unique_id = f"{item.content.get('game_id')}_{item.source}"
        else:
            return False

        if not unique_id:
            return False

        existing = await db[collection_name].find_one({"unique_id": unique_id})
        return existing is not None

    async def _store_tweet(self, db, item: RawDataItem):
        """Store tweet data in database"""
        document = {
            "unique_id": item.content["id"],
            "source": item.source,
            "text": item.content["text"],
            "created_at": item.content["created_at"],
            "author": item.content["author"],
            "metrics": item.content["metrics"],
            "context_annotations": item.content["context_annotations"],
            "processed_at": item.timestamp,
            "metadata": item.metadata,
        }

        await db.raw_tweets.insert_one(document)

    async def _store_news(self, db, item: RawDataItem):
        """Store news article data in database"""
        document = {
            "unique_id": item.content.get("id"),
            "source": item.source,
            "headline": item.content.get("headline"),
            "description": item.content.get("description"),
            "story": item.content.get("story"),
            "published": item.content.get("published"),
            "categories": item.content.get("categories", []),
            "processed_at": item.timestamp,
            "metadata": item.metadata,
        }

        await db.raw_news.insert_one(document)

    async def _store_game_data(self, db, item: RawDataItem):
        """Store game data in database"""
        document = {
            "unique_id": item.content["id"],
            "source": item.source,
            "name": item.content["name"],
            "date": item.content["date"],
            "status": item.content["status"],
            "competitions": item.content["competitions"],
            "season": item.content.get("season", {}),
            "week": item.content.get("week", {}),
            "processed_at": item.timestamp,
            "metadata": item.metadata,
        }

        await db.raw_games.insert_one(document)

    async def _store_betting_line(self, db, item: RawDataItem):
        """Store betting line data in database"""
        document = {
            "unique_id": f"{item.content.get('game_id')}_{item.source}",
            "source": item.source,
            "sportsbook": item.metadata.get("sportsbook"),
            "game_id": item.content.get("game_id"),
            "home_team": item.content.get("home_team"),
            "away_team": item.content.get("away_team"),
            "spread": item.content.get("spread"),
            "over_under": item.content.get("over_under"),
            "home_moneyline": item.content.get("home_moneyline"),
            "away_moneyline": item.content.get("away_moneyline"),
            "last_updated": item.content.get("last_updated"),
            "processed_at": item.timestamp,
            "metadata": item.metadata,
        }

        await db.raw_betting_lines.insert_one(document)


# Global instance
data_ingestion_service = DataIngestionService()
