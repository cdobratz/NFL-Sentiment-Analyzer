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
        """
        Initialize the rate limiter with limits and prepare internal request history.

        Parameters:
                max_requests (int): Maximum number of requests allowed within the rolling time window.
                time_window (int): Time window length in seconds used to count recent requests.

        Attributes:
                requests (List[float]): Internal list of request timestamps (initialized empty) used to track recent requests.
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []

    async def acquire(self):
        """
        Enforces the rate limit by delaying until a new request is allowed and records the request timestamp.

        If the number of recorded requests within the last time_window seconds has reached max_requests,
        this method sleeps until the oldest recorded request falls outside the time window. After any
        necessary wait, the current time is appended to the internal request history. The time_window is
        interpreted in seconds.
        """
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
        """
        Initialize the DataIngestionService internal state and per-source rate limiters.

        Sets up:
        - session: optional aiohttp.ClientSession, initialized to None.
        - rate_limiters: mapping from DataSource to RateLimiter instances with configured limits:
          - TWITTER: 300 requests per 900 seconds (15 minutes)
          - ESPN: 100 requests per 3600 seconds (1 hour)
          - DRAFTKINGS: 60 requests per 60 seconds (1 minute)
          - MGM: 60 requests per 60 seconds (1 minute)
        - is_running: boolean flag indicating whether the service is active, initialized to False.
        """
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limiters = {
            DataSource.TWITTER: RateLimiter(300, 900),  # 300 requests per 15 minutes
            DataSource.ESPN: RateLimiter(100, 3600),  # 100 requests per hour
            DataSource.DRAFTKINGS: RateLimiter(60, 60),  # 60 requests per minute
            DataSource.MGM: RateLimiter(60, 60),  # 60 requests per minute
        }
        self.is_running = False

    async def start(self):
        """
        Start the data ingestion service.

        If no HTTP session exists, create an aiohttp.ClientSession, mark the service as running, and record a startup log message.
        """
        if not self.session:
            self.session = aiohttp.ClientSession()
        self.is_running = True
        logger.info("Data ingestion service started")

    async def stop(self):
        """
        Stop the data ingestion service and release associated network resources.

        Sets the service running flag to False and closes the internal aiohttp ClientSession if one exists.
        """
        self.is_running = False
        if self.session:
            await self.session.close()
        logger.info("Data ingestion service stopped")

    async def collect_twitter_data(
        self, keywords: List[str], max_results: int = 100
    ) -> List[RawDataItem]:
        """
        Collect recent NFL-related tweets matching the provided keywords.

        Parameters:
            keywords (List[str]): Keywords or phrases used to build the Twitter search query.
            max_results (int): Maximum number of tweets to request (capped at 100 by the Twitter API).

        Returns:
            List[RawDataItem]: RawDataItem objects representing tweets that matched the query.
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
        """
        Convert a Twitter API v2 response into a list of RawDataItem objects representing tweets.

        Parameters:
            data (Dict): Parsed JSON response from Twitter API v2. Expected to contain a "data" list of tweet objects and an "includes" object with a "users" list for author enrichment.

        Returns:
            List[RawDataItem]: A list of normalized tweet items; each item contains tweet id, text, created_at, author details (id, name, username, verified, followers_count), public metrics, context annotations, a processing timestamp, and metadata (language and source).
        """
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
        Retrieve ESPN NFL data, including scoreboard game data and news articles.

        Parameters:
            game_ids (Optional[List[str]]): If provided, restrict scoreboard retrieval to these ESPN game IDs; if omitted, fetch the current week's scoreboard.

        Returns:
            List[RawDataItem]: Aggregated list of RawDataItem objects representing ESPN games and news.
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
        """
        Fetch the current NFL scoreboard data from ESPN and convert it into RawDataItem objects.

        Returns:
            List[RawDataItem]: A list of processed scoreboard items; returns an empty list if the request fails or an error occurs.
        """
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
        """
        Convert an ESPN scoreboard JSON response into a list of normalized game RawDataItem objects.

        Parameters:
            data (Dict): Parsed ESPN scoreboard payload expected to contain an "events" list where each event includes keys used for game construction such as "id", "name", "shortName", "date", "status", "competitions", and optional "season" and "week" objects.

        Returns:
            List[RawDataItem]: A list of RawDataItem instances with data_type "game", populated content (id, name, short_name, date, status, competitions, season, week) and metadata (source, season_type, week_number) for each event in the scoreboard.
        """
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
        """
        Fetch current NFL news articles from ESPN and normalize them for ingestion.

        Returns:
            List[RawDataItem]: A list of normalized news items parsed from the ESPN response. Returns an empty list if the request fails or the API responds with a non-200 status.
        """
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
        """
        Convert an ESPN news API response into a list of standardized RawDataItem objects.

        Parameters:
            data (Dict): ESPN news API response dictionary expected to contain an "articles" key
                whose value is a list of article dictionaries. Each article dictionary may include
                keys: "id", "headline", "description", "story", "published", "categories",
                "links", "images", and "type".

        Returns:
            List[RawDataItem]: A list of RawDataItem objects with source set to DataSource.ESPN,
                data_type "news", content populated from each article (fields: id, headline,
                description, story, published, categories, links, images), timestamp set to
                the current time, and metadata containing "source": "espn_news" and "type".
        """
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
        Retrieve betting lines from the specified sportsbooks.

        Parameters:
            sportsbooks (Optional[List[str]]): Optional list of sportsbook identifiers to fetch; supported values are "draftkings" and "mgm". If omitted, both DraftKings and MGM are fetched.

        Returns:
            List[RawDataItem]: RawDataItem objects representing betting lines retrieved from the requested sportsbooks.
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
        """
        Return mock betting lines attributed to DraftKings.

        The returned items are normalized RawDataItem objects with data_type "betting_line" and metadata indicating DraftKings as the sportsbook. This is a mock implementation used when a public DraftKings API is not available.

        Returns:
            List[RawDataItem]: A list of betting line items containing game identifiers, teams, odds, and last_updated timestamps.
        """
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
        """
        Return a list of betting-line RawDataItem objects representing MGM sportsbook lines (mock data).

        Each returned item has data_type "betting_line" and content containing keys such as `game_id`, `home_team`, `away_team`, `spread`, `over_under`, `home_moneyline`, `away_moneyline`, and `last_updated`. Metadata includes the originating sportsbook.

        Returns:
            List[RawDataItem]: Mocked betting-line items from MGM.
        """
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
        Process a list of RawDataItem objects by storing them into the database and collecting processing statistics.

        Parameters:
            raw_items (List[RawDataItem]): Items to deduplicate, route to the appropriate storage handlers, and persist.

        Returns:
            stats (Dict[str, int]): Counts of processing outcomes with keys:
                - "processed": total items successfully handled
                - "duplicates": items skipped because they already exist
                - "errors": items that failed processing
                - "tweets": number of tweet items stored
                - "news": number of news items stored
                - "games": number of game items stored
                - "betting_lines": number of betting line items stored
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
        """
        Determine whether a RawDataItem with the same unique identifier already exists in the database.

        Parameters:
            item (RawDataItem): The raw data item whose unique identifier (derived from its type and content) will be used to look up existing records.

        Returns:
            True if a matching item exists in the corresponding collection, False otherwise.
        """
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
        """
        Insert a tweet RawDataItem into the database's raw_tweets collection.

        Constructs a document from the item's content (uses content["id"] as `unique_id`, and includes text, created_at, author, metrics, context_annotations), attaches processed_at from the item's timestamp and the item's metadata, and inserts the document into db.raw_tweets.

        Parameters:
            item (RawDataItem): Raw tweet item whose content must include keys `id`, `text`, `created_at`, `author`, `metrics`, and `context_annotations`.
        """
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
        """
        Persist the given news RawDataItem into the database's raw_news collection.

        Parameters:
            db: Database handle exposing a `raw_news` collection into which the document will be inserted.
            item (RawDataItem): News item whose content will be stored. The item's content `id` is used as the document `unique_id` and `timestamp` is recorded as `processed_at`.
        """
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
        """
        Store a single game's standardized data into the raw_games collection in the database.

        Parameters:
            db: Database handle exposing a `raw_games` collection with an `insert_one` coroutine.
            item (RawDataItem): RawDataItem whose `content` contains the game's fields `id`, `name`, `date`, `status`, and `competitions`; optional `season` and `week` may be present.
        """
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
        """
        Create and insert a betting-line document into the raw_betting_lines collection.

        Parameters:
            item (RawDataItem): Raw betting-line item whose content is used to build the stored document. Expected content keys include `game_id`, `home_team`, `away_team`, `spread`, `over_under`, `home_moneyline`, `away_moneyline`, and `last_updated`. The item's `metadata` should include `sportsbook`. The stored document will include a `unique_id` formed by concatenating the `game_id` and the item's source, plus `processed_at` set to the item's timestamp.

        """
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
