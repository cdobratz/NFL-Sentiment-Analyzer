from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class SentimentLabel(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"


class SentimentCategory(str, Enum):
    PERFORMANCE = "performance"  # Player/team performance related
    INJURY = "injury"  # Injury reports and concerns
    TRADE = "trade"  # Trade rumors and news
    COACHING = "coaching"  # Coaching decisions and changes
    BETTING = "betting"  # Betting and odds related
    GENERAL = "general"  # General team/player sentiment


class DataSource(str, Enum):
    DRAFTKINGS = "draftkings"
    MGM = "mgm"
    TWITTER = "twitter"
    ESPN = "espn"
    NEWS = "news"
    BETTING = "betting"
    USER_INPUT = "user_input"
    REDDIT = "reddit"
    FANTASY = "fantasy"


class NFLContext(BaseModel):
    """NFL-specific context for sentiment analysis"""

    team_mentions: List[str] = Field(default_factory=list)
    player_mentions: List[str] = Field(default_factory=list)
    position_mentions: List[str] = Field(default_factory=list)
    game_situation: Optional[str] = None  # "playoff", "regular_season", "preseason"
    week: Optional[int] = None
    season: Optional[int] = None
    injury_related: bool = False
    trade_related: bool = False
    performance_metrics: Dict[str, float] = Field(default_factory=dict)


class AnalysisContext(BaseModel):
    keywords: List[str] = Field(default_factory=list)
    entities: List[str] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    confidence_factors: Dict[str, float] = Field(default_factory=dict)
    nfl_context: Optional[NFLContext] = None
    sentiment_weights: Dict[str, float] = Field(default_factory=dict)  # Custom weights for NFL-specific terms


class SentimentAnalysisBase(BaseModel):
    text: str
    sentiment: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)  # Normalized sentiment score
    category: SentimentCategory = SentimentCategory.GENERAL
    context: AnalysisContext = Field(default_factory=AnalysisContext)
    team_id: Optional[str] = None
    player_id: Optional[str] = None
    game_id: Optional[str] = None
    source: DataSource = DataSource.USER_INPUT
    language: str = "en"
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Additional metadata like retweet count, likes, etc.


class SentimentAnalysisCreate(SentimentAnalysisBase):
    pass


class SentimentAnalysisInDB(SentimentAnalysisBase):
    id: str = Field(alias="_id")
    timestamp: datetime
    processed_at: datetime
    model_version: str = "1.0"
    processing_time_ms: Optional[float] = None

    class Config:
        populate_by_name = True


class SentimentAnalysisResponse(SentimentAnalysisBase):
    id: str
    timestamp: datetime
    model_version: str
    processing_time_ms: Optional[float] = None


class SentimentResult(BaseModel):
    """Enhanced result model for sentiment analysis with detailed breakdown"""

    text: str
    sentiment: SentimentLabel
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    category: SentimentCategory
    context: AnalysisContext
    team_id: Optional[str] = None
    player_id: Optional[str] = None
    game_id: Optional[str] = None
    source: DataSource
    timestamp: datetime
    model_version: str
    processing_time_ms: float

    # Detailed sentiment breakdown
    emotion_scores: Dict[str, float] = Field(default_factory=dict)  # joy, anger, fear, etc.
    aspect_sentiments: Dict[str, float] = Field(default_factory=dict)  # performance, coaching, etc.
    keyword_contributions: Dict[str, float] = Field(default_factory=dict)  # How each keyword contributed to sentiment


class BatchSentimentRequest(BaseModel):
    texts: List[str] = Field(..., max_items=100)
    context: Optional[AnalysisContext] = None
    team_id: Optional[str] = None
    player_id: Optional[str] = None
    game_id: Optional[str] = None
    source: DataSource = DataSource.USER_INPUT
    include_detailed_analysis: bool = False


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResult]
    total_processed: int
    processing_time_ms: float
    model_version: str
    aggregated_sentiment: float
    sentiment_distribution: Dict[SentimentLabel, int]


class SentimentTrend(BaseModel):
    timestamp: datetime
    sentiment_score: float
    volume: int
    category_breakdown: Dict[SentimentCategory, float] = Field(default_factory=dict)
    source_breakdown: Dict[DataSource, int] = Field(default_factory=dict)


class AggregatedSentiment(BaseModel):
    """Base class for aggregated sentiment data"""

    current_sentiment: float = Field(..., ge=-1.0, le=1.0)
    sentiment_trend: List[SentimentTrend] = Field(default_factory=list)
    total_mentions: int = 0
    last_updated: datetime
    sentiment_by_category: Dict[SentimentCategory, float] = Field(default_factory=dict)
    sentiment_by_source: Dict[DataSource, float] = Field(default_factory=dict)
    confidence_score: float = Field(..., ge=0.0, le=1.0)


class TeamSentiment(AggregatedSentiment):
    team_id: str
    team_name: str
    team_abbreviation: str
    # Team-specific metrics
    performance_sentiment: float = 0.0
    coaching_sentiment: float = 0.0
    injury_sentiment: float = 0.0
    trade_sentiment: float = 0.0
    fan_engagement_score: float = 0.0


class PlayerSentiment(AggregatedSentiment):
    player_id: str
    player_name: str
    team_id: str
    position: str
    # Player-specific metrics
    performance_sentiment: float = 0.0
    injury_sentiment: float = 0.0
    trade_sentiment: float = 0.0
    fantasy_sentiment: float = 0.0
    contract_sentiment: float = 0.0


class GameSentiment(AggregatedSentiment):
    game_id: str
    home_team_id: str
    away_team_id: str
    game_date: datetime
    week: int
    season: int
    # Game-specific sentiment breakdown
    home_team_sentiment: float = 0.0
    away_team_sentiment: float = 0.0
    overall_sentiment: float = 0.0
    betting_sentiment: float = 0.0
    prediction_confidence: float = 0.0
    # Pre/during/post game sentiment
    pregame_sentiment: float = 0.0
    live_sentiment: float = 0.0
    postgame_sentiment: float = 0.0


class SentimentAggregationRequest(BaseModel):
    """Request model for sentiment aggregation queries"""

    entity_type: str  # "team", "player", "game"
    entity_id: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    sources: Optional[List[DataSource]] = None
    categories: Optional[List[SentimentCategory]] = None
    include_trends: bool = True
    trend_interval: str = "hour"  # "minute", "hour", "day"


class SentimentInsights(BaseModel):
    """Advanced insights derived from sentiment analysis"""

    entity_id: str
    entity_type: str
    key_themes: List[str] = Field(default_factory=list)
    sentiment_drivers: Dict[str, float] = Field(default_factory=dict)  # What's driving positive/negative sentiment
    anomalies: List[Dict[str, Any]] = Field(default_factory=list)  # Unusual sentiment patterns
    predictions: Dict[str, float] = Field(default_factory=dict)  # Predicted sentiment trends
    recommendations: List[str] = Field(default_factory=list)  # Actionable insights
    generated_at: datetime
