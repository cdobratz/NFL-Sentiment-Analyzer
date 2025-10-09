from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum


class Conference(str, Enum):
    AFC = "AFC"
    NFC = "NFC"


class Division(str, Enum):
    NORTH = "North"
    SOUTH = "South"
    EAST = "East"
    WEST = "West"


class GameStatus(str, Enum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class Team(BaseModel):
    id: str = Field(alias="_id")
    name: str
    abbreviation: str
    city: str
    conference: Conference
    division: Division
    logo_url: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    # Enhanced fields for sentiment analysis
    aliases: List[str] = Field(
        default_factory=list
    )  # Alternative names/nicknames for the team
    keywords: List[str] = Field(
        default_factory=list
    )  # Team-specific keywords for sentiment analysis
    established: Optional[int] = None
    stadium: Optional[str] = None
    head_coach: Optional[str] = None

    class Config:
        populate_by_name = True


class Player(BaseModel):
    id: str = Field(alias="_id")
    name: str
    team_id: str
    position: str
    jersey_number: Optional[int] = None
    height: Optional[str] = None
    weight: Optional[int] = None
    college: Optional[str] = None
    years_pro: Optional[int] = None
    # Enhanced fields for sentiment analysis
    aliases: List[str] = Field(default_factory=list)  # Nicknames and alternative names
    keywords: List[str] = Field(default_factory=list)  # Player-specific keywords
    status: str = "active"  # active, injured, suspended, retired
    contract_year: Optional[int] = None
    is_rookie: bool = False
    fantasy_relevant: bool = True

    class Config:
        populate_by_name = True


class BettingLine(BaseModel):
    sportsbook: str
    spread: Optional[float] = None
    over_under: Optional[float] = None
    home_moneyline: Optional[int] = None
    away_moneyline: Optional[int] = None
    last_updated: datetime


class Game(BaseModel):
    id: str = Field(alias="_id")
    home_team_id: str
    away_team_id: str
    game_date: datetime
    week: int
    season: int
    status: GameStatus = GameStatus.SCHEDULED
    home_score: Optional[int] = None
    away_score: Optional[int] = None
    betting_lines: List[BettingLine] = Field(default_factory=list)
    venue: Optional[str] = None
    weather: Optional[Dict] = None
    # Enhanced fields for sentiment analysis
    game_type: str = "regular"  # regular, playoff, championship
    primetime: bool = False
    rivalry_game: bool = False
    playoff_implications: bool = False
    keywords: List[str] = Field(default_factory=list)  # Game-specific keywords

    class Config:
        populate_by_name = True


class GameResponse(Game):
    home_team: Optional[Team] = None
    away_team: Optional[Team] = None


class Prediction(BaseModel):
    game_id: str
    predicted_winner: str  # team_id
    confidence: float = Field(..., ge=0.0, le=1.0)
    predicted_spread: Optional[float] = None
    predicted_total: Optional[float] = None
    model_version: str
    created_at: datetime
    factors: Dict[str, float] = Field(
        default_factory=dict
    )  # sentiment, betting_trends, etc.
