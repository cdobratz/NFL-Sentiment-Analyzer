"""
Configuration for data ingestion service
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion service"""

    # Twitter/X API configuration
    twitter_keywords: List[str] = None
    twitter_max_results: int = 100
    twitter_collection_interval: int = 5  # minutes

    # ESPN API configuration
    espn_collection_interval: int = 15  # minutes

    # Betting APIs configuration
    betting_collection_interval: int = 30  # minutes
    betting_sportsbooks: List[str] = None

    # Data processing configuration
    max_queue_size: int = 1000
    worker_count: int = 3
    data_retention_days: int = 30

    # Rate limiting configuration
    twitter_rate_limit: int = 300  # requests per 15 minutes
    espn_rate_limit: int = 100  # requests per hour
    betting_rate_limit: int = 60  # requests per minute

    def __post_init__(self):
        """
        Populate default values after initialization.
        
        If `twitter_keywords` was not provided, set a curated list of NFL- and football-related search terms. If `betting_sportsbooks` was not provided, set a default list of common sportsbooks.
        """
        if self.twitter_keywords is None:
            self.twitter_keywords = [
                "NFL",
                "football",
                "Chiefs",
                "Bills",
                "Cowboys",
                "Patriots",
                "Packers",
                "Steelers",
                "49ers",
                "Rams",
                "Super Bowl",
                "playoff",
                "quarterback",
                "touchdown",
                "fantasy football",
                "NFL draft",
                "injury report",
                "trade",
                "free agency",
            ]

        if self.betting_sportsbooks is None:
            self.betting_sportsbooks = ["draftkings", "mgm", "fanduel", "caesars"]


# Default configuration instance
default_config = DataIngestionConfig()


# NFL team keywords for better sentiment analysis
NFL_TEAM_KEYWORDS = {
    "arizona_cardinals": ["Cardinals", "Arizona Cardinals", "ARI", "Red Birds"],
    "atlanta_falcons": ["Falcons", "Atlanta Falcons", "ATL", "Dirty Birds"],
    "baltimore_ravens": ["Ravens", "Baltimore Ravens", "BAL", "Purple Birds"],
    "buffalo_bills": ["Bills", "Buffalo Bills", "BUF", "Bills Mafia"],
    "carolina_panthers": ["Panthers", "Carolina Panthers", "CAR", "Keep Pounding"],
    "chicago_bears": ["Bears", "Chicago Bears", "CHI", "Da Bears"],
    "cincinnati_bengals": ["Bengals", "Cincinnati Bengals", "CIN", "Who Dey"],
    "cleveland_browns": ["Browns", "Cleveland Browns", "CLE", "Dawg Pound"],
    "dallas_cowboys": ["Cowboys", "Dallas Cowboys", "DAL", "America's Team"],
    "denver_broncos": ["Broncos", "Denver Broncos", "DEN", "Orange Crush"],
    "detroit_lions": ["Lions", "Detroit Lions", "DET", "Motor City"],
    "green_bay_packers": ["Packers", "Green Bay Packers", "GB", "Cheeseheads"],
    "houston_texans": ["Texans", "Houston Texans", "HOU", "Bull Red"],
    "indianapolis_colts": ["Colts", "Indianapolis Colts", "IND", "Horseshoe"],
    "jacksonville_jaguars": ["Jaguars", "Jacksonville Jaguars", "JAX", "Jags"],
    "kansas_city_chiefs": ["Chiefs", "Kansas City Chiefs", "KC", "Kingdom"],
    "las_vegas_raiders": ["Raiders", "Las Vegas Raiders", "LV", "Raider Nation"],
    "los_angeles_chargers": ["Chargers", "Los Angeles Chargers", "LAC", "Bolt Up"],
    "los_angeles_rams": ["Rams", "Los Angeles Rams", "LAR", "Whose House"],
    "miami_dolphins": ["Dolphins", "Miami Dolphins", "MIA", "Fins Up"],
    "minnesota_vikings": ["Vikings", "Minnesota Vikings", "MIN", "Skol"],
    "new_england_patriots": ["Patriots", "New England Patriots", "NE", "Pats"],
    "new_orleans_saints": ["Saints", "New Orleans Saints", "NO", "Who Dat"],
    "new_york_giants": ["Giants", "New York Giants", "NYG", "Big Blue"],
    "new_york_jets": ["Jets", "New York Jets", "NYJ", "Gang Green"],
    "philadelphia_eagles": ["Eagles", "Philadelphia Eagles", "PHI", "Fly Eagles Fly"],
    "pittsburgh_steelers": ["Steelers", "Pittsburgh Steelers", "PIT", "Steel Curtain"],
    "san_francisco_49ers": ["49ers", "San Francisco 49ers", "SF", "Niners"],
    "seattle_seahawks": ["Seahawks", "Seattle Seahawks", "SEA", "12th Man"],
    "tampa_bay_buccaneers": ["Buccaneers", "Tampa Bay Buccaneers", "TB", "Bucs"],
    "tennessee_titans": ["Titans", "Tennessee Titans", "TEN", "Titan Up"],
    "washington_commanders": ["Commanders", "Washington Commanders", "WAS", "HTTC"],
}


# Common NFL-related keywords for sentiment analysis
NFL_GENERAL_KEYWORDS = [
    "NFL",
    "National Football League",
    "football",
    "American football",
    "Super Bowl",
    "playoffs",
    "postseason",
    "regular season",
    "quarterback",
    "QB",
    "running back",
    "RB",
    "wide receiver",
    "WR",
    "tight end",
    "TE",
    "offensive line",
    "defense",
    "linebacker",
    "LB",
    "cornerback",
    "CB",
    "safety",
    "defensive back",
    "DB",
    "touchdown",
    "field goal",
    "interception",
    "fumble",
    "sack",
    "draft",
    "NFL draft",
    "rookie",
    "veteran",
    "free agency",
    "trade",
    "injury",
    "injury report",
    "IR",
    "practice squad",
    "fantasy football",
    "fantasy",
    "DFS",
    "daily fantasy",
    "betting",
    "odds",
    "spread",
    "over under",
    "moneyline",
    "coach",
    "head coach",
    "offensive coordinator",
    "defensive coordinator",
    "owner",
    "general manager",
    "GM",
    "front office",
]
