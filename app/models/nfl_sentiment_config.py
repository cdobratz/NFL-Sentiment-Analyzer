"""
NFL-specific sentiment analysis configuration including keywords, weights, and context mappings.
"""

from typing import Dict, List, Set
from enum import Enum


class NFLSentimentKeywords:
    """NFL-specific keyword dictionaries for enhanced sentiment analysis"""
    
    # Positive performance keywords
    POSITIVE_PERFORMANCE = {
        "touchdown", "td", "score", "win", "victory", "champion", "mvp", "pro bowl",
        "all-pro", "record", "dominant", "explosive", "clutch", "elite", "superstar",
        "breakout", "comeback", "amazing", "incredible", "outstanding", "phenomenal",
        "beast mode", "money", "perfect", "flawless", "unstoppable", "legendary"
    }
    
    # Negative performance keywords
    NEGATIVE_PERFORMANCE = {
        "fumble", "interception", "int", "sack", "penalty", "flag", "turnover",
        "loss", "defeat", "blown", "choke", "bust", "terrible", "awful", "horrible",
        "disaster", "collapse", "meltdown", "embarrassing", "pathetic", "trash",
        "overrated", "washed", "done", "finished", "benched", "cut", "released"
    }
    
    # Injury-related keywords
    INJURY_KEYWORDS = {
        "injury", "injured", "hurt", "pain", "surgery", "ir", "injured reserve",
        "concussion", "acl", "mcl", "hamstring", "ankle", "knee", "shoulder",
        "back", "neck", "questionable", "doubtful", "out", "sidelined", "rehab",
        "recovery", "healing", "return", "cleared", "healthy"
    }
    
    # Trade and contract keywords
    TRADE_KEYWORDS = {
        "trade", "traded", "deal", "contract", "extension", "signing", "signed",
        "free agent", "fa", "franchise tag", "holdout", "negotiation", "salary",
        "cap", "money", "paid", "worth", "value", "overpaid", "underpaid",
        "restructure", "release", "cut", "waive"
    }
    
    # Coaching keywords
    COACHING_KEYWORDS = {
        "coach", "coaching", "playcall", "play calling", "strategy", "scheme",
        "gameplan", "timeout", "challenge", "decision", "management", "leadership",
        "fired", "hired", "promoted", "demoted", "coordinator", "staff"
    }
    
    # Betting and prediction keywords
    BETTING_KEYWORDS = {
        "bet", "betting", "odds", "line", "spread", "over", "under", "moneyline",
        "favorite", "underdog", "pick", "prediction", "lock", "sure thing",
        "value", "sharp", "public", "fade", "hammer", "smash", "play"
    }
    
    # Fantasy football keywords
    FANTASY_KEYWORDS = {
        "fantasy", "start", "sit", "bench", "waiver", "pickup", "drop", "trade",
        "keeper", "dynasty", "redraft", "ppr", "standard", "flex", "sleeper",
        "bust", "boom", "floor", "ceiling", "target", "touches", "usage"
    }


class NFLSentimentWeights:
    """Sentiment weights for different NFL contexts and keywords"""
    
    # Base weights for different categories
    CATEGORY_WEIGHTS = {
        "performance": 1.0,
        "injury": 0.8,
        "trade": 0.6,
        "coaching": 0.7,
        "betting": 0.5,
        "fantasy": 0.4
    }
    
    # Position-specific weights (some positions get more attention)
    POSITION_WEIGHTS = {
        "QB": 1.5,
        "RB": 1.2,
        "WR": 1.2,
        "TE": 1.0,
        "K": 0.6,
        "DEF": 0.8,
        "OL": 0.7,
        "DL": 0.8,
        "LB": 0.9,
        "DB": 0.9
    }
    
    # Source reliability weights
    SOURCE_WEIGHTS = {
        "espn": 1.0,
        "twitter": 0.7,
        "reddit": 0.6,
        "news": 0.9,
        "betting": 0.8,
        "fantasy": 0.5,
        "user_input": 0.3
    }
    
    # Time-based weights (recent content is more relevant)
    TIME_DECAY_HOURS = {
        1: 1.0,
        6: 0.9,
        24: 0.7,
        72: 0.5,
        168: 0.3,  # 1 week
        720: 0.1   # 1 month
    }


class NFLContextMappings:
    """Mappings for NFL-specific context and entity recognition"""
    
    # Common team nicknames and aliases
    TEAM_ALIASES = {
        "patriots": ["pats", "new england"],
        "cowboys": ["america's team", "dallas"],
        "packers": ["green bay", "pack"],
        "steelers": ["pittsburgh", "steel curtain"],
        "49ers": ["niners", "san francisco"],
        "chiefs": ["kansas city", "kc"],
        "bills": ["buffalo"],
        "dolphins": ["miami", "fins"],
        "jets": ["new york jets", "nyj"],
        "ravens": ["baltimore"],
        "browns": ["cleveland"],
        "bengals": ["cincinnati", "cincy"],
        "titans": ["tennessee"],
        "colts": ["indianapolis", "indy"],
        "texans": ["houston"],
        "jaguars": ["jacksonville", "jags"],
        "broncos": ["denver"],
        "chargers": ["los angeles chargers", "lac"],
        "raiders": ["las vegas", "lv"],
        "rams": ["los angeles rams", "lar"],
        "cardinals": ["arizona", "az"],
        "seahawks": ["seattle"],
        "giants": ["new york giants", "nyg"],
        "eagles": ["philadelphia", "philly"],
        "commanders": ["washington"],
        "lions": ["detroit"],
        "bears": ["chicago"],
        "vikings": ["minnesota"],
        "saints": ["new orleans"],
        "falcons": ["atlanta"],
        "panthers": ["carolina"],
        "buccaneers": ["tampa bay", "bucs"]
    }
    
    # Position abbreviations and full names
    POSITION_MAPPINGS = {
        "QB": ["quarterback", "qb"],
        "RB": ["running back", "runningback", "rb", "halfback", "hb"],
        "FB": ["fullback", "fb"],
        "WR": ["wide receiver", "receiver", "wr"],
        "TE": ["tight end", "te"],
        "OL": ["offensive line", "o-line", "ol"],
        "C": ["center"],
        "G": ["guard"],
        "T": ["tackle"],
        "DL": ["defensive line", "d-line", "dl"],
        "DE": ["defensive end", "de"],
        "DT": ["defensive tackle", "dt"],
        "LB": ["linebacker", "lb"],
        "DB": ["defensive back", "db"],
        "CB": ["cornerback", "corner", "cb"],
        "S": ["safety"],
        "FS": ["free safety", "fs"],
        "SS": ["strong safety", "ss"],
        "K": ["kicker"],
        "P": ["punter"],
        "LS": ["long snapper", "ls"]
    }
    
    # Game situation contexts
    GAME_SITUATIONS = {
        "playoff": ["playoffs", "postseason", "wildcard", "divisional", "championship"],
        "primetime": ["monday night", "sunday night", "thursday night", "mnf", "snf", "tnf"],
        "rivalry": ["rivalry", "division", "divisional", "hate", "enemy"],
        "clutch": ["clutch", "4th quarter", "overtime", "ot", "game winning", "walk off"]
    }


class NFLSentimentConfig:
    """Main configuration class for NFL sentiment analysis"""
    
    def __init__(self):
        self.keywords = NFLSentimentKeywords()
        self.weights = NFLSentimentWeights()
        self.mappings = NFLContextMappings()
    
    def get_keyword_sentiment_weight(self, keyword: str, category: str = "general") -> float:
        """Get sentiment weight for a specific keyword and category"""
        base_weight = self.weights.CATEGORY_WEIGHTS.get(category, 1.0)
        
        # Check if keyword is in positive or negative sets
        if keyword.lower() in self.keywords.POSITIVE_PERFORMANCE:
            return base_weight * 1.2
        elif keyword.lower() in self.keywords.NEGATIVE_PERFORMANCE:
            return base_weight * 1.2
        elif keyword.lower() in self.keywords.INJURY_KEYWORDS:
            return base_weight * 0.9  # Injury news is often negative but not always
        
        return base_weight
    
    def get_all_keywords(self) -> Set[str]:
        """Get all NFL-specific keywords for preprocessing"""
        all_keywords = set()
        all_keywords.update(self.keywords.POSITIVE_PERFORMANCE)
        all_keywords.update(self.keywords.NEGATIVE_PERFORMANCE)
        all_keywords.update(self.keywords.INJURY_KEYWORDS)
        all_keywords.update(self.keywords.TRADE_KEYWORDS)
        all_keywords.update(self.keywords.COACHING_KEYWORDS)
        all_keywords.update(self.keywords.BETTING_KEYWORDS)
        all_keywords.update(self.keywords.FANTASY_KEYWORDS)
        return all_keywords
    
    def categorize_text(self, text: str) -> str:
        """Categorize text based on keyword presence"""
        text_lower = text.lower()
        
        # Count keywords in each category
        category_scores = {
            "performance": sum(1 for kw in self.keywords.POSITIVE_PERFORMANCE.union(
                self.keywords.NEGATIVE_PERFORMANCE) if kw in text_lower),
            "injury": sum(1 for kw in self.keywords.INJURY_KEYWORDS if kw in text_lower),
            "trade": sum(1 for kw in self.keywords.TRADE_KEYWORDS if kw in text_lower),
            "coaching": sum(1 for kw in self.keywords.COACHING_KEYWORDS if kw in text_lower),
            "betting": sum(1 for kw in self.keywords.BETTING_KEYWORDS if kw in text_lower),
            "fantasy": sum(1 for kw in self.keywords.FANTASY_KEYWORDS if kw in text_lower)
        }
        
        # Return category with highest score
        if max(category_scores.values()) == 0:
            return "general"
        
        return max(category_scores, key=category_scores.get)