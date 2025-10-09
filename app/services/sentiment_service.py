"""
Enhanced sentiment analysis service with NFL-specific features.
"""

import re
import time
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from ..models.sentiment import (
    SentimentLabel,
    SentimentCategory,
    SentimentResult,
    AnalysisContext,
    NFLContext,
    DataSource,
)
from ..models.nfl_sentiment_config import NFLSentimentConfig

logger = logging.getLogger(__name__)


class SentimentAnalysisService:
    """Enhanced sentiment analysis service with NFL-specific features"""

    def __init__(self):
        self.nfl_config = NFLSentimentConfig()
        self.model_version = "1.0"

    async def analyze_text(
        self,
        text: str,
        context: Optional[AnalysisContext] = None,
        team_id: Optional[str] = None,
        player_id: Optional[str] = None,
        game_id: Optional[str] = None,
        source: DataSource = DataSource.USER_INPUT,
    ) -> SentimentResult:
        """Analyze sentiment of a single text with NFL-specific context"""
        start_time = time.time()

        # Preprocess text
        processed_text = self._preprocess_text(text)

        # Extract NFL context
        nfl_context = self._extract_nfl_context(
            processed_text, team_id, player_id, game_id
        )

        # Determine category
        category = self._categorize_text(processed_text)

        # Perform sentiment analysis
        sentiment_score, confidence = self._analyze_sentiment(
            processed_text, nfl_context, category
        )
        sentiment_label = self._score_to_label(sentiment_score)

        # Calculate detailed breakdowns
        emotion_scores = self._calculate_emotion_scores(processed_text)
        aspect_sentiments = self._calculate_aspect_sentiments(
            processed_text, nfl_context
        )
        keyword_contributions = self._calculate_keyword_contributions(processed_text)

        # Create enhanced context
        if context is None:
            context = AnalysisContext()

        context.nfl_context = nfl_context
        context.sentiment_weights = self._get_sentiment_weights(
            processed_text, category
        )

        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return SentimentResult(
            text=text,
            sentiment=sentiment_label,
            sentiment_score=sentiment_score,
            confidence=confidence,
            category=category,
            context=context,
            team_id=team_id,
            player_id=player_id,
            game_id=game_id,
            source=source,
            timestamp=datetime.utcnow(),
            model_version=self.model_version,
            processing_time_ms=processing_time,
            emotion_scores=emotion_scores,
            aspect_sentiments=aspect_sentiments,
            keyword_contributions=keyword_contributions,
        )

    async def analyze_batch(
        self,
        texts: List[str],
        context: Optional[AnalysisContext] = None,
        team_id: Optional[str] = None,
        player_id: Optional[str] = None,
        game_id: Optional[str] = None,
        source: DataSource = DataSource.USER_INPUT,
    ) -> List[SentimentResult]:
        """Analyze sentiment of multiple texts in batch"""
        results = []

        # Process texts concurrently for better performance
        tasks = [
            self.analyze_text(text, context, team_id, player_id, game_id, source)
            for text in texts
        ]

        results = await asyncio.gather(*tasks)
        return results

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis"""
        # Convert to lowercase for keyword matching
        processed = text.lower()

        # Remove URLs
        processed = re.sub(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
            "",
            processed,
        )

        # Remove excessive whitespace
        processed = re.sub(r"\s+", " ", processed).strip()

        return processed

    def _extract_nfl_context(
        self,
        text: str,
        team_id: Optional[str] = None,
        player_id: Optional[str] = None,
        game_id: Optional[str] = None,
    ) -> NFLContext:
        """Extract NFL-specific context from text"""
        nfl_context = NFLContext()

        # Extract team mentions
        for team, aliases in self.nfl_config.mappings.TEAM_ALIASES.items():
            if team in text or any(alias in text for alias in aliases):
                nfl_context.team_mentions.append(team)

        # Extract position mentions
        for position, aliases in self.nfl_config.mappings.POSITION_MAPPINGS.items():
            if any(alias in text for alias in aliases):
                nfl_context.position_mentions.append(position)

        # Detect injury-related content
        nfl_context.injury_related = any(
            keyword in text for keyword in self.nfl_config.keywords.INJURY_KEYWORDS
        )

        # Detect trade-related content
        nfl_context.trade_related = any(
            keyword in text for keyword in self.nfl_config.keywords.TRADE_KEYWORDS
        )

        # Detect game situation
        for situation, keywords in self.nfl_config.mappings.GAME_SITUATIONS.items():
            if any(keyword in text for keyword in keywords):
                nfl_context.game_situation = situation
                break

        return nfl_context

    def _categorize_text(self, text: str) -> SentimentCategory:
        """Categorize text based on NFL-specific keywords"""
        category_scores = {
            SentimentCategory.PERFORMANCE: self._count_keywords(
                text,
                self.nfl_config.keywords.POSITIVE_PERFORMANCE.union(
                    self.nfl_config.keywords.NEGATIVE_PERFORMANCE
                ),
            ),
            SentimentCategory.INJURY: self._count_keywords(
                text, self.nfl_config.keywords.INJURY_KEYWORDS
            ),
            SentimentCategory.TRADE: self._count_keywords(
                text, self.nfl_config.keywords.TRADE_KEYWORDS
            ),
            SentimentCategory.COACHING: self._count_keywords(
                text, self.nfl_config.keywords.COACHING_KEYWORDS
            ),
            SentimentCategory.BETTING: self._count_keywords(
                text, self.nfl_config.keywords.BETTING_KEYWORDS
            ),
        }

        # Return category with highest score
        max_score = max(category_scores.values())
        if max_score == 0:
            return SentimentCategory.GENERAL

        return max(category_scores, key=category_scores.get)

    def _count_keywords(self, text: str, keywords: set) -> int:
        """Count occurrences of keywords in text"""
        return sum(1 for keyword in keywords if keyword in text)

    def _analyze_sentiment(
        self, text: str, nfl_context: NFLContext, category: SentimentCategory
    ) -> Tuple[float, float]:
        """Perform NFL-specific sentiment analysis"""
        # Basic sentiment scoring based on keywords
        positive_score = self._count_keywords(
            text, self.nfl_config.keywords.POSITIVE_PERFORMANCE
        )
        negative_score = self._count_keywords(
            text, self.nfl_config.keywords.NEGATIVE_PERFORMANCE
        )

        # Apply category weights
        category_weight = self.nfl_config.weights.CATEGORY_WEIGHTS.get(
            category.value, 1.0
        )

        # Calculate base sentiment score
        if positive_score + negative_score == 0:
            sentiment_score = 0.0
            confidence = 0.3  # Low confidence for neutral content
        else:
            sentiment_score = (positive_score - negative_score) / (
                positive_score + negative_score
            )
            confidence = min(0.9, 0.5 + (positive_score + negative_score) * 0.1)

        # Apply NFL-specific adjustments
        if nfl_context.injury_related:
            sentiment_score *= 0.8  # Injury news tends to be negative
            confidence *= 0.9

        if nfl_context.trade_related:
            sentiment_score *= 0.9  # Trade news can be mixed

        # Apply category weight
        sentiment_score *= category_weight

        # Normalize to [-1, 1] range
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        confidence = max(0.0, min(1.0, confidence))

        return sentiment_score, confidence

    def _score_to_label(self, score: float) -> SentimentLabel:
        """Convert sentiment score to label"""
        if score > 0.1:
            return SentimentLabel.POSITIVE
        elif score < -0.1:
            return SentimentLabel.NEGATIVE
        else:
            return SentimentLabel.NEUTRAL

    def _calculate_emotion_scores(self, text: str) -> Dict[str, float]:
        """Calculate emotion scores (placeholder implementation)"""
        # This would integrate with a more sophisticated emotion analysis model
        return {
            "joy": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "sadness": 0.0,
            "surprise": 0.0,
            "trust": 0.0,
        }

    def _calculate_aspect_sentiments(
        self, text: str, nfl_context: NFLContext
    ) -> Dict[str, float]:
        """Calculate sentiment for different aspects"""
        aspects = {}

        # Performance aspect
        perf_positive = self._count_keywords(
            text, self.nfl_config.keywords.POSITIVE_PERFORMANCE
        )
        perf_negative = self._count_keywords(
            text, self.nfl_config.keywords.NEGATIVE_PERFORMANCE
        )
        if perf_positive + perf_negative > 0:
            aspects["performance"] = (perf_positive - perf_negative) / (
                perf_positive + perf_negative
            )

        # Injury aspect
        if nfl_context.injury_related:
            aspects["injury"] = -0.3  # Generally negative

        # Trade aspect
        if nfl_context.trade_related:
            aspects["trade"] = 0.0  # Neutral by default

        return aspects

    def _calculate_keyword_contributions(self, text: str) -> Dict[str, float]:
        """Calculate how each keyword contributes to sentiment"""
        contributions = {}

        # Positive keywords
        for keyword in self.nfl_config.keywords.POSITIVE_PERFORMANCE:
            if keyword in text:
                contributions[keyword] = 0.5

        # Negative keywords
        for keyword in self.nfl_config.keywords.NEGATIVE_PERFORMANCE:
            if keyword in text:
                contributions[keyword] = -0.5

        return contributions

    def _get_sentiment_weights(
        self, text: str, category: SentimentCategory
    ) -> Dict[str, float]:
        """Get sentiment weights for the analysis"""
        weights = {}

        # Category weight
        weights["category"] = self.nfl_config.weights.CATEGORY_WEIGHTS.get(
            category.value, 1.0
        )

        # Keyword-based weights
        for keyword in self.nfl_config.get_all_keywords():
            if keyword in text:
                weights[keyword] = self.nfl_config.get_keyword_sentiment_weight(
                    keyword, category.value
                )

        return weights


# Global sentiment service instance
sentiment_service = SentimentAnalysisService()


    async def get_recent_sentiment(self, limit: int = 10) -> List[Dict]:
        """Get recent sentiment analyses"""
        # This would typically query the database for recent sentiment analyses
        # For now, return a placeholder implementation
        from ..core.database import get_database
        
        try:
            db = await get_database()
            cursor = db.sentiment_analyses.find({}).sort([("created_at", -1)]).limit(limit)
            
            recent_sentiments = []
            async for doc in cursor:
                doc["id"] = str(doc["_id"])
                doc.pop("_id", None)
                # Convert datetime objects to ISO strings
                if "created_at" in doc and hasattr(doc["created_at"], "isoformat"):
                    doc["created_at"] = doc["created_at"].isoformat()
                if "timestamp" in doc and hasattr(doc["timestamp"], "isoformat"):
                    doc["timestamp"] = doc["timestamp"].isoformat()
                recent_sentiments.append(doc)
            
            return recent_sentiments
        except Exception as e:
            logger.error(f"Error getting recent sentiment: {e}")
            return []

    async def get_team_sentiment(self, team_id: str) -> Dict:
        """Get sentiment analysis for a specific team"""
        # This would typically query the database for team-specific sentiment
        # For now, return a placeholder implementation
        from ..core.database import get_database
        from datetime import timedelta
        
        try:
            db = await get_database()
            
            # Get recent sentiment for the team (last 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            cursor = db.sentiment_analyses.find({
                "team_id": team_id,
                "created_at": {"$gte": week_ago}
            }).sort([("created_at", -1)])
            
            sentiments = []
            total_score = 0
            count = 0
            
            async for doc in cursor:
                doc["id"] = str(doc["_id"])
                doc.pop("_id", None)
                # Convert datetime objects to ISO strings
                if "created_at" in doc and hasattr(doc["created_at"], "isoformat"):
                    doc["created_at"] = doc["created_at"].isoformat()
                if "timestamp" in doc and hasattr(doc["timestamp"], "isoformat"):
                    doc["timestamp"] = doc["timestamp"].isoformat()
                sentiments.append(doc)
                
                if "sentiment_score" in doc:
                    total_score += doc["sentiment_score"]
                    count += 1
            
            avg_sentiment = total_score / count if count > 0 else 0.0
            
            return {
                "team_id": team_id,
                "average_sentiment": round(avg_sentiment, 3),
                "total_analyses": count,
                "recent_sentiments": sentiments[:10],  # Limit to 10 most recent
                "period": "last_7_days"
            }
        except Exception as e:
            logger.error(f"Error getting team sentiment: {e}")
            return {
                "team_id": team_id,
                "average_sentiment": 0.0,
                "total_analyses": 0,
                "recent_sentiments": [],
                "period": "last_7_days"
            }


async def get_sentiment_service() -> SentimentAnalysisService:
    """Dependency to get sentiment service"""
    return sentiment_service
