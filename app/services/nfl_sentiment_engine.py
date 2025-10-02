"""
Enhanced NFL Sentiment Analysis Engine with team and player context,
batch processing, and confidence scoring.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time

from ..models.sentiment import (
    SentimentLabel,
    SentimentCategory,
    SentimentResult,
    AnalysisContext,
    NFLContext,
    DataSource,
    BatchSentimentResponse,
)
from ..models.nfl_sentiment_config import NFLSentimentConfig
from ..services.sentiment_service import SentimentAnalysisService

logger = logging.getLogger(__name__)


class NFLSentimentEngine:
    """
    Enhanced sentiment analysis engine with NFL-specific features including:
    - Team and player context awareness
    - NFL-specific keyword dictionaries and sentiment weights
    - Batch processing capabilities
    - Confidence scoring and sentiment categorization
    """

    def __init__(self):
        self.nfl_config = NFLSentimentConfig()
        self.sentiment_service = SentimentAnalysisService()
        self.model_version = "1.0-nfl-enhanced"

        # Performance metrics
        self.total_analyses = 0
        self.total_processing_time = 0.0

    async def analyze_with_context(
        self,
        text: str,
        team_context: Optional[str] = None,
        player_context: Optional[str] = None,
        game_context: Optional[str] = None,
        source: DataSource = DataSource.USER_INPUT,
        include_detailed_analysis: bool = False,
    ) -> SentimentResult:
        """
        Analyze sentiment with NFL-specific context enhancement

        Args:
            text: Text to analyze
            team_context: Team ID or name for context
            player_context: Player ID or name for context
            game_context: Game ID for context
            source: Data source of the text
            include_detailed_analysis: Whether to include detailed breakdowns

        Returns:
            Enhanced sentiment result with NFL context
        """
        start_time = time.time()

        try:
            # Create enhanced context
            context = AnalysisContext()

            # Extract NFL-specific context
            nfl_context = self._extract_enhanced_nfl_context(
                text, team_context, player_context, game_context
            )
            context.nfl_context = nfl_context

            # Perform sentiment analysis with context
            result = await self.sentiment_service.analyze_text(
                text=text,
                context=context,
                team_id=team_context,
                player_id=player_context,
                game_id=game_context,
                source=source,
            )

            # Enhance with NFL-specific features
            result = await self._enhance_with_nfl_features(
                result, include_detailed_analysis
            )

            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_analyses += 1
            self.total_processing_time += processing_time

            return result

        except Exception as e:
            logger.error(f"Error in NFL sentiment analysis: {str(e)}")
            raise

    async def batch_analyze_with_context(
        self,
        texts: List[str],
        team_context: Optional[str] = None,
        player_context: Optional[str] = None,
        game_context: Optional[str] = None,
        source: DataSource = DataSource.USER_INPUT,
        include_detailed_analysis: bool = False,
        max_concurrent: int = 10,
    ) -> BatchSentimentResponse:
        """
        Batch analyze multiple texts with NFL context and optimized processing

        Args:
            texts: List of texts to analyze
            team_context: Team ID or name for context
            player_context: Player ID or name for context
            game_context: Game ID for context
            source: Data source of the texts
            include_detailed_analysis: Whether to include detailed breakdowns
            max_concurrent: Maximum concurrent analyses

        Returns:
            Batch sentiment response with aggregated metrics
        """
        start_time = time.time()

        if len(texts) > 100:
            raise ValueError("Maximum 100 texts allowed per batch")

        try:
            # Process texts in batches to avoid overwhelming the system
            results = []

            # Split into chunks for concurrent processing
            chunks = [
                texts[i : i + max_concurrent]
                for i in range(0, len(texts), max_concurrent)
            ]

            for chunk in chunks:
                # Process chunk concurrently
                tasks = [
                    self.analyze_with_context(
                        text=text,
                        team_context=team_context,
                        player_context=player_context,
                        game_context=game_context,
                        source=source,
                        include_detailed_analysis=include_detailed_analysis,
                    )
                    for text in chunk
                ]

                chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out exceptions and log errors
                for i, result in enumerate(chunk_results):
                    if isinstance(result, Exception):
                        logger.error(f"Error processing text {i}: {str(result)}")
                    else:
                        results.append(result)

            # Calculate aggregated metrics
            total_processing_time = (time.time() - start_time) * 1000
            aggregated_sentiment = self._calculate_aggregated_sentiment(results)
            sentiment_distribution = self._calculate_sentiment_distribution(results)

            return BatchSentimentResponse(
                results=results,
                total_processed=len(results),
                processing_time_ms=total_processing_time,
                model_version=self.model_version,
                aggregated_sentiment=aggregated_sentiment,
                sentiment_distribution=sentiment_distribution,
            )

        except Exception as e:
            logger.error(f"Error in batch NFL sentiment analysis: {str(e)}")
            raise

    def get_nfl_keywords_for_entity(
        self, entity_type: str, entity_id: str
    ) -> Dict[str, List[str]]:
        """
        Get NFL-specific keywords for a team, player, or game

        Args:
            entity_type: "team", "player", or "game"
            entity_id: ID of the entity

        Returns:
            Dictionary of keyword categories and their keywords
        """
        keywords = {
            "positive": list(self.nfl_config.keywords.POSITIVE_PERFORMANCE),
            "negative": list(self.nfl_config.keywords.NEGATIVE_PERFORMANCE),
            "injury": list(self.nfl_config.keywords.INJURY_KEYWORDS),
            "trade": list(self.nfl_config.keywords.TRADE_KEYWORDS),
            "coaching": list(self.nfl_config.keywords.COACHING_KEYWORDS),
            "betting": list(self.nfl_config.keywords.BETTING_KEYWORDS),
            "fantasy": list(self.nfl_config.keywords.FANTASY_KEYWORDS),
        }

        # Add entity-specific keywords based on type
        if entity_type == "team":
            # Add team-specific aliases and keywords
            for team, aliases in self.nfl_config.mappings.TEAM_ALIASES.items():
                if team.lower() in entity_id.lower():
                    keywords["team_specific"] = aliases
                    break

        elif entity_type == "player":
            # Add position-specific keywords
            keywords["positions"] = []
            for position, aliases in self.nfl_config.mappings.POSITION_MAPPINGS.items():
                keywords["positions"].extend(aliases)

        return keywords

    def calculate_confidence_score(
        self, text: str, sentiment_score: float, nfl_context: NFLContext
    ) -> float:
        """
        Calculate confidence score based on NFL-specific factors

        Args:
            text: Original text
            sentiment_score: Calculated sentiment score
            nfl_context: NFL context information

        Returns:
            Confidence score between 0.0 and 1.0
        """
        base_confidence = 0.5

        # Increase confidence based on keyword matches
        keyword_matches = 0
        all_keywords = self.nfl_config.get_all_keywords()
        text_lower = text.lower()

        for keyword in all_keywords:
            if keyword in text_lower:
                keyword_matches += 1

        # More keyword matches = higher confidence
        keyword_confidence = min(0.4, keyword_matches * 0.05)

        # NFL context increases confidence
        context_confidence = 0.0
        if nfl_context.team_mentions:
            context_confidence += 0.1
        if nfl_context.player_mentions:
            context_confidence += 0.1
        if nfl_context.position_mentions:
            context_confidence += 0.05
        if nfl_context.game_situation:
            context_confidence += 0.1

        # Text length factor (longer texts generally more reliable)
        length_factor = min(0.1, len(text.split()) * 0.01)

        # Combine factors
        total_confidence = (
            base_confidence + keyword_confidence + context_confidence + length_factor
        )

        # Cap at 0.95 (never 100% confident)
        return min(0.95, total_confidence)

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the sentiment engine"""
        avg_processing_time = (
            self.total_processing_time / self.total_analyses
            if self.total_analyses > 0
            else 0.0
        )

        return {
            "total_analyses": self.total_analyses,
            "total_processing_time_seconds": self.total_processing_time,
            "average_processing_time_seconds": avg_processing_time,
            "model_version": self.model_version,
            "nfl_keywords_count": len(self.nfl_config.get_all_keywords()),
            "supported_categories": [cat.value for cat in SentimentCategory],
            "supported_sources": [src.value for src in DataSource],
        }

    def _extract_enhanced_nfl_context(
        self,
        text: str,
        team_context: Optional[str] = None,
        player_context: Optional[str] = None,
        game_context: Optional[str] = None,
    ) -> NFLContext:
        """Extract enhanced NFL context from text and provided context"""
        nfl_context = NFLContext()
        text_lower = text.lower()

        # Extract team mentions (from text and context)
        for team, aliases in self.nfl_config.mappings.TEAM_ALIASES.items():
            if team in text_lower or any(alias in text_lower for alias in aliases):
                nfl_context.team_mentions.append(team)

        if team_context and team_context not in nfl_context.team_mentions:
            nfl_context.team_mentions.append(team_context)

        # Extract position mentions
        for position, aliases in self.nfl_config.mappings.POSITION_MAPPINGS.items():
            if any(alias in text_lower for alias in aliases):
                nfl_context.position_mentions.append(position)

        # Detect specific NFL contexts
        nfl_context.injury_related = any(
            keyword in text_lower
            for keyword in self.nfl_config.keywords.INJURY_KEYWORDS
        )

        nfl_context.trade_related = any(
            keyword in text_lower for keyword in self.nfl_config.keywords.TRADE_KEYWORDS
        )

        # Detect game situation
        for situation, keywords in self.nfl_config.mappings.GAME_SITUATIONS.items():
            if any(keyword in text_lower for keyword in keywords):
                nfl_context.game_situation = situation
                break

        # Extract performance metrics mentions
        performance_indicators = [
            "yards",
            "touchdowns",
            "sacks",
            "interceptions",
            "rating",
            "qbr",
        ]
        for indicator in performance_indicators:
            if indicator in text_lower:
                # This could be enhanced to extract actual numbers
                nfl_context.performance_metrics[indicator] = 1.0

        return nfl_context

    async def _enhance_with_nfl_features(
        self, result: SentimentResult, include_detailed_analysis: bool
    ) -> SentimentResult:
        """Enhance sentiment result with additional NFL-specific features"""

        # Recalculate confidence with NFL-specific factors
        if result.context.nfl_context:
            enhanced_confidence = self.calculate_confidence_score(
                result.text, result.sentiment_score, result.context.nfl_context
            )
            result.confidence = enhanced_confidence

        # Add NFL-specific aspect sentiments if detailed analysis requested
        if include_detailed_analysis:
            result.aspect_sentiments.update(
                self._calculate_nfl_aspect_sentiments(
                    result.text, result.context.nfl_context
                )
            )

        return result

    def _calculate_nfl_aspect_sentiments(
        self, text: str, nfl_context: Optional[NFLContext]
    ) -> Dict[str, float]:
        """Calculate NFL-specific aspect sentiments"""
        aspects = {}
        text_lower = text.lower()

        # Performance aspect
        perf_positive = sum(
            1
            for kw in self.nfl_config.keywords.POSITIVE_PERFORMANCE
            if kw in text_lower
        )
        perf_negative = sum(
            1
            for kw in self.nfl_config.keywords.NEGATIVE_PERFORMANCE
            if kw in text_lower
        )

        if perf_positive + perf_negative > 0:
            aspects["performance"] = (perf_positive - perf_negative) / (
                perf_positive + perf_negative
            )

        # Coaching aspect
        coaching_mentions = sum(
            1 for kw in self.nfl_config.keywords.COACHING_KEYWORDS if kw in text_lower
        )
        if coaching_mentions > 0:
            # Determine if coaching mentions are positive or negative based on context
            aspects["coaching"] = 0.0  # Neutral by default, could be enhanced

        # Fantasy aspect
        fantasy_mentions = sum(
            1 for kw in self.nfl_config.keywords.FANTASY_KEYWORDS if kw in text_lower
        )
        if fantasy_mentions > 0:
            aspects["fantasy"] = 0.1  # Slightly positive by default

        # Injury aspect (generally negative)
        if nfl_context and nfl_context.injury_related:
            aspects["injury"] = -0.4

        # Trade aspect (mixed sentiment)
        if nfl_context and nfl_context.trade_related:
            aspects["trade"] = 0.0

        return aspects

    def _calculate_aggregated_sentiment(self, results: List[SentimentResult]) -> float:
        """Calculate aggregated sentiment score from batch results"""
        if not results:
            return 0.0

        # Weight by confidence scores
        weighted_sum = sum(r.sentiment_score * r.confidence for r in results)
        total_weight = sum(r.confidence for r in results)

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_sentiment_distribution(
        self, results: List[SentimentResult]
    ) -> Dict[SentimentLabel, int]:
        """Calculate sentiment label distribution from batch results"""
        distribution = {
            SentimentLabel.POSITIVE: 0,
            SentimentLabel.NEGATIVE: 0,
            SentimentLabel.NEUTRAL: 0,
        }

        for result in results:
            distribution[result.sentiment] += 1

        return distribution
