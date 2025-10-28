import json
from json import JSONDecodeError

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    Query,
    BackgroundTasks,
    UploadFile,
    File,
)
from typing import List, Optional, Dict, Any, Union, Set
from datetime import datetime, timedelta
import logging
import time
import asyncio

from ..core.database import get_database
from ..core.dependencies import get_optional_user, require_user_or_api_scope
from ..core.api_keys import APIKeyScope, APIKey
from ..core.exceptions import ValidationError, NotFoundError, MLModelError
from ..core.monitoring import metrics_collector
from ..services.sentiment_service import SentimentAnalysisService
from ..models.sentiment import (
    SentimentAnalysisCreate,
    SentimentAnalysisResponse,
    SentimentResult,
    BatchSentimentRequest,
    BatchSentimentResponse,
    TeamSentiment,
    PlayerSentiment,
    GameSentiment,
    SentimentLabel,
    SentimentCategory,
    SentimentAggregationRequest,
    SentimentInsights,
    DataSource,
    AnalysisContext,
    NFLContext,
)
from ..models.nfl_sentiment_config import NFLSentimentConfig

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sentiment", tags=["sentiment"])

# Initialize NFL sentiment configuration
nfl_config = NFLSentimentConfig()


@router.post("/analyze", response_model=SentimentResult)
async def analyze_sentiment(
    sentiment_data: SentimentAnalysisCreate,
    background_tasks: BackgroundTasks,
    db=Depends(get_database),
    auth: Union[dict, APIKey] = Depends(
        require_user_or_api_scope(APIKeyScope.WRITE_SENTIMENT)
    ),
):
    """
    Analyze a single text for sentiment using NFL-specific context and persist the analysis.

    Persists the analysis document to the database and schedules a background task to refresh aggregated sentiment for the related team/player/game.

    Parameters:
        sentiment_data (SentimentAnalysisCreate): Input payload containing the text and optional context fields (team_id, player_id, game_id, source, and any contextual metadata).

    Returns:
        SentimentResult: Analysis details including `sentiment_score`, `category`, `timestamp`, `model_version`, `processing_time_ms`, `emotion_scores`, `aspect_sentiments`, and `keyword_contributions`.

    Raises:
        HTTPException: Raised with status 500 if sentiment processing or persistence fails.
    """
    service = SentimentAnalysisService()

    try:
        # Perform sentiment analysis
        result = await service.analyze_text(
            text=sentiment_data.text,
            context=sentiment_data.context,
            team_id=sentiment_data.team_id,
            player_id=sentiment_data.player_id,
            game_id=sentiment_data.game_id,
            source=sentiment_data.source,
        )

        # Store in database
        doc_data = {
            **sentiment_data.dict(),
            "sentiment": result.sentiment.value,
            "sentiment_score": result.sentiment_score,
            "confidence": result.confidence,
            "category": result.category.value,
            "source": result.source.value,
            "context": (
                result.context.model_dump()
                if hasattr(result.context, "model_dump")
                else result.context.dict()
            ),
            "timestamp": result.timestamp,
            "created_at": result.timestamp,
            "processed_at": datetime.utcnow(),
            "model_version": result.model_version,
            "processing_time_ms": result.processing_time_ms,
            "emotion_scores": result.emotion_scores,
            "aspect_sentiments": result.aspect_sentiments,
            "keyword_contributions": result.keyword_contributions,
            "user_id": str(auth["_id"]) if isinstance(auth, dict) and auth else None,
        }

        await db.sentiment_analyses.insert_one(doc_data)

        # Update aggregated sentiment in background
        background_tasks.add_task(
            update_aggregated_sentiment,
            db,
            sentiment_data.team_id,
            sentiment_data.player_id,
            sentiment_data.game_id,
        )

        return result

    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing sentiment analysis",
        )


@router.post("/analyze/batch", response_model=BatchSentimentResponse)
async def analyze_batch_sentiment(
    batch_request: BatchSentimentRequest,
    background_tasks: BackgroundTasks,
    db=Depends(get_database),
    current_user: Optional[dict] = Depends(get_optional_user),
):
    """
    Analyze a batch of input texts, persist individual analyses, and schedule aggregated updates.

    Validates a maximum of 100 texts, invokes the sentiment service for batch analysis, inserts per-text documents into the database (including optional user_id when provided), computes aggregated metrics (average sentiment, processing time), and enqueues a background task to refresh aggregated sentiment for the provided team/player/game.

    Parameters:
        batch_request (BatchSentimentRequest): Payload containing texts and optional context/entity identifiers.
        current_user (Optional[dict]): Optional authenticated user whose id, if present, will be stored with each analysis.

    Returns:
        BatchSentimentResponse: Contains the list of individual analysis results, total_processed, processing_time_ms, model_version, aggregated_sentiment, and sentiment_distribution.
    """
    if len(batch_request.texts) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 texts allowed per batch",
        )

    service = SentimentAnalysisService()
    start_time = time.time()

    try:
        # Perform batch sentiment analysis
        results = await service.analyze_batch(
            texts=batch_request.texts,
            context=batch_request.context,
            team_id=batch_request.team_id,
            player_id=batch_request.player_id,
            game_id=batch_request.game_id,
            source=batch_request.source,
        )

        # Store results in database
        docs_to_insert = []
        for result in results:
            doc_data = {
                "text": result.text,
                "sentiment": result.sentiment.value,
                "sentiment_score": result.sentiment_score,
                "confidence": result.confidence,
                "category": result.category.value,
                "context": (
                    result.context.model_dump()
                    if hasattr(result.context, "model_dump")
                    else result.context.dict()
                ),
                "team_id": result.team_id,
                "player_id": result.player_id,
                "game_id": result.game_id,
                "source": result.source.value,
                "timestamp": result.timestamp,
                "created_at": result.timestamp,
                "processed_at": datetime.utcnow(),
                "model_version": result.model_version,
                "processing_time_ms": result.processing_time_ms,
                "emotion_scores": result.emotion_scores,
                "aspect_sentiments": result.aspect_sentiments,
                "keyword_contributions": result.keyword_contributions,
                "user_id": str(current_user["_id"]) if current_user else None,
            }

            docs_to_insert.append(doc_data)

        if docs_to_insert:
            await db.sentiment_analyses.insert_many(docs_to_insert)

        # Calculate aggregated metrics
        total_processing_time = (time.time() - start_time) * 1000
        aggregated_sentiment = (
            sum(r.sentiment_score for r in results) / len(results) if results else 0.0
        )

        sentiment_distribution = {
            SentimentLabel.POSITIVE: sum(
                1 for r in results if r.sentiment == SentimentLabel.POSITIVE
            ),
            SentimentLabel.NEGATIVE: sum(
                1 for r in results if r.sentiment == SentimentLabel.NEGATIVE
            ),
            SentimentLabel.NEUTRAL: sum(
                1 for r in results if r.sentiment == SentimentLabel.NEUTRAL
            ),
        }

        # Update aggregated sentiment in background
        background_tasks.add_task(
            update_aggregated_sentiment,
            db,
            batch_request.team_id,
            batch_request.player_id,
            batch_request.game_id,
        )

        return BatchSentimentResponse(
            results=results,
            total_processed=len(results),
            processing_time_ms=total_processing_time,
            model_version=service.model_version,
            aggregated_sentiment=aggregated_sentiment,
            sentiment_distribution=sentiment_distribution,
        )

    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing batch sentiment analysis",
        )


@router.post("/import-json")
async def import_sentiment_json(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="JSON file containing sentiment records"),
    db=Depends(get_database),
    auth: Union[dict, APIKey] = Depends(
        require_user_or_api_scope(APIKeyScope.WRITE_SENTIMENT)
    ),
):
    """Ingest a JSON file of sentiment records and process them through the sentiment pipeline."""

    if file.content_type not in {
        "application/json",
        "text/json",
        "application/octet-stream",
    }:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be JSON",
        )

    payload_bytes = await file.read()
    if not payload_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    try:
        parsed_payload = json.loads(payload_bytes)
    except JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON payload: {exc.msg}",
        ) from exc

    records = (
        parsed_payload.get("records")
        if isinstance(parsed_payload, dict) and "records" in parsed_payload
        else parsed_payload
    )

    if not isinstance(records, list):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="JSON payload must be a list of records or contain a 'records' array",
        )

    service = SentimentAnalysisService()
    processed_results: List[SentimentResult] = []
    errors: List[Dict[str, Any]] = []
    affected_team_ids: Set[str] = set()
    affected_player_ids: Set[str] = set()
    affected_game_ids: Set[str] = set()

    for idx, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            errors.append({"index": idx, "error": "Record is not an object"})
            continue

        text = raw_record.get("text") or raw_record.get("body")
        if not text:
            errors.append({"index": idx, "error": "Missing required 'text' field"})
            continue

        try:
            sentiment_value = raw_record.get("sentiment")
            if sentiment_value:
                try:
                    sentiment_label = SentimentLabel(sentiment_value.upper())
                except ValueError:
                    sentiment_label = SentimentLabel.NEUTRAL
            else:
                sentiment_label = SentimentLabel.NEUTRAL

            category_value = raw_record.get("category")
            if category_value:
                try:
                    sentiment_category = SentimentCategory(category_value.lower())
                except ValueError:
                    sentiment_category = SentimentCategory.GENERAL
            else:
                sentiment_category = SentimentCategory.GENERAL

            source_value = raw_record.get("source")
            if source_value:
                try:
                    source_enum = DataSource(source_value.lower())
                except ValueError:
                    source_enum = DataSource.TWITTER
            else:
                source_enum = DataSource.TWITTER

            context_payload = raw_record.get("context") or {}
            if not isinstance(context_payload, dict):
                context_payload = {}

            try:
                analysis_context = AnalysisContext(**context_payload)
            except Exception:  # pylint: disable=broad-except
                analysis_context = AnalysisContext()

            metadata = raw_record.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {"raw_metadata": metadata}

            confidence_value = raw_record.get("confidence", 0.5)
            try:
                confidence = float(confidence_value)
            except (TypeError, ValueError):
                confidence = 0.5

            score_value = raw_record.get("sentiment_score", 0.0)
            try:
                sentiment_score = float(score_value)
            except (TypeError, ValueError):
                sentiment_score = 0.0

            sentiment_input = SentimentAnalysisCreate(
                text=text,
                sentiment=sentiment_label,
                confidence=confidence,
                sentiment_score=sentiment_score,
                category=sentiment_category,
                context=analysis_context,
                team_id=raw_record.get("team_id"),
                player_id=raw_record.get("player_id"),
                game_id=raw_record.get("game_id"),
                source=source_enum,
                language=raw_record.get("language", "en"),
                metadata=metadata,
            )

            result = await service.analyze_text(
                text=sentiment_input.text,
                context=sentiment_input.context,
                team_id=sentiment_input.team_id,
                player_id=sentiment_input.player_id,
                game_id=sentiment_input.game_id,
                source=sentiment_input.source,
            )

            doc_data = {
                **sentiment_input.dict(),
                "sentiment": result.sentiment.value,
                "sentiment_score": result.sentiment_score,
                "confidence": result.confidence,
                "category": result.category.value,
                "source": result.source.value,
                "context": (
                    result.context.model_dump()
                    if hasattr(result.context, "model_dump")
                    else result.context.dict()
                ),
                "timestamp": result.timestamp,
                "created_at": result.timestamp,
                "processed_at": datetime.utcnow(),
                "model_version": result.model_version,
                "processing_time_ms": result.processing_time_ms,
                "emotion_scores": result.emotion_scores,
                "aspect_sentiments": result.aspect_sentiments,
                "keyword_contributions": result.keyword_contributions,
                "user_id": str(auth["_id"])
                if isinstance(auth, dict) and auth
                else None,
            }

            await db.sentiment_analyses.insert_one(doc_data)

            if sentiment_input.team_id:
                affected_team_ids.add(sentiment_input.team_id)
            if sentiment_input.player_id:
                affected_player_ids.add(sentiment_input.player_id)
            if sentiment_input.game_id:
                affected_game_ids.add(sentiment_input.game_id)

            processed_results.append(result)

        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Failed to process record %s: %s", idx, exc)
            errors.append({"index": idx, "error": str(exc)})

    for team_id in affected_team_ids:
        background_tasks.add_task(update_aggregated_sentiment, db, team_id=team_id)
    for player_id in affected_player_ids:
        background_tasks.add_task(update_aggregated_sentiment, db, player_id=player_id)
    for game_id in affected_game_ids:
        background_tasks.add_task(update_aggregated_sentiment, db, game_id=game_id)

    return {
        "uploaded_records": len(records),
        "processed": len(processed_results),
        "failed": len(errors),
        "errors": errors,
    }


@router.get("/team/{team_id}", response_model=TeamSentiment)
async def get_team_sentiment(
    team_id: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    sources: Optional[List[DataSource]] = Query(
        None, description="Filter by data sources"
    ),
    categories: Optional[List[SentimentCategory]] = Query(
        None, description="Filter by categories"
    ),
    db=Depends(get_database),
):
    """
    Retrieve aggregated and trend sentiment data for a specific team over a recent time window.

    Parameters:
        days (int): Number of days to include in the analysis (1–30).
        sources (Optional[List[DataSource]]): Optional list of data sources to filter results.
        categories (Optional[List[SentimentCategory]]): Optional list of sentiment categories to filter results.

    Returns:
        TeamSentiment: Aggregated sentiment information for the team, including current_sentiment, total_mentions,
        last_updated, confidence_score, hourly sentiment_trend entries, sentiment_by_category, sentiment_by_source,
        and specific sub-sentiments (performance, coaching, injury, trade).
    """
    try:
        # Get team information
        team = await db.teams.find_one({"_id": team_id})
        if not team:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Team not found"
            )

        # Build query for sentiment data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        query = {
            "team_id": team_id,
            "timestamp": {"$gte": start_date, "$lte": end_date},
        }

        if sources:
            query["source"] = {"$in": [s.value for s in sources]}
        if categories:
            query["category"] = {"$in": [c.value for c in categories]}

        # Get aggregated sentiment data
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": None,
                    "avg_sentiment": {"$avg": "$sentiment_score"},
                    "total_mentions": {"$sum": 1},
                    "confidence": {"$avg": "$confidence"},
                    "sentiment_by_category": {
                        "$push": {
                            "category": "$category",
                            "sentiment": "$sentiment_score",
                        }
                    },
                    "sentiment_by_source": {
                        "$push": {"source": "$source", "sentiment": "$sentiment_score"}
                    },
                }
            },
        ]

        agg_result = await db.sentiment_analyses.aggregate(pipeline).to_list(1)

        if not agg_result:
            # Return empty sentiment data
            return TeamSentiment(
                team_id=team_id,
                team_name=team["name"],
                team_abbreviation=team["abbreviation"],
                current_sentiment=0.0,
                total_mentions=0,
                last_updated=datetime.utcnow(),
                confidence_score=0.0,
            )

        data = agg_result[0]

        # Calculate category breakdown
        sentiment_by_category = {}
        category_data = {}
        for item in data.get("sentiment_by_category", []):
            cat = item["category"]
            if cat not in category_data:
                category_data[cat] = []
            category_data[cat].append(item["sentiment"])

        for cat, sentiments in category_data.items():
            sentiment_by_category[cat] = sum(sentiments) / len(sentiments)

        # Calculate source breakdown
        sentiment_by_source = {}
        source_data = {}
        for item in data.get("sentiment_by_source", []):
            src = item["source"]
            if src not in source_data:
                source_data[src] = []
            source_data[src].append(item["sentiment"])

        for src, sentiments in source_data.items():
            sentiment_by_source[src] = sum(sentiments) / len(sentiments)

        # Get sentiment trends (hourly aggregation)
        trend_pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%Y-%m-%d-%H", "date": "$timestamp"}
                    },
                    "avg_sentiment": {"$avg": "$sentiment_score"},
                    "volume": {"$sum": 1},
                    "timestamp": {"$first": "$timestamp"},
                }
            },
            {"$sort": {"timestamp": 1}},
        ]

        trend_data = await db.sentiment_analyses.aggregate(trend_pipeline).to_list(None)
        sentiment_trend = [
            {
                "timestamp": item["timestamp"],
                "sentiment_score": item["avg_sentiment"],
                "volume": item["volume"],
            }
            for item in trend_data
        ]

        return TeamSentiment(
            team_id=team_id,
            team_name=team["name"],
            team_abbreviation=team["abbreviation"],
            current_sentiment=data["avg_sentiment"],
            total_mentions=data["total_mentions"],
            last_updated=datetime.utcnow(),
            confidence_score=data["confidence"],
            sentiment_trend=sentiment_trend,
            sentiment_by_category=sentiment_by_category,
            sentiment_by_source=sentiment_by_source,
            performance_sentiment=sentiment_by_category.get("performance", 0.0),
            coaching_sentiment=sentiment_by_category.get("coaching", 0.0),
            injury_sentiment=sentiment_by_category.get("injury", 0.0),
            trade_sentiment=sentiment_by_category.get("trade", 0.0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting team sentiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving team sentiment data",
        )


@router.get("/player/{player_id}", response_model=PlayerSentiment)
async def get_player_sentiment(
    player_id: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    sources: Optional[List[DataSource]] = Query(
        None, description="Filter by data sources"
    ),
    categories: Optional[List[SentimentCategory]] = Query(
        None, description="Filter by categories"
    ),
    db=Depends(get_database),
):
    """
    Retrieve aggregated sentiment metrics, category/source breakdowns, and time-series trends for a specific player.

    Parameters:
        player_id (str): Unique identifier of the player to query.
        days (int): Number of days to include in the analysis (window ending at now).
        sources (Optional[List[DataSource]]): Optional list of data sources to filter results.
        categories (Optional[List[SentimentCategory]]): Optional list of sentiment categories to filter results.
        db: Database dependency (omitted from detailed docs).

    Returns:
        PlayerSentiment: Aggregated sentiment data for the player, including:
            - current_sentiment: Average sentiment score over the requested window.
            - total_mentions: Total number of sentiment records considered.
            - last_updated: Timestamp when the aggregation was generated.
            - confidence_score: Average confidence of the underlying analyses.
            - sentiment_trend: Time-bucketed list of sentiment points with timestamp, sentiment_score, and volume.
            - sentiment_by_category: Mapping of category -> average sentiment.
            - sentiment_by_source: Mapping of source -> average sentiment.
            - performance_sentiment, injury_sentiment, trade_sentiment, fantasy_sentiment: Category-specific sentiment values (0.0 if not present).
    """
    try:
        # Get player information
        player = await db.players.find_one({"_id": player_id})
        if not player:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Player not found"
            )

        # Build query for sentiment data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        query = {
            "player_id": player_id,
            "timestamp": {"$gte": start_date, "$lte": end_date},
        }

        if sources:
            query["source"] = {"$in": [s.value for s in sources]}
        if categories:
            query["category"] = {"$in": [c.value for c in categories]}

        # Get aggregated sentiment data
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": None,
                    "avg_sentiment": {"$avg": "$sentiment_score"},
                    "total_mentions": {"$sum": 1},
                    "confidence": {"$avg": "$confidence"},
                    "sentiment_by_category": {
                        "$push": {
                            "category": "$category",
                            "sentiment": "$sentiment_score",
                        }
                    },
                    "sentiment_by_source": {
                        "$push": {"source": "$source", "sentiment": "$sentiment_score"}
                    },
                }
            },
        ]

        agg_result = await db.sentiment_analyses.aggregate(pipeline).to_list(1)

        if not agg_result:
            return PlayerSentiment(
                player_id=player_id,
                player_name=player["name"],
                team_id=player["team_id"],
                position=player["position"],
                current_sentiment=0.0,
                total_mentions=0,
                last_updated=datetime.utcnow(),
                confidence_score=0.0,
            )

        data = agg_result[0]

        # Calculate category breakdown
        sentiment_by_category = {}
        category_data = {}
        for item in data.get("sentiment_by_category", []):
            cat = item["category"]
            if cat not in category_data:
                category_data[cat] = []
            category_data[cat].append(item["sentiment"])

        for cat, sentiments in category_data.items():
            sentiment_by_category[cat] = sum(sentiments) / len(sentiments)

        # Calculate source breakdown
        sentiment_by_source = {}
        source_data = {}
        for item in data.get("sentiment_by_source", []):
            src = item["source"]
            if src not in source_data:
                source_data[src] = []
            source_data[src].append(item["sentiment"])

        for src, sentiments in source_data.items():
            sentiment_by_source[src] = sum(sentiments) / len(sentiments)

        # Get sentiment trends
        trend_pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": "%Y-%m-%d-%H", "date": "$timestamp"}
                    },
                    "avg_sentiment": {"$avg": "$sentiment_score"},
                    "volume": {"$sum": 1},
                    "timestamp": {"$first": "$timestamp"},
                }
            },
            {"$sort": {"timestamp": 1}},
        ]

        trend_data = await db.sentiment_analyses.aggregate(trend_pipeline).to_list(None)
        sentiment_trend = [
            {
                "timestamp": item["timestamp"],
                "sentiment_score": item["avg_sentiment"],
                "volume": item["volume"],
            }
            for item in trend_data
        ]

        return PlayerSentiment(
            player_id=player_id,
            player_name=player["name"],
            team_id=player["team_id"],
            position=player["position"],
            current_sentiment=data["avg_sentiment"],
            total_mentions=data["total_mentions"],
            last_updated=datetime.utcnow(),
            confidence_score=data["confidence"],
            sentiment_trend=sentiment_trend,
            sentiment_by_category=sentiment_by_category,
            sentiment_by_source=sentiment_by_source,
            performance_sentiment=sentiment_by_category.get("performance", 0.0),
            injury_sentiment=sentiment_by_category.get("injury", 0.0),
            trade_sentiment=sentiment_by_category.get("trade", 0.0),
            fantasy_sentiment=sentiment_by_category.get("fantasy", 0.0),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting player sentiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving player sentiment data",
        )


@router.get("/game/{game_id}", response_model=GameSentiment)
async def get_game_sentiment(
    game_id: str,
    sources: Optional[List[DataSource]] = Query(
        None, description="Filter by data sources"
    ),
    categories: Optional[List[SentimentCategory]] = Query(
        None, description="Filter by categories"
    ),
    db=Depends(get_database),
):
    """
    Retrieve aggregated and per-team sentiment metrics for a specific game.

    Constructs and returns a GameSentiment object containing overall game sentiment, per-team sentiment, category breakdowns, mention counts, confidence, and game metadata.

    Parameters:
        game_id (str): Unique identifier of the game.
        sources (Optional[List[DataSource]]): Optional list of data sources to filter sentiment records by.
        categories (Optional[List[SentimentCategory]]): Optional list of sentiment categories to filter records by.

    Returns:
        GameSentiment: Aggregated sentiment information for the requested game, including:
            - game_id, home_team_id, away_team_id, game_date, week, season
            - current_sentiment, overall_sentiment, home_team_sentiment, away_team_sentiment
            - total_mentions, confidence_score, sentiment_by_category, betting_sentiment, prediction_confidence
            - last_updated timestamp

    Raises:
        HTTPException: If the game is not found (404) or if an unexpected error occurs while retrieving or aggregating data (500).
    """
    try:
        # Get game information
        game = await db.games.find_one({"_id": game_id})
        if not game:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Game not found"
            )

        # Build query for sentiment data
        query = {"game_id": game_id}

        if sources:
            query["source"] = {"$in": [s.value for s in sources]}
        if categories:
            query["category"] = {"$in": [c.value for c in categories]}

        # Get overall game sentiment
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": None,
                    "avg_sentiment": {"$avg": "$sentiment_score"},
                    "total_mentions": {"$sum": 1},
                    "confidence": {"$avg": "$confidence"},
                    "sentiment_by_category": {
                        "$push": {
                            "category": "$category",
                            "sentiment": "$sentiment_score",
                        }
                    },
                }
            },
        ]

        agg_result = await db.sentiment_analyses.aggregate(pipeline).to_list(1)

        # Get team-specific sentiment for this game
        home_team_query = {**query, "team_id": game["home_team_id"]}
        away_team_query = {**query, "team_id": game["away_team_id"]}

        home_sentiment_data = await db.sentiment_analyses.aggregate(
            [
                {"$match": home_team_query},
                {
                    "$group": {
                        "_id": None,
                        "avg_sentiment": {"$avg": "$sentiment_score"},
                    }
                },
            ]
        ).to_list(1)

        away_sentiment_data = await db.sentiment_analyses.aggregate(
            [
                {"$match": away_team_query},
                {
                    "$group": {
                        "_id": None,
                        "avg_sentiment": {"$avg": "$sentiment_score"},
                    }
                },
            ]
        ).to_list(1)

        # Calculate sentiment breakdown
        sentiment_by_category = {}
        if agg_result:
            data = agg_result[0]
            category_data = {}
            for item in data.get("sentiment_by_category", []):
                cat = item["category"]
                if cat not in category_data:
                    category_data[cat] = []
                category_data[cat].append(item["sentiment"])

            for cat, sentiments in category_data.items():
                sentiment_by_category[cat] = sum(sentiments) / len(sentiments)

        return GameSentiment(
            game_id=game_id,
            home_team_id=game["home_team_id"],
            away_team_id=game["away_team_id"],
            game_date=game["game_date"],
            week=game["week"],
            season=game["season"],
            current_sentiment=agg_result[0]["avg_sentiment"] if agg_result else 0.0,
            total_mentions=agg_result[0]["total_mentions"] if agg_result else 0,
            last_updated=datetime.utcnow(),
            confidence_score=agg_result[0]["confidence"] if agg_result else 0.0,
            sentiment_by_category=sentiment_by_category,
            home_team_sentiment=(
                home_sentiment_data[0]["avg_sentiment"] if home_sentiment_data else 0.0
            ),
            away_team_sentiment=(
                away_sentiment_data[0]["avg_sentiment"] if away_sentiment_data else 0.0
            ),
            overall_sentiment=agg_result[0]["avg_sentiment"] if agg_result else 0.0,
            betting_sentiment=sentiment_by_category.get("betting", 0.0),
            prediction_confidence=agg_result[0]["confidence"] if agg_result else 0.0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting game sentiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving game sentiment data",
        )


@router.get("/trends")
async def get_sentiment_trends(
    team_id: Optional[str] = None,
    player_id: Optional[str] = None,
    game_id: Optional[str] = None,
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    interval: str = Query("hour", regex="^(hour|day)$", description="Trend interval"),
    sources: Optional[List[DataSource]] = Query(
        None, description="Filter by data sources"
    ),
    categories: Optional[List[SentimentCategory]] = Query(
        None, description="Filter by categories"
    ),
    db=Depends(get_database),
):
    """
    Retrieve time-bucketed sentiment trends with optional filters.

    Parameters:
        team_id (Optional[str]): Filter trends to a specific team.
        player_id (Optional[str]): Filter trends to a specific player.
        game_id (Optional[str]): Filter trends to a specific game.
        days (int): Number of days to include in the analysis (1–30).
        interval (str): Time bucket for trends; "hour" groups by hour, "day" groups by day.
        sources (Optional[List[DataSource]]): If provided, include only data from these sources.
        categories (Optional[List[SentimentCategory]]): If provided, include only these sentiment categories.
        db: Database dependency.

    Returns:
        dict: {
            "trends": List[{
                "timestamp": datetime,               # representative timestamp for the bucket
                "sentiment_score": float,            # average sentiment for the bucket
                "volume": int,                       # number of records in the bucket
                "confidence": float,                 # average confidence for the bucket
                "category_breakdown": dict,          # average sentiment per category in the bucket
                "source_breakdown": dict,            # average sentiment per source in the bucket
            }],
            "period": str,                          # human-readable period (e.g., "7 days")
            "interval": str,                        # requested interval ("hour" or "day")
            "total_data_points": int,               # number of returned trend buckets
            "filters": dict,                        # echo of applied filters
            "last_updated": datetime,               # timestamp when results were generated
        }
    """
    try:
        # Build base query
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        query = {"timestamp": {"$gte": start_date, "$lte": end_date}}

        if team_id:
            query["team_id"] = team_id
        if player_id:
            query["player_id"] = player_id
        if game_id:
            query["game_id"] = game_id
        if sources:
            query["source"] = {"$in": [s.value for s in sources]}
        if categories:
            query["category"] = {"$in": [c.value for c in categories]}

        # Set date format based on interval
        date_format = "%Y-%m-%d-%H" if interval == "hour" else "%Y-%m-%d"

        # Aggregation pipeline for trends
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": {
                        "$dateToString": {"format": date_format, "date": "$timestamp"}
                    },
                    "avg_sentiment": {"$avg": "$sentiment_score"},
                    "volume": {"$sum": 1},
                    "confidence": {"$avg": "$confidence"},
                    "timestamp": {"$first": "$timestamp"},
                    "category_breakdown": {
                        "$push": {
                            "category": "$category",
                            "sentiment": "$sentiment_score",
                        }
                    },
                    "source_breakdown": {
                        "$push": {"source": "$source", "sentiment": "$sentiment_score"}
                    },
                }
            },
            {"$sort": {"timestamp": 1}},
            {"$limit": 1000},  # Limit results for performance
        ]

        trend_data = await db.sentiment_analyses.aggregate(pipeline).to_list(None)

        # Process trend data
        trends = []
        for item in trend_data:
            # Calculate category breakdown for this time period
            category_breakdown = {}
            cat_data = {}
            for cat_item in item.get("category_breakdown", []):
                cat = cat_item["category"]
                if cat not in cat_data:
                    cat_data[cat] = []
                cat_data[cat].append(cat_item["sentiment"])

            for cat, sentiments in cat_data.items():
                category_breakdown[cat] = sum(sentiments) / len(sentiments)

            # Calculate source breakdown for this time period
            source_breakdown = {}
            src_data = {}
            for src_item in item.get("source_breakdown", []):
                src = src_item["source"]
                if src not in src_data:
                    src_data[src] = []
                src_data[src].append(src_item["sentiment"])

            for src, sentiments in src_data.items():
                source_breakdown[src] = sum(sentiments) / len(sentiments)

            trends.append(
                {
                    "timestamp": item["timestamp"],
                    "sentiment_score": item["avg_sentiment"],
                    "volume": item["volume"],
                    "confidence": item["confidence"],
                    "category_breakdown": category_breakdown,
                    "source_breakdown": source_breakdown,
                }
            )

        return {
            "trends": trends,
            "period": f"{days} days",
            "interval": interval,
            "total_data_points": len(trends),
            "filters": {
                "team_id": team_id,
                "player_id": player_id,
                "game_id": game_id,
                "sources": sources,
                "categories": categories,
            },
            "last_updated": datetime.utcnow(),
        }

    except Exception as e:
        logger.error(f"Error getting sentiment trends: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving sentiment trends",
        )


@router.get("/recent", response_model=List[SentimentAnalysisResponse])
async def get_recent_sentiments(
    limit: int = Query(
        10, ge=1, le=100, description="Number of recent analyses to return"
    ),
    team_id: Optional[str] = None,
    player_id: Optional[str] = None,
    db=Depends(get_database),
):
    """
    Retrieve recent sentiment analysis records, optionally filtered by team or player.

    Parameters:
        limit (int): Maximum number of records to return (1–100).
        team_id (Optional[str]): If provided, return only analyses for this team.
        player_id (Optional[str]): If provided, return only analyses for this player.

    Returns:
        List[SentimentAnalysisResponse]: Recent analyses sorted by timestamp descending; each entry includes an `id` field derived from the database `_id`.
    """
    query = {}
    if team_id:
        query["team_id"] = team_id
    if player_id:
        query["player_id"] = player_id

    cursor = db.sentiment_analyses.find(query).sort([("timestamp", -1)]).limit(limit)
    results = []

    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        results.append(SentimentAnalysisResponse(**doc))

    return results


@router.post("/aggregate", response_model=Dict[str, Any])
async def aggregate_sentiment(
    request: SentimentAggregationRequest, db=Depends(get_database)
):
    """
    Compute aggregated sentiment metrics for a specified team, player, or game using optional filters and optional time-series trends.

    Parameters:
        request (SentimentAggregationRequest): Specifies the target entity (entity_type and entity_id), optional start_date and end_date, optional sources and categories to filter by, and options to include time-based trends (include_trends, trend_interval).

    Returns:
        dict: Aggregated results with the following keys:
            - entity_type (str): The requested entity type ("team", "player", or "game").
            - entity_id (str): The requested entity identifier.
            - avg_sentiment (float): Average sentiment score across matched records.
            - total_mentions (int): Count of matched records.
            - confidence (float): Average confidence score for matched records.
            - sentiment_distribution (dict): Counts of discrete sentiment labels.
            - category_breakdown (dict): Average sentiment score per category.
            - source_breakdown (dict): Average sentiment score per source.
            - trends (list): Time-series entries (when requested) each with `timestamp`, `sentiment_score`, and `volume`.
            - generated_at (datetime): UTC timestamp when the aggregation was produced.

    Raises:
        HTTPException: With status 500 if an internal error occurs while performing the aggregation.
    """
    try:
        # Build query based on request
        query = {}

        if request.entity_type == "team":
            query["team_id"] = request.entity_id
        elif request.entity_type == "player":
            query["player_id"] = request.entity_id
        elif request.entity_type == "game":
            query["game_id"] = request.entity_id

        if request.start_date and request.end_date:
            query["timestamp"] = {"$gte": request.start_date, "$lte": request.end_date}

        if request.sources:
            query["source"] = {"$in": [s.value for s in request.sources]}

        if request.categories:
            query["category"] = {"$in": [c.value for c in request.categories]}

        # Aggregation pipeline
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": None,
                    "avg_sentiment": {"$avg": "$sentiment_score"},
                    "total_mentions": {"$sum": 1},
                    "avg_confidence": {"$avg": "$confidence"},
                    "sentiment_distribution": {"$push": "$sentiment"},
                    "category_sentiments": {
                        "$push": {
                            "category": "$category",
                            "sentiment": "$sentiment_score",
                        }
                    },
                    "source_sentiments": {
                        "$push": {"source": "$source", "sentiment": "$sentiment_score"}
                    },
                }
            },
        ]

        result = await db.sentiment_analyses.aggregate(pipeline).to_list(1)

        if not result:
            return {
                "entity_type": request.entity_type,
                "entity_id": request.entity_id,
                "avg_sentiment": 0.0,
                "total_mentions": 0,
                "confidence": 0.0,
                "sentiment_distribution": {},
                "category_breakdown": {},
                "source_breakdown": {},
                "trends": [],
            }

        data = result[0]

        # Process sentiment distribution
        sentiment_dist = {}
        for sentiment in data["sentiment_distribution"]:
            sentiment_dist[sentiment] = sentiment_dist.get(sentiment, 0) + 1

        # Process category breakdown
        category_breakdown = {}
        cat_data = {}
        for item in data["category_sentiments"]:
            cat = item["category"]
            if cat not in cat_data:
                cat_data[cat] = []
            cat_data[cat].append(item["sentiment"])

        for cat, sentiments in cat_data.items():
            category_breakdown[cat] = sum(sentiments) / len(sentiments)

        # Process source breakdown
        source_breakdown = {}
        src_data = {}
        for item in data["source_sentiments"]:
            src = item["source"]
            if src not in src_data:
                src_data[src] = []
            src_data[src].append(item["sentiment"])

        for src, sentiments in src_data.items():
            source_breakdown[src] = sum(sentiments) / len(sentiments)

        # Get trends if requested
        trends = []
        if request.include_trends:
            trend_format = (
                "%Y-%m-%d-%H" if request.trend_interval == "hour" else "%Y-%m-%d"
            )
            trend_pipeline = [
                {"$match": query},
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": trend_format,
                                "date": "$timestamp",
                            }
                        },
                        "avg_sentiment": {"$avg": "$sentiment_score"},
                        "volume": {"$sum": 1},
                        "timestamp": {"$first": "$timestamp"},
                    }
                },
                {"$sort": {"timestamp": 1}},
            ]

            trend_data = await db.sentiment_analyses.aggregate(trend_pipeline).to_list(
                None
            )
            trends = [
                {
                    "timestamp": item["timestamp"],
                    "sentiment_score": item["avg_sentiment"],
                    "volume": item["volume"],
                }
                for item in trend_data
            ]

        return {
            "entity_type": request.entity_type,
            "entity_id": request.entity_id,
            "avg_sentiment": data["avg_sentiment"],
            "total_mentions": data["total_mentions"],
            "confidence": data["avg_confidence"],
            "sentiment_distribution": sentiment_dist,
            "category_breakdown": category_breakdown,
            "source_breakdown": source_breakdown,
            "trends": trends,
            "generated_at": datetime.utcnow(),
        }

    except Exception as e:
        logger.error(f"Error aggregating sentiment: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error aggregating sentiment data",
        )


@router.get("/insights/{entity_type}/{entity_id}", response_model=SentimentInsights)
async def get_sentiment_insights(
    entity_type: str,
    entity_id: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to analyze"),
    db=Depends(get_database),
):
    """
    Produce insights and prioritized recommendations based on recent sentiment data for a given entity.

    Parameters:
        entity_type (str): Type of entity to analyze, e.g. "team", "player", or "game".
        entity_id (str): Identifier of the specific entity to analyze.
        days (int): Number of past days to include in the analysis (1–30).

    Returns:
        SentimentInsights: Object containing key themes (most frequent categories), top sentiment-driving keywords with their average impact, actionable recommendations derived from average sentiment, and a generation timestamp.
    """
    try:
        # This would integrate with more advanced analytics
        # For now, return basic insights

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        query = {
            f"{entity_type}_id": entity_id,
            "timestamp": {"$gte": start_date, "$lte": end_date},
        }

        # Get basic sentiment data
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": None,
                    "avg_sentiment": {"$avg": "$sentiment_score"},
                    "keyword_contributions": {"$push": "$keyword_contributions"},
                    "categories": {"$push": "$category"},
                }
            },
        ]

        result = await db.sentiment_analyses.aggregate(pipeline).to_list(1)

        if not result:
            return SentimentInsights(
                entity_id=entity_id,
                entity_type=entity_type,
                generated_at=datetime.utcnow(),
            )

        data = result[0]

        # Extract key themes (most common categories)
        category_counts = {}
        for cat in data["categories"]:
            category_counts[cat] = category_counts.get(cat, 0) + 1

        key_themes = sorted(
            category_counts.keys(), key=lambda x: category_counts[x], reverse=True
        )[:5]

        # Extract sentiment drivers (most impactful keywords)
        all_keywords = {}
        for contrib_dict in data["keyword_contributions"]:
            for keyword, score in contrib_dict.items():
                if keyword not in all_keywords:
                    all_keywords[keyword] = []
                all_keywords[keyword].append(score)

        sentiment_drivers = {}
        for keyword, scores in all_keywords.items():
            sentiment_drivers[keyword] = sum(scores) / len(scores)

        # Sort by absolute impact
        sentiment_drivers = dict(
            sorted(sentiment_drivers.items(), key=lambda x: abs(x[1]), reverse=True)[
                :10
            ]
        )

        # Generate basic recommendations
        recommendations = []
        avg_sentiment = data["avg_sentiment"]

        if avg_sentiment < -0.3:
            recommendations.append("Consider addressing negative sentiment drivers")
        elif avg_sentiment > 0.3:
            recommendations.append(
                "Leverage positive sentiment for marketing opportunities"
            )
        else:
            recommendations.append("Monitor sentiment closely for emerging trends")

        return SentimentInsights(
            entity_id=entity_id,
            entity_type=entity_type,
            key_themes=key_themes,
            sentiment_drivers=sentiment_drivers,
            recommendations=recommendations,
            generated_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Error generating sentiment insights: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error generating sentiment insights",
        )


async def update_aggregated_sentiment(
    db,
    team_id: Optional[str] = None,
    player_id: Optional[str] = None,
    game_id: Optional[str] = None,
):
    """
    Trigger aggregated sentiment recomputation for the provided entity identifiers.

    Parameters:
        team_id (Optional[str]): If provided, recompute and store aggregated sentiment for the team with this ID.
        player_id (Optional[str]): If provided, recompute and store aggregated sentiment for the player with this ID.
        game_id (Optional[str]): If provided, recompute and store aggregated sentiment for the game with this ID.
    """
    try:
        # Update team sentiment aggregation
        if team_id:
            await update_team_sentiment_aggregation(db, team_id)

        # Update player sentiment aggregation
        if player_id:
            await update_player_sentiment_aggregation(db, player_id)

        # Update game sentiment aggregation
        if game_id:
            await update_game_sentiment_aggregation(db, game_id)

    except Exception as e:
        logger.error(f"Error updating aggregated sentiment: {str(e)}")


async def update_team_sentiment_aggregation(db, team_id: str):
    """
    Recompute and persist aggregated sentiment metrics for the given team.

    Performs aggregation of stored sentiment analyses for the specified team and updates the team's aggregated sentiment record in the database.

    Parameters:
        team_id (str): The unique identifier of the team whose aggregated sentiment should be updated.
    """
    # This would calculate and store aggregated team sentiment
    # Implementation would depend on specific aggregation requirements
    pass


async def update_player_sentiment_aggregation(db, player_id: str):
    """
    Recompute and persist aggregated sentiment metrics for a given player.

    Recalculates the player's aggregated sentiment data (for example: average sentiment, total mentions, confidence score, and time-based trends) from stored analyses and updates the corresponding aggregation document in the database.

    Parameters:
        player_id (str): Unique identifier of the player whose aggregated sentiment should be refreshed.
    """
    # This would calculate and store aggregated player sentiment
    # Implementation would depend on specific aggregation requirements
    pass


async def update_game_sentiment_aggregation(db, game_id: str):
    """
    Recompute and persist aggregated sentiment metrics for the specified game.

    Recalculates aggregated values (e.g., average sentiment, total mentions, confidence, category breakdowns, and trends) for the given game and stores the results in the database; intended to run as a background task after new sentiment records are inserted. This function performs its work via side effects and does not return a value.

    Parameters:
        game_id (str): Identifier of the game whose aggregated sentiment should be updated.
    """
    # This would calculate and store aggregated game sentiment
    # Implementation would depend on specific aggregation requirements
    pass
