from fastapi import APIRouter, Depends, HTTPException, status, Query
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel

from ..core.database import get_database
from ..models.nfl import Team, Player, Game, GameResponse, BettingLine
from ..services.data_processing_pipeline import data_processing_pipeline


class TeamOut(Team):
    """Extended Team model with optional dynamic fields for API responses"""
    current_stats: Optional[Dict[str, Any]] = None
    current_sentiment: Optional[Dict[str, Any]] = None


class PlayerOut(Player):
    """Extended Player model with optional dynamic fields for API responses"""
    current_stats: Optional[Dict[str, Any]] = None
    current_sentiment: Optional[Dict[str, Any]] = None


class GameResponseOut(GameResponse):
    """Extended GameResponse model with optional dynamic fields for API responses"""
    predictions: Optional[List[Dict[str, Any]]] = None
    sentiment_summary: Optional[Dict[str, Any]] = None

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["data"])


@router.get("/teams", response_model=List[TeamOut])
async def get_teams(
    conference: Optional[str] = None,
    division: Optional[str] = None,
    include_stats: bool = Query(False, description="Include current season stats"),
    include_sentiment: bool = Query(
        False, description="Include current sentiment data"
    ),
    db=Depends(get_database),
):
    """
    Retrieve NFL teams matching optional conference and division filters.

    Filters:
    - `conference`: matched after converting to uppercase.
    - `division`: matched after converting to title case.

    Parameters:
    - include_stats: When true, each team will include a `current_stats` field with the team's current-season aggregated statistics.
    - include_sentiment: When true, each team will include a `current_sentiment` field summarizing recent sentiment.

    Returns:
    A list of Team models; each item may include `current_stats` and/or `current_sentiment` when those options are enabled.
    """
    query = {}
    if conference:
        query["conference"] = conference.upper()
    if division:
        query["division"] = division.title()

    cursor = db.teams.find(query)
    teams = []

    async for doc in cursor:
        doc["id"] = str(doc["_id"])

        # Add current season stats if requested
        if include_stats:
            stats = await _get_team_stats(db, doc["id"])
            doc["current_stats"] = stats

        # Add sentiment data if requested
        if include_sentiment:
            sentiment = await _get_team_sentiment(db, doc["id"])
            doc["current_sentiment"] = sentiment

        teams.append(TeamOut(**doc))

    return teams


async def _get_team_stats(db, team_id: str) -> Dict[str, Any]:
    """
    Compute aggregate statistics for a team's current-season completed games.

    Parameters:
        team_id (str): Team document `_id` to compute stats for.

    Returns:
        dict: Aggregated statistics with keys:
            - `wins`: Number of games the team won this season.
            - `losses`: Number of games the team lost this season.
            - `points_for`: Total points scored by the team.
            - `points_against`: Total points scored against the team.
            - `point_differential`: `points_for - points_against`.
    """
    current_season = datetime.now().year

    # Get games for current season
    games_cursor = db.games.find(
        {
            "$or": [{"home_team_id": team_id}, {"away_team_id": team_id}],
            "season": current_season,
            "status": "completed",
        }
    )

    wins = 0
    losses = 0
    points_for = 0
    points_against = 0

    async for game in games_cursor:
        home_score = game.get("home_score", 0) or 0
        away_score = game.get("away_score", 0) or 0

        if game["home_team_id"] == team_id:
            points_for += home_score
            points_against += away_score
            if home_score > away_score:
                wins += 1
            else:
                losses += 1
        else:
            points_for += away_score
            points_against += home_score
            if away_score > home_score:
                wins += 1
            else:
                losses += 1

    return {
        "wins": wins,
        "losses": losses,
        "points_for": points_for,
        "points_against": points_against,
        "point_differential": points_for - points_against,
    }


async def _get_team_sentiment(db, team_id: str) -> Dict[str, Any]:
    """
    Compute recent sentiment metrics for a team based on sentiment analyses from the last 7 days.

    Parameters:
        team_id (str): The team's document `_id` in the database.

    Returns:
        dict: {
            "average_sentiment": float,  # average sentiment score rounded to 3 decimals; +1=positive, -1=negative, 0=neutral
            "total_mentions": int,       # number of sentiment analyses matched
            "period": str                # descriptor of the aggregation window (e.g., "last_7_days")
        }
        Returns defaults (average_sentiment 0.0 and total_mentions 0) if the team is not found.
    """
    # Get team info for keyword matching
    team = await db.teams.find_one({"_id": team_id})
    if not team:
        return {"average_sentiment": 0.0, "total_mentions": 0}

    # Get recent sentiment analyses (last 7 days)
    week_ago = datetime.now() - timedelta(days=7)

    # Search for sentiment analyses mentioning this team
    team_keywords = [team["name"].lower(), team["abbreviation"].lower()] + team.get(
        "aliases", []
    )

    sentiment_cursor = db.sentiment_analyses.find(
        {
            "created_at": {"$gte": week_ago},
            "$or": [
                {"text": {"$regex": keyword, "$options": "i"}}
                for keyword in team_keywords
            ],
        }
    )

    sentiments = []
    async for sentiment in sentiment_cursor:
        if sentiment.get("sentiment") == "positive":
            sentiments.append(1.0)
        elif sentiment.get("sentiment") == "negative":
            sentiments.append(-1.0)
        else:
            sentiments.append(0.0)

    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
    else:
        avg_sentiment = 0.0

    return {
        "average_sentiment": round(avg_sentiment, 3),
        "total_mentions": len(sentiments),
        "period": "last_7_days",
    }


@router.get("/teams/{team_id}", response_model=Team)
async def get_team(team_id: str, db=Depends(get_database)):
    """
    Retrieve a team by its unique identifier and return it as a Team model.

    Parameters:
        team_id (str): The team's unique identifier.

    Returns:
        Team: The team model constructed from the database document.

    Raises:
        HTTPException: Raised with status 404 if no team with the given ID exists.
    """
    team = await db.teams.find_one({"_id": team_id})
    if not team:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Team not found"
        )

    team["id"] = str(team["_id"])
    return Team(**team)


@router.get("/players", response_model=List[PlayerOut])
async def get_players(
    team_id: Optional[str] = None,
    position: Optional[str] = None,
    status: Optional[str] = Query(
        None, description="Player status: active, injured, suspended"
    ),
    fantasy_relevant: Optional[bool] = Query(
        None, description="Filter by fantasy relevance"
    ),
    include_stats: bool = Query(False, description="Include current season stats"),
    include_sentiment: bool = Query(
        False, description="Include current sentiment data"
    ),
    limit: int = Query(50, ge=1, le=200, description="Number of players to return"),
    db=Depends(get_database),
):
    """
    Retrieve a list of NFL players matching optional filters, optionally enriched with current-season stats and recent sentiment.

    Parameters:
        team_id (Optional[str]): Filter players by team ID.
        position (Optional[str]): Filter by position (case-insensitive; stored as uppercase).
        status (Optional[str]): Filter by player status (e.g., "active", "injured", "suspended").
        fantasy_relevant (Optional[bool]): Filter players flagged as relevant for fantasy.
        include_stats (bool): If true, include `current_stats` for each player.
        include_sentiment (bool): If true, include `current_sentiment` for each player.
        limit (int): Maximum number of players to return (1–200).

    Returns:
        List[Player]: Players matching the query. Each Player may include `current_stats` and/or `current_sentiment` when requested.
    """
    query = {}
    if team_id:
        query["team_id"] = team_id
    if position:
        query["position"] = position.upper()
    if status:
        query["status"] = status.lower()
    if fantasy_relevant is not None:
        query["fantasy_relevant"] = fantasy_relevant

    cursor = db.players.find(query).limit(limit)
    players = []

    async for doc in cursor:
        doc["id"] = str(doc["_id"])

        # Add current season stats if requested
        if include_stats:
            stats = await _get_player_stats(db, doc["id"])
            doc["current_stats"] = stats

        # Add sentiment data if requested
        if include_sentiment:
            sentiment = await _get_player_sentiment(db, doc["id"])
            doc["current_sentiment"] = sentiment

        players.append(PlayerOut(**doc))

    return players


async def _get_player_stats(db, player_id: str) -> Dict[str, Any]:
    """
    Return placeholder current-season statistics for a player.

    This provides a temporary structure until a real stats source is integrated.

    Returns:
        dict: A mapping containing:
            - games_played (int): Number of games played this season.
            - stats (dict): Per-category statistics (empty for placeholder).
            - last_updated (str): ISO 8601 timestamp indicating when the stats were generated.
    """
    # This would typically come from a stats API or database
    # For now, return a placeholder structure
    return {"games_played": 0, "stats": {}, "last_updated": datetime.now().isoformat()}


async def _get_player_sentiment(db, player_id: str) -> Dict[str, Any]:
    """
    Compute recent sentiment metrics for a player based on sentiment analyses from the last 7 days.

    Returns:
        result (Dict[str, Any]): A dictionary containing:
            - average_sentiment (float): Average sentiment mapped to [-1.0, 1.0] and rounded to 3 decimals.
            - total_mentions (int): Number of sentiment analysis entries that mentioned the player.
            - period (str): Label of the time window used (e.g., "last_7_days").
    """
    # Get player info for keyword matching
    player = await db.players.find_one({"_id": player_id})
    if not player:
        return {"average_sentiment": 0.0, "total_mentions": 0}

    # Get recent sentiment analyses (last 7 days)
    week_ago = datetime.now() - timedelta(days=7)

    # Search for sentiment analyses mentioning this player
    player_keywords = [player["name"].lower()] + player.get("aliases", [])

    sentiment_cursor = db.sentiment_analyses.find(
        {
            "created_at": {"$gte": week_ago},
            "$or": [
                {"text": {"$regex": keyword, "$options": "i"}}
                for keyword in player_keywords
            ],
        }
    )

    sentiments = []
    async for sentiment in sentiment_cursor:
        if sentiment.get("sentiment") == "positive":
            sentiments.append(1.0)
        elif sentiment.get("sentiment") == "negative":
            sentiments.append(-1.0)
        else:
            sentiments.append(0.0)

    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
    else:
        avg_sentiment = 0.0

    return {
        "average_sentiment": round(avg_sentiment, 3),
        "total_mentions": len(sentiments),
        "period": "last_7_days",
    }


@router.get("/players/{player_id}", response_model=PlayerOut)
async def get_player(player_id: str, db=Depends(get_database)):
    """
    Retrieve a single player by its ID.

    Parameters:
        player_id (str): The player's identifier.

    Returns:
        Player: The Player model constructed from the database document.

    Raises:
        HTTPException: 404 if no player with the given ID exists.
    """
    player = await db.players.find_one({"_id": player_id})
    if not player:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Player not found"
        )

    player["id"] = str(player["_id"])
    return PlayerOut(**player)


@router.get("/games", response_model=List[GameResponseOut])
async def get_games(
    week: Optional[int] = None,
    season: Optional[int] = None,
    team_id: Optional[str] = None,
    status: Optional[str] = None,
    game_type: Optional[str] = Query(
        None, description="Game type: regular, playoff, championship"
    ),
    primetime: Optional[bool] = Query(None, description="Filter primetime games"),
    include_predictions: bool = Query(False, description="Include game predictions"),
    include_sentiment: bool = Query(False, description="Include game sentiment data"),
    limit: int = Query(20, ge=1, le=100, description="Number of games to return"),
    db=Depends(get_database),
):
    """
    Retrieve games matching the provided filters, optionally enriched with team data, predictions, and sentiment.

    Parameters:
        week (Optional[int]): Filter by NFL week.
        season (Optional[int]): Filter by season year.
        team_id (Optional[str]): Return games where the team is home or away.
        status (Optional[str]): Game status filter (e.g., "scheduled", "completed").
        game_type (Optional[str]): Game type filter; expected values include "regular", "playoff", "championship".
        primetime (Optional[bool]): If provided, filter for primetime (True) or non-primetime (False) games.
        include_predictions (bool): If True, include model predictions for each game.
        include_sentiment (bool): If True, include a sentiment summary for each game.
        limit (int): Maximum number of games to return (1-100).
        db: Database dependency (omitted from detailed docs).

    Returns:
        List[GameResponse]: A list of game response objects matching the filters. Each item may include embedded `home_team` and `away_team` data; when requested, `predictions` and `sentiment_summary` fields are added.
    """
    query = {}
    if week:
        query["week"] = week
    if season:
        query["season"] = season
    if team_id:
        query["$or"] = [{"home_team_id": team_id}, {"away_team_id": team_id}]
    if status:
        query["status"] = status.lower()
    if game_type:
        query["game_type"] = game_type.lower()
    if primetime is not None:
        query["primetime"] = primetime

    cursor = db.games.find(query).sort([("game_date", -1)]).limit(limit)
    games = []

    async for doc in cursor:
        doc["id"] = str(doc["_id"])

        # Get team information
        home_team = await db.teams.find_one({"_id": doc["home_team_id"]})
        away_team = await db.teams.find_one({"_id": doc["away_team_id"]})

        if home_team:
            home_team["id"] = str(home_team["_id"])
            doc["home_team"] = Team(**home_team)

        if away_team:
            away_team["id"] = str(away_team["_id"])
            doc["away_team"] = Team(**away_team)

        # Add predictions if requested
        if include_predictions:
            predictions = await _get_game_predictions(db, doc["id"])
            doc["predictions"] = predictions

        # Add sentiment data if requested
        if include_sentiment:
            sentiment = await _get_game_sentiment(db, doc["id"])
            doc["sentiment_summary"] = sentiment

        games.append(GameResponseOut(**doc))

    return games


async def _get_game_predictions(db, game_id: str) -> List[Dict[str, Any]]:
    """
    Retrieve prediction documents for a given game, ordered from newest to oldest.

    Each returned dictionary represents a prediction document and includes an `id` field containing the stringified original `_id`.

    Returns:
        List[Dict[str, Any]]: Prediction documents for the specified game with `id` added.
    """
    cursor = db.predictions.find({"game_id": game_id}).sort([("created_at", -1)])
    predictions = []

    async for pred in cursor:
        pred["id"] = str(pred.get("_id"))
        pred.pop("_id", None)
        pred["id"] = pred_id
        predictions.append(pred)

    return predictions


async def _get_game_sentiment(db, game_id: str) -> Dict[str, Any]:
    """
    Compute a 7-day sentiment summary for the specified game.

    Returns:
        dict: Summary containing:
            - overall_sentiment (float): Average sentiment score for mentions of either team over the last 7 days, rounded to 3 decimals.
            - total_mentions (int): Number of sentiment mentions considered.
            - home_team_sentiment (float): Placeholder sentiment score for the home team (0.0 if not computed separately).
            - away_team_sentiment (float): Placeholder sentiment score for the away team (0.0 if not computed separately).
            - period (str): Time window label (e.g., "last_7_days").
    """
    # Get game info
    game = await db.games.find_one({"_id": game_id})
    if not game:
        return {"overall_sentiment": 0.0, "total_mentions": 0}

    # Get team names for keyword matching
    home_team = await db.teams.find_one({"_id": game["home_team_id"]})
    away_team = await db.teams.find_one({"_id": game["away_team_id"]})

    if not home_team or not away_team:
        return {"overall_sentiment": 0.0, "total_mentions": 0}

    # Get recent sentiment analyses mentioning either team
    week_ago = datetime.now() - timedelta(days=7)

    team_keywords = [
        home_team["name"].lower(),
        home_team["abbreviation"].lower(),
        away_team["name"].lower(),
        away_team["abbreviation"].lower(),
    ]

    sentiment_cursor = db.sentiment_analyses.find(
        {
            "created_at": {"$gte": week_ago},
            "$or": [
                {"text": {"$regex": keyword, "$options": "i"}}
                for keyword in team_keywords
            ],
        }
    )

    sentiments = []
    async for sentiment in sentiment_cursor:
        if sentiment.get("sentiment") == "positive":
            sentiments.append(1.0)
        elif sentiment.get("sentiment") == "negative":
            sentiments.append(-1.0)
        else:
            sentiments.append(0.0)

    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
    else:
        avg_sentiment = 0.0

    return {
        "overall_sentiment": round(avg_sentiment, 3),
        "total_mentions": len(sentiments),
        "home_team_sentiment": 0.0,  # Could be calculated separately
        "away_team_sentiment": 0.0,  # Could be calculated separately
        "period": "last_7_days",
    }


@router.get("/games/{game_id}", response_model=GameResponseOut)
async def get_game(game_id: str, db=Depends(get_database)):
    """
    Retrieve a single game by its ID and include embedded home and away team models when available.

    Returns:
        GameResponse: The game data with `home_team` and `away_team` fields populated as `Team` models when those teams exist.

    Raises:
        HTTPException: Raised with status 404 if no game with the given `game_id` is found.
    """
    game = await db.games.find_one({"_id": game_id})
    if not game:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Game not found"
        )

    game["id"] = str(game["_id"])

    # Get team information
    home_team = await db.teams.find_one({"_id": game["home_team_id"]})
    away_team = await db.teams.find_one({"_id": game["away_team_id"]})

    if home_team:
        home_team["id"] = str(home_team["_id"])
        game["home_team"] = Team(**home_team)

    if away_team:
        away_team["id"] = str(away_team["_id"])
        game["away_team"] = Team(**away_team)

    return GameResponseOut(**game)


@router.get("/betting-lines")
async def get_betting_lines(
    game_id: Optional[str] = None,
    sportsbook: Optional[str] = None,
    week: Optional[int] = None,
    season: Optional[int] = None,
    include_history: bool = Query(
        False, description="Include historical line movements"
    ),
    include_team_info: bool = Query(True, description="Include team information"),
    db=Depends(get_database),
):
    """
    Retrieve current betting lines with optional historical movements and team information.

    Parameters:
        game_id (str | None): Filter lines for a specific game by its ID.
        sportsbook (str | None): Filter lines by sportsbook identifier.
        week (int | None): Filter games by week number.
        season (int | None): Filter games by season year.
        include_history (bool): If true and `game_id` is provided, include historical line movements.
        include_team_info (bool): If true, attach basic home/away team info (name, abbreviation, city).
        db: Database dependency (omitted from parameter docs as a provided dependency).

    Returns:
        dict: If `include_history` and `game_id` are provided, returns:
            {
                "current_lines": [ ... ],        # list of current line objects with game metadata and optional team info
                "historical_lines": [ ... ],     # list of historical line movement objects
                "total_current": int             # count of current_lines
            }
        Otherwise returns:
            {
                "betting_lines": [ ... ],        # list of current line objects with game metadata and optional team info
                "raw_betting_data": [ ... ],     # recent raw betting line records from ingestion
                "total": int,                    # count of betting_lines
                "last_updated": str              # ISO8601 timestamp when response was generated
            }
    """
    query = {}
    if game_id:
        query["_id"] = game_id
    if week:
        query["week"] = week
    if season:
        query["season"] = season

    # Get games with betting lines
    cursor = db.games.find(
        query,
        {
            "betting_lines": 1,
            "home_team_id": 1,
            "away_team_id": 1,
            "game_date": 1,
            "week": 1,
            "season": 1,
            "status": 1,
        },
    )

    betting_data = []

    async for game in cursor:
        game_betting_lines = []

        for line in game.get("betting_lines", []):
            if sportsbook and line.get("sportsbook") != sportsbook:
                continue

            line_data = {
                "game_id": str(game["_id"]),
                "home_team_id": game["home_team_id"],
                "away_team_id": game["away_team_id"],
                "game_date": game["game_date"],
                "week": game.get("week"),
                "season": game.get("season"),
                "game_status": game.get("status"),
                **line,
            }

            # Add team information if requested
            if include_team_info:
                home_team = await db.teams.find_one({"_id": game["home_team_id"]})
                away_team = await db.teams.find_one({"_id": game["away_team_id"]})

                if home_team:
                    line_data["home_team"] = {
                        "name": home_team["name"],
                        "abbreviation": home_team["abbreviation"],
                        "city": home_team["city"],
                    }

                if away_team:
                    line_data["away_team"] = {
                        "name": away_team["name"],
                        "abbreviation": away_team["abbreviation"],
                        "city": away_team["city"],
                    }

            game_betting_lines.append(line_data)

        betting_data.extend(game_betting_lines)

    # Get historical line movements if requested
    if include_history and game_id:
        historical_lines = await _get_historical_betting_lines(db, game_id, sportsbook)
        return {
            "current_lines": betting_data,
            "historical_lines": historical_lines,
            "total_current": len(betting_data),
        }

    # Also get raw betting lines from data ingestion
    raw_lines = await _get_raw_betting_lines(db, sportsbook, week, season)

    return {
        "betting_lines": betting_data,
        "raw_betting_data": raw_lines,
        "total": len(betting_data),
        "last_updated": datetime.now().isoformat(),
    }


async def _get_historical_betting_lines(
    db, game_id: str, sportsbook: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Retrieve historical betting line movements for a game.

    Parameters:
        game_id (str): Identifier of the game to fetch historical lines for.
        sportsbook (Optional[str]): If provided, filter historical lines to this sportsbook.

    Returns:
        List[Dict[str, Any]]: Historical betting line documents (each has `_id` converted to `id`), ordered by `processed_at` ascending.
    """
    query = {"game_id": game_id}
    if sportsbook:
        query["sportsbook"] = sportsbook

    cursor = db.raw_betting_lines.find(query).sort([("processed_at", 1)])
    historical_lines = []

    async for line in cursor:
        line["id"] = str(line["_id"])
        historical_lines.append(line)

    return historical_lines


async def _get_raw_betting_lines(
    db, sportsbook: Optional[str], week: Optional[int], season: Optional[int]
) -> List[Dict[str, Any]]:
    """
    Retrieve recent raw betting lines ingested in the last 24 hours, optionally filtered by sportsbook.

    Parameters:
        sportsbook (Optional[str]): If provided, only return lines from this sportsbook.
        week (Optional[int]): Currently ignored; included for API compatibility.
        season (Optional[int]): Currently ignored; included for API compatibility.

    Returns:
        List[Dict[str, Any]]: List of raw betting line documents. Each document includes an `id`
        field (string) derived from the original `_id`.
    """
    query = {}
    if sportsbook:
        query["sportsbook"] = sportsbook

    # Get recent raw betting lines (last 24 hours)
    yesterday = datetime.now() - timedelta(days=1)
    query["processed_at"] = {"$gte": yesterday}

    cursor = db.raw_betting_lines.find(query).sort([("processed_at", -1)]).limit(50)
    raw_lines = []

    async for line in cursor:
        line["id"] = str(line["_id"])
        raw_lines.append(line)

    return raw_lines


@router.get("/schedule")
async def get_schedule(
    week: Optional[int] = None,
    season: Optional[int] = None,
    team_id: Optional[str] = None,
    db=Depends(get_database),
):
    """
    Retrieve the upcoming and in-progress NFL schedule, optionally filtered by week, season, or team.

    Parameters:
        week (Optional[int]): If provided, restrict results to the given week.
        season (Optional[int]): If provided, restrict results to the given season (year).
        team_id (Optional[str]): If provided, return games where the team is home or away.

    Returns:
        dict: A mapping with key `"schedule"` containing a list of game documents. Each game document has its database `_id` converted to string as the `id` field.
    """
    query = {"status": {"$in": ["scheduled", "in_progress"]}}

    if week:
        query["week"] = week
    if season:
        query["season"] = season
    if team_id:
        query["$or"] = [{"home_team_id": team_id}, {"away_team_id": team_id}]

    cursor = db.games.find(query).sort([("game_date", 1)])
    schedule = []

    async for doc in cursor:
        doc["id"] = str(doc["_id"])
        schedule.append(doc)

    return {"schedule": schedule}


@router.get("/pipeline-status")
async def get_pipeline_status():
    """
    Retrieve current status of the data ingestion pipeline.

    Returns:
        result (dict): A mapping with keys "status" (typically "success") and "data" containing pipeline statistics.

    Raises:
        HTTPException: Raised with status code 500 if pipeline statistics cannot be retrieved.
    """
    try:
        stats = await data_processing_pipeline.get_pipeline_stats()
        return {"status": "success", "data": stats}
    except Exception as e:
        logger.error(f"Error getting pipeline status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving pipeline status",
        )


@router.post("/trigger-collection")
async def trigger_data_collection(
    source: str = Query(..., description="Data source: twitter, espn, betting"),
    db=Depends(get_database),
):
    """
    Trigger manual collection and processing of data from a specified source.

    Parameters:
        source (str): Data source to collect from; accepted values (case-insensitive): 'twitter', 'espn', or 'betting'.
        db: Database dependency (injected); omitted from detailed description as it's a standard dependency.

    Returns:
        dict: Result object containing:
            - `status` (str): "success" on successful collection.
            - `source` (str): The requested source.
            - `collection_stats` (Any): Statistics returned by the processing step.
            - `timestamp` (str): ISO-formatted timestamp when the collection completed.

    Raises:
        HTTPException: Raises a 400 Bad Request if `source` is not one of the accepted values.
        HTTPException: Raises a 500 Internal Server Error if collection or processing fails.
    """
    try:
        if source.lower() == "twitter":
            keywords = ["NFL", "football", "Chiefs", "Bills", "Cowboys"]
            raw_items = (
                await data_processing_pipeline.data_ingestion.collect_twitter_data(
                    keywords, 50
                )
            )
            stats = await data_processing_pipeline.data_ingestion.process_raw_data(
                raw_items
            )
        elif source.lower() == "espn":
            raw_items = await data_processing_pipeline.data_ingestion.fetch_espn_data()
            stats = await data_processing_pipeline.data_ingestion.process_raw_data(
                raw_items
            )
        elif source.lower() == "betting":
            raw_items = await data_processing_pipeline.data_ingestion.get_betting_lines(
                ["draftkings", "mgm"]
            )
            stats = await data_processing_pipeline.data_ingestion.process_raw_data(
                raw_items
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid source. Must be 'twitter', 'espn', or 'betting'",
            )

        return {
            "status": "success",
            "source": source,
            "collection_stats": stats,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error triggering data collection: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error collecting data from {source}",
        )


@router.get("/data-summary")
async def get_data_summary(
    days: int = Query(7, ge=1, le=30, description="Number of days to summarize"),
    db=Depends(get_database),
):
    """
    Summarize collected raw data and sentiment counts over the past N days.

    Parameters:
        days (int): Number of days to include in the summary (minimum 1, maximum 30).

    Returns:
        dict: Summary containing:
            - period (str): period label like "last_7_days".
            - start_date (str): ISO timestamp for the start of the period.
            - end_date (str): ISO timestamp for the end of the period (now).
            - data_collected (dict): counts for 'tweets', 'news_articles', 'games', 'betting_lines', and 'sentiment_analyses'.
            - sentiment_by_source (dict): mapping of sentiment source -> mention count.
            - sentiment_breakdown (dict): mapping of sentiment label -> count.

    Raises:
        HTTPException: with status 500 if an error occurs while retrieving the summary.
    """
    try:
        start_date = datetime.now() - timedelta(days=days)

        # Count documents in each collection
        tweets_count = await db.raw_tweets.count_documents(
            {"processed_at": {"$gte": start_date}}
        )

        news_count = await db.raw_news.count_documents(
            {"processed_at": {"$gte": start_date}}
        )

        games_count = await db.raw_games.count_documents(
            {"processed_at": {"$gte": start_date}}
        )

        betting_lines_count = await db.raw_betting_lines.count_documents(
            {"processed_at": {"$gte": start_date}}
        )

        sentiment_count = await db.sentiment_analyses.count_documents(
            {"created_at": {"$gte": start_date}}
        )

        # Get source breakdown for sentiment analyses
        sentiment_sources = await db.sentiment_analyses.aggregate(
            [
                {"$match": {"created_at": {"$gte": start_date}}},
                {"$group": {"_id": "$source", "count": {"$sum": 1}}},
            ]
        ).to_list(length=None)

        # Get sentiment breakdown
        sentiment_breakdown = await db.sentiment_analyses.aggregate(
            [
                {"$match": {"created_at": {"$gte": start_date}}},
                {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}},
            ]
        ).to_list(length=None)

        return {
            "period": f"last_{days}_days",
            "start_date": start_date.isoformat(),
            "end_date": datetime.now().isoformat(),
            "data_collected": {
                "tweets": tweets_count,
                "news_articles": news_count,
                "games": games_count,
                "betting_lines": betting_lines_count,
                "sentiment_analyses": sentiment_count,
            },
            "sentiment_by_source": {
                item["_id"]: item["count"] for item in sentiment_sources
            },
            "sentiment_breakdown": {
                item["_id"]: item["count"] for item in sentiment_breakdown
            },
        }

    except Exception as e:
        logger.error(f"Error getting data summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving data summary",
        )


@router.get("/recent-sentiment")
async def get_recent_sentiment(
    hours: int = Query(
        24, ge=1, le=168, description="Hours of recent sentiment to retrieve"
    ),
    source: Optional[str] = Query(None, description="Filter by data source"),
    sentiment_type: Optional[str] = Query(
        None, description="Filter by sentiment: positive, negative, neutral"
    ),
    limit: int = Query(100, ge=1, le=500, description="Maximum number of results"),
    db=Depends(get_database),
):
    """
    Retrieve recent sentiment analyses within a sliding window of past hours, optionally filtered by source and sentiment.

    Parameters:
        hours (int): Number of past hours to include in the search (1–168).
        source (Optional[str]): Optional data source to filter results (e.g., "twitter", "espn").
        sentiment_type (Optional[str]): Optional sentiment filter; one of "positive", "negative", or "neutral".
        limit (int): Maximum number of results to return (1–500).

    Returns:
        dict: {
            "period_hours": int,                 # the requested hours window
            "total_results": int,                # number of returned analyses
            "sentiment_analyses": list,          # list of sentiment analysis documents from the database; each doc includes an added "id" field (string) derived from the original "_id"
            "filters": dict                      # echo of applied filters: {"source": source, "sentiment_type": sentiment_type}
        }

    Raises:
        HTTPException: If retrieval from the database fails, raises a 500 HTTPException with a retrieval error detail.
    """
    try:
        start_time = datetime.now() - timedelta(hours=hours)

        query = {"created_at": {"$gte": start_time}}
        if source:
            query["source"] = source
        if sentiment_type:
            query["sentiment"] = sentiment_type.lower()

        cursor = (
            db.sentiment_analyses.find(query).sort([("created_at", -1)]).limit(limit)
        )

        sentiment_data = []
        async for doc in cursor:
            doc["id"] = str(doc["_id"])
            doc.pop("_id", None)
            sentiment_data.append(doc)

        return {
            "period_hours": hours,
            "total_results": len(sentiment_data),
            "sentiment_analyses": sentiment_data,
            "filters": {"source": source, "sentiment_type": sentiment_type},
        }

    except Exception as e:
        logger.error(f"Error getting recent sentiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving recent sentiment data",
        )


@router.get("/trending-topics")
async def get_trending_topics(
    hours: int = Query(24, ge=1, le=168, description="Hours to analyze for trends"),
    min_mentions: int = Query(
        5, ge=1, le=100, description="Minimum mentions to be considered trending"
    ),
    db=Depends(get_database),
):
    """
    Retrieve trending NFL keywords and teams from recent sentiment analyses.

    Parameters:
        hours (int): Time window in hours to analyze for trends.
        min_mentions (int): Minimum number of mentions required for an item to be included.

    Returns:
        dict: {
            "period_hours": int,           # analyzed hours window
            "min_mentions": int,           # threshold used
            "trending_keywords": list,     # list of keyword documents with fields like `_id` (keyword), `mention_count`, `avg_sentiment`
            "trending_teams": list,        # list of team documents with fields like `_id` (team id), `team_name`, `mention_count`, `avg_sentiment`
            "generated_at": str            # ISO 8601 timestamp when results were generated
        }
    """
    try:
        start_time = datetime.now() - timedelta(hours=hours)

        # Aggregate trending keywords from recent sentiment analyses
        pipeline = [
            {"$match": {"created_at": {"$gte": start_time}}},
            {"$unwind": "$metadata.nfl_keywords"},
            {
                "$group": {
                    "_id": "$metadata.nfl_keywords",
                    "mention_count": {"$sum": 1},
                    "avg_sentiment": {
                        "$avg": {
                            "$cond": [
                                {"$eq": ["$sentiment", "positive"]},
                                1,
                                {"$cond": [{"$eq": ["$sentiment", "negative"]}, -1, 0]},
                            ]
                        }
                    },
                }
            },
            {"$match": {"mention_count": {"$gte": min_mentions}}},
            {"$sort": {"mention_count": -1}},
            {"$limit": 20},
        ]

        trending_keywords = await db.sentiment_analyses.aggregate(pipeline).to_list(
            length=None
        )

        # Get trending teams (most mentioned)
        team_pipeline = [
            {"$match": {"created_at": {"$gte": start_time}}},
            {
                "$lookup": {
                    "from": "teams",
                    "let": {"text": "$text"},
                    "pipeline": [
                        {
                            "$match": {
                                "$expr": {
                                    "$or": [
                                        {
                                            "$regexMatch": {
                                                "input": "$$text",
                                                "regex": {"$concat": ["(?i)", "$name"]},
                                            }
                                        },
                                        {
                                            "$regexMatch": {
                                                "input": "$$text",
                                                "regex": {
                                                    "$concat": ["(?i)", "$abbreviation"]
                                                },
                                            }
                                        },
                                    ]
                                }
                            }
                        }
                    ],
                    "as": "mentioned_teams",
                }
            },
            {"$unwind": "$mentioned_teams"},
            {
                "$group": {
                    "_id": "$mentioned_teams._id",
                    "team_name": {"$first": "$mentioned_teams.name"},
                    "mention_count": {"$sum": 1},
                    "avg_sentiment": {
                        "$avg": {
                            "$cond": [
                                {"$eq": ["$sentiment", "positive"]},
                                1,
                                {"$cond": [{"$eq": ["$sentiment", "negative"]}, -1, 0]},
                            ]
                        }
                    },
                }
            },
            {"$match": {"mention_count": {"$gte": min_mentions}}},
            {"$sort": {"mention_count": -1}},
            {"$limit": 10},
        ]

        trending_teams = await db.sentiment_analyses.aggregate(team_pipeline).to_list(
            length=None
        )

        return {
            "period_hours": hours,
            "min_mentions": min_mentions,
            "trending_keywords": trending_keywords,
            "trending_teams": trending_teams,
            "generated_at": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting trending topics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving trending topics",
        )
