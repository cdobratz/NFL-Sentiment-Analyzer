from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import List, Dict, Any
import json
import asyncio
import logging
from datetime import datetime

from ..core.dependencies import get_current_user_websocket
from ..services.sentiment_service import sentiment_service
from ..services.data_ingestion_service import data_ingestion_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])


class ConnectionManager:
    """Manages WebSocket connections and broadcasts"""

    def __init__(self):
        """
        Initialize the ConnectionManager.
        
        Create storage for all active WebSocket connections and for user-scoped connections keyed by user ID.
        """
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str = None):
        """
        Register and accept a new WebSocket connection and track it for broadcasting.
        
        Registers the connection in the manager's global active connections list and, if a user_id is provided, associates the connection with that user's connection list for targeted messaging. Logs the updated total of active connections.
        
        Parameters:
            websocket (WebSocket): The WebSocket connection to accept and track.
            user_id (str, optional): Identifier of the authenticated user to associate this connection with; if omitted the connection is tracked only globally.
        """
        await websocket.accept()
        self.active_connections.append(websocket)

        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)

        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket, user_id: str = None):
        """
        Remove a WebSocket connection from the manager's tracked state.
        
        If `user_id` is provided, also remove the connection from that user's list and delete the user's entry if it becomes empty.
        
        Parameters:
            websocket (WebSocket): The WebSocket connection to remove.
            user_id (str, optional): The associated user ID whose connection list should also be updated.
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

        if user_id and user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]

        logger.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def send_to_user(self, message: str, user_id: str):
        """
        Send a text message to every active WebSocket connection associated with a user.
        
        If sending to a connection fails, the failure is logged and the affected connection is removed
        from the manager's tracking.
        
        Parameters:
            message (str): Text payload to send to the user's connections.
            user_id (str): Identifier of the target user whose connections will receive the message.
        """
        if user_id in self.user_connections:
            disconnected = []
            for websocket in self.user_connections[user_id]:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id}: {e}")
                    disconnected.append(websocket)

            # Clean up disconnected websockets
            for ws in disconnected:
                self.disconnect(ws, user_id)

    async def broadcast(self, message: str):
        """
        Send a text message to every active WebSocket connection.
        
        If sending to a connection fails, that connection is recorded and removed from the manager after all send attempts complete.
        
        Parameters:
            message (str): The text payload to send to all connected clients.
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected.append(connection)

        # Clean up disconnected websockets
        for ws in disconnected:
            self.disconnect(ws)


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/sentiment")
async def websocket_sentiment_endpoint(
    websocket: WebSocket, user: dict = Depends(get_current_user_websocket)
):
    """
    Manage a WebSocket connection for real-time sentiment updates and client subscriptions.
    
    Accepts an authenticated WebSocket connection, registers it with the connection manager, sends a connection confirmation and recent sentiment data, and handles incoming client messages such as "ping" (responds with "pong") and "subscribe" for team-specific sentiment. On invalid JSON or internal errors, sends an error payload to the client. Ensures the connection is removed from the manager on disconnect or error.
    
    Parameters:
        websocket (WebSocket): The WebSocket connection instance.
        user (dict): Authenticated user document from MongoDB (contains "_id" field).
    """
    user_id = str(user["_id"]) if user and user.get("_id") else None
    await manager.connect(websocket, user_id)

    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps(
                {
                    "type": "connection",
                    "status": "connected",
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": user_id,
                }
            ),
            websocket,
        )

        # Send initial data
        try:
            # Get recent sentiment data
            recent_sentiments = await sentiment_service.get_recent_sentiment(limit=10)
            await manager.send_personal_message(
                json.dumps(
                    {
                        "type": "initial_data",
                        "data": {"recent_sentiments": recent_sentiments},
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ),
                websocket,
            )
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_text()
                message = json.loads(data)

                # Handle different message types
                if message.get("type") == "ping":
                    await manager.send_personal_message(
                        json.dumps(
                            {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
                        ),
                        websocket,
                    )
                elif message.get("type") == "subscribe":
                    # Handle subscription to specific data streams
                    subscription_type = message.get("subscription")
                    if subscription_type == "team_sentiment":
                        team_id = message.get("team_id")
                        # Send team-specific sentiment data
                        team_sentiment = await sentiment_service.get_team_sentiment(
                            team_id
                        )
                        await manager.send_personal_message(
                            json.dumps(
                                {
                                    "type": "team_sentiment",
                                    "team_id": team_id,
                                    "data": team_sentiment,
                                    "timestamp": datetime.utcnow().isoformat(),
                                }
                            ),
                            websocket,
                        )

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Invalid JSON format",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    ),
                    websocket,
                )
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await manager.send_personal_message(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Internal server error",
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    ),
                    websocket,
                )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, user_id)


async def broadcast_sentiment_update(sentiment_data: Dict[str, Any]):
    """
    Broadcast a sentiment update payload to all connected WebSocket clients.
    
    Constructs a `sentiment_update` message containing the provided sentiment data and the current UTC timestamp, then sends it to all active connections.
    
    Parameters:
        sentiment_data (Dict[str, Any]): The sentiment payload to include in the broadcast.
    """
    message = json.dumps(
        {
            "type": "sentiment_update",
            "data": sentiment_data,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    await manager.broadcast(message)


async def broadcast_team_sentiment_update(team_id: str, sentiment_data: Dict[str, Any]):
    """
    Broadcasts a team-specific sentiment update to all connected WebSocket clients.
    
    Parameters:
    	team_id (str): Identifier of the team for which the sentiment update applies.
    	sentiment_data (Dict[str, Any]): Sentiment payload containing metrics and any relevant metadata.
    """
    message = json.dumps(
        {
            "type": "team_sentiment_update",
            "team_id": team_id,
            "data": sentiment_data,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    await manager.broadcast(message)


async def broadcast_game_prediction_update(
    game_id: str, prediction_data: Dict[str, Any]
):
    """
    Broadcasts a game prediction update to all connected WebSocket clients.
    
    Sends a JSON message containing the update type, the target `game_id`, the provided `prediction_data`, and a UTC timestamp.
    
    Parameters:
        game_id (str): Identifier of the game the prediction applies to.
        prediction_data (dict): SerializabIe payload with prediction details to include in the message.
    """
    message = json.dumps(
        {
            "type": "game_prediction_update",
            "game_id": game_id,
            "data": prediction_data,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )
    await manager.broadcast(message)


# Export the manager for use in other modules
__all__ = [
    "router",
    "manager",
    "broadcast_sentiment_update",
    "broadcast_team_sentiment_update",
    "broadcast_game_prediction_update",
]
