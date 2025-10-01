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
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[str, List[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(websocket)
        
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, user_id: str = None):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        if user_id and user_id in self.user_connections:
            if websocket in self.user_connections[user_id]:
                self.user_connections[user_id].remove(websocket)
            if not self.user_connections[user_id]:
                del self.user_connections[user_id]
        
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def send_to_user(self, message: str, user_id: str):
        """Send a message to all connections for a specific user"""
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
        """Broadcast a message to all connected clients"""
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
    websocket: WebSocket,
    user: dict = Depends(get_current_user_websocket)
):
    """WebSocket endpoint for real-time sentiment updates"""
    user_id = user.get("sub") if user else None
    await manager.connect(websocket, user_id)
    
    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connection",
                "status": "connected",
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id
            }),
            websocket
        )
        
        # Send initial data
        try:
            # Get recent sentiment data
            recent_sentiments = await sentiment_service.get_recent_sentiment(limit=10)
            await manager.send_personal_message(
                json.dumps({
                    "type": "initial_data",
                    "data": {
                        "recent_sentiments": recent_sentiments
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }),
                websocket
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
                        json.dumps({
                            "type": "pong",
                            "timestamp": datetime.utcnow().isoformat()
                        }),
                        websocket
                    )
                elif message.get("type") == "subscribe":
                    # Handle subscription to specific data streams
                    subscription_type = message.get("subscription")
                    if subscription_type == "team_sentiment":
                        team_id = message.get("team_id")
                        # Send team-specific sentiment data
                        team_sentiment = await sentiment_service.get_team_sentiment(team_id)
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "team_sentiment",
                                "team_id": team_id,
                                "data": team_sentiment,
                                "timestamp": datetime.utcnow().isoformat()
                            }),
                            websocket
                        )
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    websocket
                )
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await manager.send_personal_message(
                    json.dumps({
                        "type": "error",
                        "message": "Internal server error",
                        "timestamp": datetime.utcnow().isoformat()
                    }),
                    websocket
                )
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket, user_id)


async def broadcast_sentiment_update(sentiment_data: Dict[str, Any]):
    """Broadcast sentiment update to all connected clients"""
    message = json.dumps({
        "type": "sentiment_update",
        "data": sentiment_data,
        "timestamp": datetime.utcnow().isoformat()
    })
    await manager.broadcast(message)


async def broadcast_team_sentiment_update(team_id: str, sentiment_data: Dict[str, Any]):
    """Broadcast team-specific sentiment update"""
    message = json.dumps({
        "type": "team_sentiment_update",
        "team_id": team_id,
        "data": sentiment_data,
        "timestamp": datetime.utcnow().isoformat()
    })
    await manager.broadcast(message)


async def broadcast_game_prediction_update(game_id: str, prediction_data: Dict[str, Any]):
    """Broadcast game prediction update"""
    message = json.dumps({
        "type": "game_prediction_update",
        "game_id": game_id,
        "data": prediction_data,
        "timestamp": datetime.utcnow().isoformat()
    })
    await manager.broadcast(message)


# Export the manager for use in other modules
__all__ = ["router", "manager", "broadcast_sentiment_update", "broadcast_team_sentiment_update", "broadcast_game_prediction_update"]