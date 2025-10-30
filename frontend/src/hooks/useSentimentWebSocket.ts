import { useState, useCallback, useEffect } from 'react'
import useWebSocket, { WebSocketMessage } from './useWebSocket'

export interface SentimentData {
  id: string
  text: string
  sentiment: 'positive' | 'negative' | 'neutral'
  confidence: number
  team_id?: string
  player_id?: string
  game_id?: string
  source: string
  timestamp: string
}

export interface TeamSentiment {
  team_id: string
  team_name: string
  current_sentiment: number
  sentiment_trend: Array<{
    timestamp: string
    sentiment: number
  }>
  total_mentions: number
}

export interface GamePrediction {
  game_id: string
  home_team: string
  away_team: string
  prediction: {
    home_win_probability: number
    away_win_probability: number
    sentiment_factor: number
  }
  betting_lines?: {
    spread: number
    over_under: number
    moneyline: {
      home: number
      away: number
    }
  }
}

export interface SentimentWebSocketData {
  recentSentiments: SentimentData[]
  teamSentiments: Record<string, TeamSentiment>
  gamePredictions: Record<string, GamePrediction>
  connectionStatus: {
    isConnected: boolean
    isConnecting: boolean
    error: string | null
    reconnectAttempts: number
  }
}

export function useSentimentWebSocket() {
  const [data, setData] = useState<SentimentWebSocketData>({
    recentSentiments: [],
    teamSentiments: {},
    gamePredictions: {},
    connectionStatus: {
      isConnected: false,
      isConnecting: false,
      error: null,
      reconnectAttempts: 0
    }
  })

  const handleMessage = useCallback((message: WebSocketMessage) => {
    console.log('Received WebSocket message:', message)

    switch (message.type) {
      case 'connection':
        console.log('WebSocket connection confirmed')
        break

      case 'initial_data':
        if (message.data && typeof message.data === 'object' && message.data !== null) {
          const data = message.data as { recent_sentiments?: SentimentData[] }
          if (data.recent_sentiments) {
            setData(prev => ({
              ...prev,
              recentSentiments: data.recent_sentiments || []
            }))
          }
        }
        break

      case 'sentiment_update':
        if (message.data) {
          const sentimentData = message.data as SentimentData
          setData(prev => ({
            ...prev,
            recentSentiments: [sentimentData, ...prev.recentSentiments.slice(0, 9)]
          }))
        }
        break

      case 'team_sentiment_update':
        if (message.team_id && message.data) {
          const teamId = message.team_id as string
          const teamData = message.data as TeamSentiment
          setData(prev => ({
            ...prev,
            teamSentiments: {
              ...prev.teamSentiments,
              [teamId]: teamData
            }
          }))
        }
        break

      case 'game_prediction_update':
        if (message.game_id && message.data) {
          const gameId = message.game_id as string
          const gameData = message.data as GamePrediction
          setData(prev => ({
            ...prev,
            gamePredictions: {
              ...prev.gamePredictions,
              [gameId]: gameData
            }
          }))
        }
        break

      case 'pong':
        // Handle ping/pong for connection health
        break

      case 'error':
        console.error('WebSocket error:', message.message)
        break

      default:
        console.log('Unknown message type:', message.type)
    }
  }, [])

  const handleConnect = useCallback(() => {
    console.log('WebSocket connected')
  }, [])

  const handleDisconnect = useCallback(() => {
    console.log('WebSocket disconnected')
  }, [])

  const handleError = useCallback((error: Event) => {
    console.error('WebSocket error:', error)
  }, [])

  const websocket = useWebSocket('/ws/sentiment', {
    onMessage: handleMessage,
    onConnect: handleConnect,
    onDisconnect: handleDisconnect,
    onError: handleError,
    reconnectInterval: 3000,
    maxReconnectAttempts: 5
  })

  // Update connection status
  useEffect(() => {
    setData(prev => ({
      ...prev,
      connectionStatus: {
        isConnected: websocket.isConnected,
        isConnecting: websocket.isConnecting,
        error: websocket.error,
        reconnectAttempts: websocket.reconnectAttempts
      }
    }))
  }, [websocket.isConnected, websocket.isConnecting, websocket.error, websocket.reconnectAttempts])

  const subscribeToTeam = useCallback((teamId: string) => {
    websocket.sendMessage({
      type: 'subscribe',
      subscription: 'team_sentiment',
      team_id: teamId
    })
  }, [websocket])

  const subscribeToGame = useCallback((gameId: string) => {
    websocket.sendMessage({
      type: 'subscribe',
      subscription: 'game_prediction',
      game_id: gameId
    })
  }, [websocket])

  const ping = useCallback(() => {
    websocket.sendMessage({
      type: 'ping',
      timestamp: new Date().toISOString()
    })
  }, [websocket])

  // Send periodic pings to keep connection alive
  useEffect(() => {
    if (websocket.isConnected) {
      const interval = setInterval(ping, 30000) // Ping every 30 seconds
      return () => { clearInterval(interval) }
    }
  }, [websocket.isConnected, ping])

  return {
    ...data,
    subscribeToTeam,
    subscribeToGame,
    reconnect: websocket.reconnect,
    disconnect: websocket.disconnect
  }
}

export default useSentimentWebSocket