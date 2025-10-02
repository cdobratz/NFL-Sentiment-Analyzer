import { describe, it, expect, vi, beforeEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import useSentimentWebSocket from '../useSentimentWebSocket'
import * as useWebSocketModule from '../useWebSocket'

// Mock the base WebSocket hook
vi.mock('../useWebSocket')

const mockWebSocketHook = {
  isConnected: true,
  isConnecting: false,
  error: null,
  reconnectAttempts: 0,
  connect: vi.fn(),
  disconnect: vi.fn(),
  reconnect: vi.fn(),
  sendMessage: vi.fn(),
}

describe('useSentimentWebSocket', () => {
  let mockOnMessage: (message: any) => void

  beforeEach(() => {
    vi.clearAllMocks()
    
    // Mock the useWebSocket hook and capture the onMessage callback
    vi.mocked(useWebSocketModule.default).mockImplementation((endpoint, options) => {
      mockOnMessage = options?.onMessage || (() => {})
      return mockWebSocketHook
    })
  })

  it('initializes with correct default state', () => {
    const { result } = renderHook(() => useSentimentWebSocket())

    expect(result.current.recentSentiments).toEqual([])
    expect(result.current.teamSentiments).toEqual({})
    expect(result.current.gamePredictions).toEqual({})
    expect(result.current.connectionStatus.isConnected).toBe(true)
    expect(result.current.connectionStatus.isConnecting).toBe(false)
    expect(result.current.connectionStatus.error).toBe(null)
    expect(result.current.connectionStatus.reconnectAttempts).toBe(0)
  })

  it('connects to correct WebSocket endpoint', () => {
    renderHook(() => useSentimentWebSocket())

    expect(useWebSocketModule.default).toHaveBeenCalledWith(
      '/ws/sentiment',
      expect.objectContaining({
        onMessage: expect.any(Function),
        onConnect: expect.any(Function),
        onDisconnect: expect.any(Function),
        onError: expect.any(Function),
        reconnectInterval: 3000,
        maxReconnectAttempts: 5,
      })
    )
  })

  it('handles connection message', () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
    
    renderHook(() => useSentimentWebSocket())

    act(() => {
      mockOnMessage({
        type: 'connection',
        timestamp: '2024-01-15T10:00:00Z',
      })
    })

    expect(consoleSpy).toHaveBeenCalledWith('WebSocket connection confirmed')
    consoleSpy.mockRestore()
  })

  it('handles initial data message', () => {
    const { result } = renderHook(() => useSentimentWebSocket())

    const initialData = [
      {
        id: '1',
        text: 'Great game!',
        sentiment: 'positive',
        confidence: 0.9,
        team_id: '1',
        source: 'twitter',
        timestamp: '2024-01-15T10:00:00Z',
      },
    ]

    act(() => {
      mockOnMessage({
        type: 'initial_data',
        data: {
          recent_sentiments: initialData,
        },
        timestamp: '2024-01-15T10:00:00Z',
      })
    })

    expect(result.current.recentSentiments).toEqual(initialData)
  })

  it('handles sentiment update message', () => {
    const { result } = renderHook(() => useSentimentWebSocket())

    const newSentiment = {
      id: '2',
      text: 'Amazing play!',
      sentiment: 'positive',
      confidence: 0.95,
      team_id: '1',
      source: 'twitter',
      timestamp: '2024-01-15T10:05:00Z',
    }

    act(() => {
      mockOnMessage({
        type: 'sentiment_update',
        data: newSentiment,
        timestamp: '2024-01-15T10:05:00Z',
      })
    })

    expect(result.current.recentSentiments).toHaveLength(1)
    expect(result.current.recentSentiments[0]).toEqual(newSentiment)
  })

  it('limits recent sentiments to 10 items', () => {
    const { result } = renderHook(() => useSentimentWebSocket())

    // Add 12 sentiments
    for (let i = 1; i <= 12; i++) {
      act(() => {
        mockOnMessage({
          type: 'sentiment_update',
          data: {
            id: i.toString(),
            text: `Sentiment ${i}`,
            sentiment: 'positive',
            confidence: 0.8,
            team_id: '1',
            source: 'twitter',
            timestamp: `2024-01-15T10:${i.toString().padStart(2, '0')}:00Z`,
          },
          timestamp: `2024-01-15T10:${i.toString().padStart(2, '0')}:00Z`,
        })
      })
    }

    expect(result.current.recentSentiments).toHaveLength(10)
    expect(result.current.recentSentiments[0].id).toBe('12') // Most recent first
    expect(result.current.recentSentiments[9].id).toBe('3') // Oldest kept
  })

  it('handles team sentiment update message', () => {
    const { result } = renderHook(() => useSentimentWebSocket())

    const teamSentiment = {
      team_id: '1',
      team_name: 'New England Patriots',
      current_sentiment: 0.75,
      sentiment_trend: [
        { timestamp: '2024-01-15T09:00:00Z', sentiment: 0.7 },
        { timestamp: '2024-01-15T10:00:00Z', sentiment: 0.75 },
      ],
      total_mentions: 150,
    }

    act(() => {
      mockOnMessage({
        type: 'team_sentiment_update',
        team_id: '1',
        data: teamSentiment,
        timestamp: '2024-01-15T10:00:00Z',
      })
    })

    expect(result.current.teamSentiments['1']).toEqual(teamSentiment)
  })

  it('handles game prediction update message', () => {
    const { result } = renderHook(() => useSentimentWebSocket())

    const gamePrediction = {
      game_id: '1',
      home_team: 'New England Patriots',
      away_team: 'Dallas Cowboys',
      prediction: {
        home_win_probability: 0.65,
        away_win_probability: 0.35,
        sentiment_factor: 0.8,
      },
      betting_lines: {
        spread: -3.5,
        over_under: 45.5,
        moneyline: {
          home: -150,
          away: 130,
        },
      },
    }

    act(() => {
      mockOnMessage({
        type: 'game_prediction_update',
        game_id: '1',
        data: gamePrediction,
        timestamp: '2024-01-15T10:00:00Z',
      })
    })

    expect(result.current.gamePredictions['1']).toEqual(gamePrediction)
  })

  it('handles pong message', () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
    
    renderHook(() => useSentimentWebSocket())

    act(() => {
      mockOnMessage({
        type: 'pong',
        timestamp: '2024-01-15T10:00:00Z',
      })
    })

    // Should not log anything for pong messages
    expect(consoleSpy).not.toHaveBeenCalled()
    consoleSpy.mockRestore()
  })

  it('handles error message', () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    
    renderHook(() => useSentimentWebSocket())

    act(() => {
      mockOnMessage({
        type: 'error',
        message: 'Something went wrong',
        timestamp: '2024-01-15T10:00:00Z',
      })
    })

    expect(consoleSpy).toHaveBeenCalledWith('WebSocket error:', 'Something went wrong')
    consoleSpy.mockRestore()
  })

  it('handles unknown message types', () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
    
    renderHook(() => useSentimentWebSocket())

    act(() => {
      mockOnMessage({
        type: 'unknown_type',
        timestamp: '2024-01-15T10:00:00Z',
      })
    })

    expect(consoleSpy).toHaveBeenCalledWith('Unknown message type:', 'unknown_type')
    consoleSpy.mockRestore()
  })

  it('subscribes to team sentiment updates', () => {
    const { result } = renderHook(() => useSentimentWebSocket())

    act(() => {
      result.current.subscribeToTeam('1')
    })

    expect(mockWebSocketHook.sendMessage).toHaveBeenCalledWith({
      type: 'subscribe',
      subscription: 'team_sentiment',
      team_id: '1',
    })
  })

  it('subscribes to game prediction updates', () => {
    const { result } = renderHook(() => useSentimentWebSocket())

    act(() => {
      result.current.subscribeToGame('1')
    })

    expect(mockWebSocketHook.sendMessage).toHaveBeenCalledWith({
      type: 'subscribe',
      subscription: 'game_prediction',
      game_id: '1',
    })
  })

  it('sends periodic ping messages when connected', () => {
    vi.useFakeTimers()
    
    const { result } = renderHook(() => useSentimentWebSocket())

    // Fast-forward 30 seconds
    act(() => {
      vi.advanceTimersByTime(30000)
    })

    expect(mockWebSocketHook.sendMessage).toHaveBeenCalledWith({
      type: 'ping',
      timestamp: expect.any(String),
    })

    vi.useRealTimers()
  })

  it('does not send ping when disconnected', () => {
    vi.useFakeTimers()
    
    // Mock disconnected state
    const disconnectedWebSocketHook = {
      ...mockWebSocketHook,
      isConnected: false,
    }
    
    vi.mocked(useWebSocketModule.default).mockReturnValue(disconnectedWebSocketHook)
    
    renderHook(() => useSentimentWebSocket())

    // Fast-forward 30 seconds
    act(() => {
      vi.advanceTimersByTime(30000)
    })

    expect(mockWebSocketHook.sendMessage).not.toHaveBeenCalled()

    vi.useRealTimers()
  })

  it('updates connection status when WebSocket state changes', () => {
    const { result, rerender } = renderHook(() => useSentimentWebSocket())

    expect(result.current.connectionStatus.isConnected).toBe(true)

    // Mock disconnected state
    const disconnectedWebSocketHook = {
      ...mockWebSocketHook,
      isConnected: false,
      error: 'Connection lost',
      reconnectAttempts: 2,
    }
    
    vi.mocked(useWebSocketModule.default).mockReturnValue(disconnectedWebSocketHook)
    
    rerender()

    expect(result.current.connectionStatus.isConnected).toBe(false)
    expect(result.current.connectionStatus.error).toBe('Connection lost')
    expect(result.current.connectionStatus.reconnectAttempts).toBe(2)
  })

  it('provides reconnect and disconnect functions', () => {
    const { result } = renderHook(() => useSentimentWebSocket())

    expect(typeof result.current.reconnect).toBe('function')
    expect(typeof result.current.disconnect).toBe('function')
    expect(result.current.reconnect).toBe(mockWebSocketHook.reconnect)
    expect(result.current.disconnect).toBe(mockWebSocketHook.disconnect)
  })

  it('ignores messages without required data', () => {
    const { result } = renderHook(() => useSentimentWebSocket())

    // Team sentiment update without team_id
    act(() => {
      mockOnMessage({
        type: 'team_sentiment_update',
        data: { team_name: 'Test Team' },
        timestamp: '2024-01-15T10:00:00Z',
      })
    })

    expect(Object.keys(result.current.teamSentiments)).toHaveLength(0)

    // Game prediction update without game_id
    act(() => {
      mockOnMessage({
        type: 'game_prediction_update',
        data: { home_team: 'Team A' },
        timestamp: '2024-01-15T10:00:00Z',
      })
    })

    expect(Object.keys(result.current.gamePredictions)).toHaveLength(0)
  })
})