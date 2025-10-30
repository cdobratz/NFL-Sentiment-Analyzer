import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, act } from '../../../test/utils'
import SentimentDashboard from '../../SentimentDashboard'
import * as api from '../../../services/api'
// import * as useWebSocketModule from '../../../hooks/useWebSocket' // Unused

// Mock the API
vi.mock('../../../services/api', () => ({
  dataApi: {
    getTeams: vi.fn(),
    getGames: vi.fn(),
  },
}))

// Mock Chart.js components
vi.mock('react-chartjs-2', () => ({
  Line: vi.fn(() => <div data-testid="mock-chart">Chart</div>),
}))

// Mock child components
vi.mock('../../TeamSentimentCard', () => ({
  default: vi.fn(({ team, sentimentData, onSelect, isSelected }) => (
    <div 
      data-testid={`team-card-${team.id}`}
      onClick={onSelect}
      className={isSelected ? 'selected' : ''}
    >
      <div data-testid={`sentiment-${team.id}`}>
        {sentimentData?.current_sentiment || 0}
      </div>
      <div data-testid={`mentions-${team.id}`}>
        {sentimentData?.total_mentions || 0}
      </div>
    </div>
  )),
}))

vi.mock('../../RealTimeChart', () => ({
  default: vi.fn(({ data }) => (
    <div data-testid="real-time-chart">
      <div data-testid="chart-data-count">{data.length}</div>
    </div>
  )),
}))

vi.mock('../../GamePredictionPanel', () => ({
  default: vi.fn(({ predictions }) => (
    <div data-testid="game-prediction-panel">
      <div data-testid="predictions-count">{predictions.length}</div>
    </div>
  )),
}))

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  readyState = MockWebSocket.CONNECTING
  onopen: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null

  constructor(public url: string) {
    // Simulate async connection
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN
      this.onopen?.(new Event('open'))
    }, 10)
  }

  send(data: string) {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('WebSocket is not open')
    }
  }

  close() {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.(new CloseEvent('close'))
  }

  // Helper methods for testing
  simulateMessage(data: any) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }))
    }
  }

  simulateError() {
    this.onerror?.(new Event('error'))
  }

  simulateClose(code = 1000, reason = '') {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.(new CloseEvent('close', { code, reason }))
  }
}

// Replace global WebSocket
global.WebSocket = MockWebSocket as any

const mockTeams = [
  {
    id: '1',
    name: 'New England Patriots',
    abbreviation: 'NE',
    conference: 'AFC',
    division: 'East',
  },
  {
    id: '2',
    name: 'Dallas Cowboys',
    abbreviation: 'DAL',
    conference: 'NFC',
    division: 'East',
  },
]

describe('Real-Time Updates Integration Tests', () => {
  let mockWebSocket: MockWebSocket

  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
    
    // Mock successful API response
    vi.mocked(api.dataApi.getTeams).mockResolvedValue(
      
    )

    // Clear any existing WebSocket instances
    ;(global.WebSocket as any).mock?.instances?.forEach((ws: MockWebSocket) => {
      ws.close()
    })
  })

  afterEach(() => {
    vi.useRealTimers()
    mockWebSocket?.close()
  })

  it('establishes WebSocket connection on mount', async () => {
    render(<SentimentDashboard />)

    // Fast-forward to establish connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })

    // Verify WebSocket was created
    expect(MockWebSocket).toHaveBeenCalledWith(
      expect.stringContaining('/ws/sentiment')
    )
  })

  it('receives and displays initial sentiment data', async () => {
    render(<SentimentDashboard />)

    // Establish connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })

    // Get the WebSocket instance
    mockWebSocket = (global.WebSocket as any).mock.instances[0]

    // Simulate receiving initial data
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
      mockWebSocket.simulateMessage({
        type: 'initial_data',
        data: {
          recent_sentiments: initialData,
        },
        timestamp: '2024-01-15T10:00:00Z',
      })
    })

    await waitFor(() => {
      expect(screen.getByText('Great game!')).toBeInTheDocument()
    })

    // Verify chart received the data
    expect(screen.getByTestId('chart-data-count')).toHaveTextContent('1')
  })

  it('updates sentiment data in real-time', async () => {
    render(<SentimentDashboard />)

    // Establish connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })

    mockWebSocket = (global.WebSocket as any).mock.instances[0]

    // Send multiple sentiment updates
    const sentiments = [
      {
        id: '1',
        text: 'Amazing play!',
        sentiment: 'positive',
        confidence: 0.95,
        team_id: '1',
        source: 'twitter',
        timestamp: '2024-01-15T10:00:00Z',
      },
      {
        id: '2',
        text: 'Terrible decision',
        sentiment: 'negative',
        confidence: 0.85,
        team_id: '1',
        source: 'twitter',
        timestamp: '2024-01-15T10:01:00Z',
      },
      {
        id: '3',
        text: 'Good effort',
        sentiment: 'positive',
        confidence: 0.75,
        team_id: '2',
        source: 'twitter',
        timestamp: '2024-01-15T10:02:00Z',
      },
    ]

    // Send updates one by one
    for (const sentiment of sentiments) {
      act(() => {
        mockWebSocket.simulateMessage({
          type: 'sentiment_update',
          data: sentiment,
          timestamp: sentiment.timestamp,
        })
      })

      await waitFor(() => {
        expect(screen.getByText(sentiment.text)).toBeInTheDocument()
      })
    }

    // Verify chart data count increased
    expect(screen.getByTestId('chart-data-count')).toHaveTextContent('3')
  })

  it('updates team sentiment data', async () => {
    render(<SentimentDashboard />)

    // Establish connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })

    mockWebSocket = (global.WebSocket as any).mock.instances[0]

    // Send team sentiment update
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
      mockWebSocket.simulateMessage({
        type: 'team_sentiment_update',
        team_id: '1',
        data: teamSentiment,
        timestamp: '2024-01-15T10:00:00Z',
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId('sentiment-1')).toHaveTextContent('0.75')
      expect(screen.getByTestId('mentions-1')).toHaveTextContent('150')
    })
  })

  it('updates game predictions', async () => {
    render(<SentimentDashboard />)

    // Establish connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })

    mockWebSocket = (global.WebSocket as any).mock.instances[0]

    // Send game prediction update
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
      mockWebSocket.simulateMessage({
        type: 'game_prediction_update',
        game_id: '1',
        data: gamePrediction,
        timestamp: '2024-01-15T10:00:00Z',
      })
    })

    await waitFor(() => {
      expect(screen.getByTestId('predictions-count')).toHaveTextContent('1')
    })
  })

  it('handles connection loss and reconnection', async () => {
    render(<SentimentDashboard />)

    // Establish connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })

    mockWebSocket = (global.WebSocket as any).mock.instances[0]

    // Simulate connection loss
    act(() => {
      mockWebSocket.simulateClose()
    })

    await waitFor(() => {
      expect(screen.getByText(/Disconnected/)).toBeInTheDocument()
    })

    // Fast-forward to trigger reconnection attempt
    act(() => {
      vi.advanceTimersByTime(3000)
    })

    // Should create a new WebSocket instance
    expect(MockWebSocket).toHaveBeenCalledTimes(2)

    // Establish new connection
    const newMockWebSocket = (global.WebSocket as any).mock.instances[1]
    
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })
  })

  it('subscribes to team updates when team is selected', async () => {
    render(<SentimentDashboard />)

    // Establish connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })

    mockWebSocket = (global.WebSocket as any).mock.instances[0]
    const sendSpy = vi.spyOn(mockWebSocket, 'send')

    // Select a team
    const teamCard = screen.getByTestId('team-card-1')
    act(() => {
      teamCard.click()
    })

    // Should send subscription message
    expect(sendSpy).toHaveBeenCalledWith(
      JSON.stringify({
        type: 'subscribe',
        subscription: 'team_sentiment',
        team_id: '1',
      })
    )
  })

  it('sends periodic ping messages', async () => {
    render(<SentimentDashboard />)

    // Establish connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })

    mockWebSocket = (global.WebSocket as any).mock.instances[0]
    const sendSpy = vi.spyOn(mockWebSocket, 'send')

    // Fast-forward 30 seconds to trigger ping
    act(() => {
      vi.advanceTimersByTime(30000)
    })

    expect(sendSpy).toHaveBeenCalledWith(
      expect.stringContaining('"type":"ping"')
    )
  })

  it('limits recent sentiments to 10 items', async () => {
    render(<SentimentDashboard />)

    // Establish connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })

    mockWebSocket = (global.WebSocket as any).mock.instances[0]

    // Send 15 sentiment updates
    for (let i = 1; i <= 15; i++) {
      act(() => {
        mockWebSocket.simulateMessage({
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

    await waitFor(() => {
      expect(screen.getByText('Sentiment 15')).toBeInTheDocument()
    })

    // Should only show 10 most recent
    expect(screen.getByTestId('chart-data-count')).toHaveTextContent('10')
    expect(screen.queryByText('Sentiment 1')).not.toBeInTheDocument()
    expect(screen.queryByText('Sentiment 5')).not.toBeInTheDocument()
    expect(screen.getByText('Sentiment 6')).toBeInTheDocument()
  })

  it('handles WebSocket errors gracefully', async () => {
    render(<SentimentDashboard />)

    // Establish connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })

    mockWebSocket = (global.WebSocket as any).mock.instances[0]

    // Simulate WebSocket error
    act(() => {
      mockWebSocket.simulateError()
    })

    await waitFor(() => {
      expect(screen.getByText(/Disconnected/)).toBeInTheDocument()
    })
  })

  it('handles malformed WebSocket messages', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    
    render(<SentimentDashboard />)

    // Establish connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })

    mockWebSocket = (global.WebSocket as any).mock.instances[0]

    // Send malformed message
    act(() => {
      if (mockWebSocket.onmessage) {
        mockWebSocket.onmessage(new MessageEvent('message', { data: 'invalid json' }))
      }
    })

    // Should log error but not crash
    expect(consoleSpy).toHaveBeenCalledWith(
      'Failed to parse WebSocket message:',
      expect.any(Error)
    )

    // Dashboard should still be functional
    expect(screen.getByText('Live')).toBeInTheDocument()

    consoleSpy.mockRestore()
  })

  it('updates connection status indicators', async () => {
    render(<SentimentDashboard />)

    // Initially connecting
    expect(screen.getByText('Connecting...')).toBeInTheDocument()

    // Establish connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })

    mockWebSocket = (global.WebSocket as any).mock.instances[0]

    // Simulate disconnection
    act(() => {
      mockWebSocket.simulateClose()
    })

    await waitFor(() => {
      expect(screen.getByText(/Disconnected/)).toBeInTheDocument()
    })

    // Should show retry button
    expect(screen.getByText('Retry')).toBeInTheDocument()
  })
})