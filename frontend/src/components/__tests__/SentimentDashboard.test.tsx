import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '../../test/utils'
import SentimentDashboard from '../SentimentDashboard'
import * as api from '../../services/api'
import * as useSentimentWebSocketModule from '../../hooks/useSentimentWebSocket'

// Mock the API
vi.mock('../../services/api', () => ({
  dataApi: {
    getTeams: vi.fn(),
  },
}))

// Mock the WebSocket hook
vi.mock('../../hooks/useSentimentWebSocket')

// Mock Chart.js components
vi.mock('react-chartjs-2', () => ({
  Line: vi.fn(() => <div data-testid="mock-chart">Chart</div>),
}))

// Mock child components
vi.mock('../TeamSentimentCard', () => ({
  default: vi.fn(({ team, onSelect, isSelected }) => (
    <div 
      data-testid={`team-card-${team.id}`}
      onClick={onSelect}
      className={isSelected ? 'selected' : ''}
    >
      {team.name}
    </div>
  )),
}))

vi.mock('../RealTimeChart', () => ({
  default: vi.fn(() => <div data-testid="real-time-chart">Real Time Chart</div>),
}))

vi.mock('../GamePredictionPanel', () => ({
  default: vi.fn(() => <div data-testid="game-prediction-panel">Game Predictions</div>),
}))

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

const mockWebSocketData = {
  recentSentiments: [
    {
      id: '1',
      text: 'Great game!',
      sentiment: 'positive' as const,
      confidence: 0.9,
      team_id: '1',
      source: 'twitter',
      timestamp: '2024-01-15T10:00:00Z',
    },
  ],
  teamSentiments: {
    '1': {
      team_id: '1',
      team_name: 'New England Patriots',
      current_sentiment: 0.75,
      sentiment_trend: [
        { timestamp: '2024-01-15T09:00:00Z', sentiment: 0.7 },
        { timestamp: '2024-01-15T10:00:00Z', sentiment: 0.75 },
      ],
      total_mentions: 150,
    },
  },
  gamePredictions: {},
  connectionStatus: {
    isConnected: true,
    isConnecting: false,
    error: null,
    reconnectAttempts: 0,
  },
  subscribeToTeam: vi.fn(),
  subscribeToGame: vi.fn(),
  reconnect: vi.fn(),
  disconnect: vi.fn(),
}

describe('SentimentDashboard', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    
    // Mock successful API response
    vi.mocked(api.dataApi.getTeams).mockResolvedValue({
      data: { data: mockTeams },
    })

    // Mock WebSocket hook
    vi.mocked(useSentimentWebSocketModule.default).mockReturnValue(mockWebSocketData)
  })

  it('renders dashboard with loading state initially', async () => {
    // Mock loading state
    vi.mocked(api.dataApi.getTeams).mockImplementation(
      () => new Promise(() => {}) // Never resolves
    )

    render(<SentimentDashboard />)

    expect(screen.getByRole('generic')).toBeInTheDocument()
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument()
  })

  it('renders dashboard with teams data', async () => {
    render(<SentimentDashboard />)

    await waitFor(() => {
      expect(screen.getByText('NFL Sentiment Dashboard')).toBeInTheDocument()
    })

    expect(screen.getByText('Real-time sentiment analysis and betting insights')).toBeInTheDocument()
    expect(screen.getByTestId('real-time-chart')).toBeInTheDocument()
    expect(screen.getByTestId('game-prediction-panel')).toBeInTheDocument()
  })

  it('displays connection status correctly', async () => {
    render(<SentimentDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Live')).toBeInTheDocument()
    })
  })

  it('shows disconnected status when WebSocket is not connected', async () => {
    const disconnectedWebSocketData = {
      ...mockWebSocketData,
      connectionStatus: {
        isConnected: false,
        isConnecting: false,
        error: 'Connection failed',
        reconnectAttempts: 2,
      },
    }

    vi.mocked(useSentimentWebSocketModule.default).mockReturnValue(disconnectedWebSocketData)

    render(<SentimentDashboard />)

    await waitFor(() => {
      expect(screen.getByText(/Disconnected/)).toBeInTheDocument()
      expect(screen.getByText('Retry')).toBeInTheDocument()
    })
  })

  it('handles conference filter changes', async () => {
    render(<SentimentDashboard />)

    await waitFor(() => {
      expect(screen.getByDisplayValue('All Conferences')).toBeInTheDocument()
    })

    const select = screen.getByDisplayValue('All Conferences')
    fireEvent.change(select, { target: { value: 'AFC' } })

    expect(select).toHaveValue('AFC')
  })

  it('handles team selection', async () => {
    render(<SentimentDashboard />)

    await waitFor(() => {
      expect(screen.getByTestId('team-card-1')).toBeInTheDocument()
    })

    const teamCard = screen.getByTestId('team-card-1')
    fireEvent.click(teamCard)

    expect(mockWebSocketData.subscribeToTeam).toHaveBeenCalledWith('1')
  })

  it('displays recent sentiment activity', async () => {
    render(<SentimentDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Recent Sentiment Activity')).toBeInTheDocument()
      expect(screen.getByText('Great game!')).toBeInTheDocument()
    })
  })

  it('shows empty state when no recent sentiments', async () => {
    const emptyWebSocketData = {
      ...mockWebSocketData,
      recentSentiments: [],
    }

    vi.mocked(useSentimentWebSocketModule.default).mockReturnValue(emptyWebSocketData)

    render(<SentimentDashboard />)

    await waitFor(() => {
      expect(screen.getByText('No recent sentiment data available.')).toBeInTheDocument()
    })
  })

  it('handles API error gracefully', async () => {
    vi.mocked(api.dataApi.getTeams).mockRejectedValue(new Error('API Error'))

    render(<SentimentDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Error Loading Data')).toBeInTheDocument()
      expect(screen.getByText('Failed to load teams data. Please try again.')).toBeInTheDocument()
    })
  })

  it('handles reconnect button click', async () => {
    const disconnectedWebSocketData = {
      ...mockWebSocketData,
      connectionStatus: {
        isConnected: false,
        isConnecting: false,
        error: 'Connection failed',
        reconnectAttempts: 1,
      },
    }

    vi.mocked(useSentimentWebSocketModule.default).mockReturnValue(disconnectedWebSocketData)

    render(<SentimentDashboard />)

    await waitFor(() => {
      const retryButton = screen.getByText('Retry')
      fireEvent.click(retryButton)
      expect(mockWebSocketData.reconnect).toHaveBeenCalled()
    })
  })

  it('filters teams by conference correctly', async () => {
    render(<SentimentDashboard />)

    await waitFor(() => {
      expect(screen.getByTestId('team-card-1')).toBeInTheDocument()
      expect(screen.getByTestId('team-card-2')).toBeInTheDocument()
    })

    const select = screen.getByDisplayValue('All Conferences')
    fireEvent.change(select, { target: { value: 'AFC' } })

    // Should only show AFC teams (team 1)
    expect(screen.getByTestId('team-card-1')).toBeInTheDocument()
    expect(screen.queryByTestId('team-card-2')).not.toBeInTheDocument()
  })

  it('shows connecting status', async () => {
    const connectingWebSocketData = {
      ...mockWebSocketData,
      connectionStatus: {
        isConnected: false,
        isConnecting: true,
        error: null,
        reconnectAttempts: 0,
      },
    }

    vi.mocked(useSentimentWebSocketModule.default).mockReturnValue(connectingWebSocketData)

    render(<SentimentDashboard />)

    await waitFor(() => {
      expect(screen.getByText('Connecting...')).toBeInTheDocument()
    })
  })
})