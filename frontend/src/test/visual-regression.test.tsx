import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render } from './utils'
import SentimentDashboard from '../components/SentimentDashboard'
import TeamSentimentCard from '../components/TeamSentimentCard'
import RealTimeChart from '../components/RealTimeChart'
import GamePredictionPanel from '../components/GamePredictionPanel'
import Navbar from '../components/Navbar'
import Login from '../pages/Login'
import * as api from '../services/api'
import * as useSentimentWebSocketModule from '../hooks/useSentimentWebSocket'
import { useAuth } from '../hooks/useAuth'
import { useAuthStore } from '../stores/authStore'

// Mock dependencies
vi.mock('../services/api', () => ({
  dataApi: {
    getTeams: vi.fn(),
    getGames: vi.fn(),
  },
}))

vi.mock('../hooks/useSentimentWebSocket')
vi.mock('../hooks/useAuth')
vi.mock('../stores/authStore')

// Mock Chart.js components for consistent rendering
vi.mock('react-chartjs-2', () => ({
  Line: vi.fn(() => (
    <div 
      data-testid="mock-chart" 
      style={{ width: '100%', height: '300px', backgroundColor: '#f3f4f6' }}
    >
      Chart Placeholder
    </div>
  )),
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

const mockSentimentData = [
  {
    id: '1',
    text: 'Great game!',
    sentiment: 'positive' as const,
    confidence: 0.9,
    team_id: '1',
    source: 'twitter',
    timestamp: '2024-01-15T10:00:00Z',
  },
  {
    id: '2',
    text: 'Bad play',
    sentiment: 'negative' as const,
    confidence: 0.8,
    team_id: '2',
    source: 'twitter',
    timestamp: '2024-01-15T10:05:00Z',
  },
]

const mockWebSocketData = {
  recentSentiments: mockSentimentData,
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

const mockPredictions = [
  {
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
  },
]

/**
 * Visual regression tests to ensure UI consistency
 * These tests capture the rendered output and can be used with visual testing tools
 */
describe('Visual Regression Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    
    vi.mocked(api.dataApi.getTeams).mockResolvedValue({
      data: { data: mockTeams },
      status: 200,
      statusText: 'OK',
      headers: {},
      config: {},
    } as any)

    vi.mocked(api.dataApi.getGames).mockResolvedValue({
      data: { data: [] },
      status: 200,
      statusText: 'OK',
      headers: {},
      config: {},
    } as any)

    vi.mocked(useSentimentWebSocketModule.default).mockReturnValue(mockWebSocketData)
  })

  describe('SentimentDashboard', () => {
    it('renders dashboard with default state', async () => {
      const { container } = render(<SentimentDashboard />)
      
      // Wait for component to stabilize
      await new Promise(resolve => setTimeout(resolve, 100))
      
      expect(container.firstChild).toMatchSnapshot('dashboard-default-state')
    })

    it('renders dashboard with loading state', async () => {
      vi.mocked(api.dataApi.getTeams).mockImplementation(
        () => new Promise(() => {}) as any // Never resolves
      )

      const { container } = render(<SentimentDashboard />)
      
      expect(container.firstChild).toMatchSnapshot('dashboard-loading-state')
    })

    it('renders dashboard with error state', async () => {
      vi.mocked(api.dataApi.getTeams).mockRejectedValue(new Error('API Error'))

      const { container } = render(<SentimentDashboard />)
      
      // Wait for error to appear
      await new Promise(resolve => setTimeout(resolve, 100))
      
      expect(container.firstChild).toMatchSnapshot('dashboard-error-state')
    })

    it('renders dashboard with disconnected WebSocket', async () => {
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

      const { container } = render(<SentimentDashboard />)
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      expect(container.firstChild).toMatchSnapshot('dashboard-disconnected-state')
    })
  })

  describe('TeamSentimentCard', () => {
    const mockTeam = {
      id: '1',
      name: 'New England Patriots',
      abbreviation: 'NE',
      conference: 'AFC',
      division: 'East',
    }

    it('renders card with positive sentiment', () => {
      const positiveSentimentData = {
        team_id: '1',
        team_name: 'New England Patriots',
        current_sentiment: 0.75,
        sentiment_trend: [
          { timestamp: '2024-01-15T09:00:00Z', sentiment: 0.7 },
          { timestamp: '2024-01-15T10:00:00Z', sentiment: 0.75 },
        ],
        total_mentions: 150,
      }

      const { container } = render(
        <TeamSentimentCard
          team={mockTeam}
          sentimentData={positiveSentimentData}
          isSelected={false}
          onSelect={vi.fn()}
        />
      )
      
      expect(container.firstChild).toMatchSnapshot('team-card-positive-sentiment')
    })

    it('renders card with negative sentiment', () => {
      const negativeSentimentData = {
        team_id: '1',
        team_name: 'New England Patriots',
        current_sentiment: -0.6,
        sentiment_trend: [
          { timestamp: '2024-01-15T09:00:00Z', sentiment: -0.5 },
          { timestamp: '2024-01-15T10:00:00Z', sentiment: -0.6 },
        ],
        total_mentions: 75,
      }

      const { container } = render(
        <TeamSentimentCard
          team={mockTeam}
          sentimentData={negativeSentimentData}
          isSelected={false}
          onSelect={vi.fn()}
        />
      )
      
      expect(container.firstChild).toMatchSnapshot('team-card-negative-sentiment')
    })

    it('renders card in selected state', () => {
      const { container } = render(
        <TeamSentimentCard
          team={mockTeam}
          sentimentData={mockWebSocketData.teamSentiments['1']}
          isSelected={true}
          onSelect={vi.fn()}
        />
      )
      
      expect(container.firstChild).toMatchSnapshot('team-card-selected-state')
    })

    it('renders card with loading state', () => {
      const { container } = render(
        <TeamSentimentCard
          team={mockTeam}
          sentimentData={undefined}
          isSelected={false}
          onSelect={vi.fn()}
        />
      )
      
      expect(container.firstChild).toMatchSnapshot('team-card-loading-state')
    })
  })

  describe('RealTimeChart', () => {
    it('renders chart with data', () => {
      const { container } = render(
        <RealTimeChart data={mockSentimentData} />
      )
      
      expect(container.firstChild).toMatchSnapshot('chart-with-data')
    })

    it('renders chart with empty state', () => {
      const { container } = render(
        <RealTimeChart data={[]} />
      )
      
      expect(container.firstChild).toMatchSnapshot('chart-empty-state')
    })

    it('renders chart for mobile', () => {
      const { container } = render(
        <RealTimeChart 
          data={mockSentimentData} 
          isMobile={true}
          height={250}
        />
      )
      
      expect(container.firstChild).toMatchSnapshot('chart-mobile-view')
    })

    it('renders chart with selected team', () => {
      const { container } = render(
        <RealTimeChart 
          data={mockSentimentData} 
          selectedTeam="1"
        />
      )
      
      expect(container.firstChild).toMatchSnapshot('chart-selected-team')
    })
  })

  describe('GamePredictionPanel', () => {
    it('renders panel with upcoming games', async () => {
      const { container } = render(
        <GamePredictionPanel predictions={mockPredictions} />
      )
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      expect(container.firstChild).toMatchSnapshot('prediction-panel-upcoming-games')
    })

    it('renders panel with predictions', async () => {
      const { container } = render(
        <GamePredictionPanel predictions={mockPredictions} />
      )
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      expect(container.firstChild).toMatchSnapshot('prediction-panel-predictions')
    })

    it('renders panel with empty state', async () => {
      vi.mocked(api.dataApi.getGames).mockResolvedValue({
        data: { data: [] },
        status: 200,
        statusText: 'OK',
        headers: {},
        config: {},
      } as any)

      const { container } = render(
        <GamePredictionPanel predictions={[]} />
      )
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      expect(container.firstChild).toMatchSnapshot('prediction-panel-empty-state')
    })
  })

  describe('Navbar', () => {
    it('renders navbar when unauthenticated', () => {
      vi.mocked(useAuthStore).mockReturnValue({
        user: null,
        isAuthenticated: false,
        logout: vi.fn(),
      })

      const { container } = render(<Navbar />)
      
      expect(container.firstChild).toMatchSnapshot('navbar-unauthenticated')
    })

    it('renders navbar when authenticated as user', () => {
      vi.mocked(useAuthStore).mockReturnValue({
        user: {
          id: '1',
          username: 'testuser',
          email: 'test@example.com',
          role: 'USER',
        },
        isAuthenticated: true,
        logout: vi.fn(),
      })

      const { container } = render(<Navbar />)
      
      expect(container.firstChild).toMatchSnapshot('navbar-authenticated-user')
    })

    it('renders navbar when authenticated as admin', () => {
      vi.mocked(useAuthStore).mockReturnValue({
        user: {
          id: '1',
          username: 'admin',
          email: 'admin@example.com',
          role: 'admin',
        },
        isAuthenticated: true,
        logout: vi.fn(),
      })

      const { container } = render(<Navbar />)
      
      expect(container.firstChild).toMatchSnapshot('navbar-authenticated-admin')
    })
  })

  describe('Login', () => {
    beforeEach(() => {
      vi.mocked(useAuth).mockReturnValue({
        user: null,
        token: null,
        isLoading: false,
        isAuthenticated: false,
        login: vi.fn(),
        register: vi.fn(),
        logout: vi.fn(),
        checkAuth: vi.fn(),
        refreshToken: vi.fn(),
        updateProfile: vi.fn(),
        changePassword: vi.fn(),
        isAdmin: false,
        isUser: false,
      })
    })

    it('renders login form', () => {
      const { container } = render(<Login />)
      
      expect(container.firstChild).toMatchSnapshot('login-form')
    })

    it('renders login form with validation errors', async () => {
      const { container, getByRole } = render(<Login />)
      
      // Trigger validation
      const submitButton = getByRole('button', { name: /sign in/i })
      submitButton.click()
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      expect(container.firstChild).toMatchSnapshot('login-form-with-errors')
    })

    it('renders login form in loading state', () => {
      vi.mocked(useAuth).mockReturnValue({
        user: null,
        token: null,
        isLoading: true,
        isAuthenticated: false,
        login: vi.fn(),
        register: vi.fn(),
        logout: vi.fn(),
        checkAuth: vi.fn(),
        refreshToken: vi.fn(),
        updateProfile: vi.fn(),
        changePassword: vi.fn(),
        isAdmin: false,
        isUser: false,
      })

      const { container } = render(<Login />)
      
      expect(container.firstChild).toMatchSnapshot('login-form-loading')
    })
  })

  describe('Responsive Design', () => {
    it('renders dashboard on mobile viewport', async () => {
      // Mock mobile viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      })

      const { container } = render(<SentimentDashboard />)
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      expect(container.firstChild).toMatchSnapshot('dashboard-mobile-viewport')
    })

    it('renders dashboard on tablet viewport', async () => {
      // Mock tablet viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      })

      const { container } = render(<SentimentDashboard />)
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      expect(container.firstChild).toMatchSnapshot('dashboard-tablet-viewport')
    })

    it('renders dashboard on desktop viewport', async () => {
      // Mock desktop viewport
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 1024,
      })

      const { container } = render(<SentimentDashboard />)
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      expect(container.firstChild).toMatchSnapshot('dashboard-desktop-viewport')
    })
  })

  describe('Dark Mode Support', () => {
    it('renders components with dark mode classes', async () => {
      // Mock dark mode preference
      Object.defineProperty(window, 'matchMedia', {
        writable: true,
        value: vi.fn().mockImplementation(query => ({
          matches: query === '(prefers-color-scheme: dark)',
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn(),
        })),
      })

      const { container } = render(<SentimentDashboard />)
      
      await new Promise(resolve => setTimeout(resolve, 100))
      
      expect(container.firstChild).toMatchSnapshot('dashboard-dark-mode')
    })
  })
})

/**
 * Utility function to create consistent test data
 */
export const createTestData = {
  team: (overrides = {}) => ({
    id: '1',
    name: 'Test Team',
    abbreviation: 'TT',
    conference: 'AFC',
    division: 'North',
    ...overrides,
  }),

  sentimentData: (overrides = {}) => ({
    team_id: '1',
    team_name: 'Test Team',
    current_sentiment: 0.5,
    sentiment_trend: [
      { timestamp: '2024-01-15T09:00:00Z', sentiment: 0.4 },
      { timestamp: '2024-01-15T10:00:00Z', sentiment: 0.5 },
    ],
    total_mentions: 100,
    ...overrides,
  }),

  sentiment: (overrides = {}) => ({
    id: '1',
    text: 'Test sentiment',
    sentiment: 'neutral' as const,
    confidence: 0.8,
    team_id: '1',
    source: 'twitter',
    timestamp: '2024-01-15T10:00:00Z',
    ...overrides,
  }),

  prediction: (overrides = {}) => ({
    game_id: '1',
    home_team: 'Home Team',
    away_team: 'Away Team',
    prediction: {
      home_win_probability: 0.5,
      away_win_probability: 0.5,
      sentiment_factor: 0.7,
    },
    betting_lines: {
      spread: 0,
      over_under: 45,
      moneyline: {
        home: 100,
        away: -100,
      },
    },
    ...overrides,
  }),
}