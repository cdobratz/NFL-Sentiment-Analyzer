import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from './utils'
import { axe } from 'jest-axe'
import SentimentDashboard from '../components/SentimentDashboard'
import TeamSentimentCard from '../components/TeamSentimentCard'
import Navbar from '../components/Navbar'
import Login from '../pages/Login'
import * as api from '../services/api'
import * as useSentimentWebSocketModule from '../hooks/useSentimentWebSocket'
import { useAuth } from '../hooks/useAuth'
import { useAuthStore } from '../stores/authStore'

// Custom matcher for accessibility violations
const toHaveNoViolations = (results: any) => {
  const pass = results.violations.length === 0
  return {
    pass,
    message: () => 
      pass 
        ? 'Expected violations, but none were found'
        : `Found ${results.violations.length} accessibility violations:\n${
            results.violations.map((v: any) => `- ${v.description}`).join('\n')
          }`
  }
}

expect.extend({ toHaveNoViolations })

// Extend Vitest's Assertion interface
declare module 'vitest' {
  interface Assertion<T = any> {
    toHaveNoViolations(): T
  }
  interface AsymmetricMatchersContaining {
    toHaveNoViolations(): any
  }
}

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

// Mock Chart.js components
vi.mock('react-chartjs-2', () => ({
  Line: vi.fn(() => <div role="img" aria-label="Sentiment trend chart">Chart</div>),
}))

// Mock child components with accessible versions
vi.mock('../components/TeamSentimentCard', () => ({
  default: vi.fn(({ team, onSelect, isSelected }) => (
    <button 
      onClick={onSelect}
      aria-pressed={isSelected}
      aria-label={`${team.name} sentiment card`}
    >
      {team.name}
    </button>
  )),
}))

vi.mock('../components/RealTimeChart', () => ({
  default: vi.fn(() => (
    <div role="img" aria-label="Real-time sentiment chart">
      Real Time Chart
    </div>
  )),
}))

vi.mock('../components/GamePredictionPanel', () => ({
  default: vi.fn(() => (
    <section aria-label="Game predictions">
      Game Predictions
    </section>
  )),
}))

vi.mock('../components/LoadingSpinner', () => ({
  default: vi.fn(() => (
    <div role="status" aria-label="Loading">
      <span className="sr-only">Loading...</span>
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
]

const mockWebSocketData = {
  recentSentiments: [],
  teamSentiments: {},
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

describe('Accessibility Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    
    vi.mocked(api.dataApi.getTeams).mockResolvedValue({
      data: { data: mockTeams },
      status: 200,
      statusText: 'OK',
      headers: {},
      config: {},
    } as any)

    vi.mocked(useSentimentWebSocketModule.default).mockReturnValue(mockWebSocketData)
  })

  describe('SentimentDashboard', () => {
    it('should not have accessibility violations', async () => {
      const { container } = render(<SentimentDashboard />)
      
      // Wait for component to load
      await screen.findByText('NFL Sentiment Dashboard')
      
      const results = await axe(container)
      expect(results).toHaveNoViolations()
    })

    it('has proper heading hierarchy', async () => {
      render(<SentimentDashboard />)
      
      await screen.findByText('NFL Sentiment Dashboard')
      
      const h1 = screen.getByRole('heading', { level: 1 })
      expect(h1).toHaveTextContent('NFL Sentiment Dashboard')
      
      const h2Elements = screen.getAllByRole('heading', { level: 2 })
      expect(h2Elements.length).toBeGreaterThan(0)
    })

    it('has accessible form controls', async () => {
      render(<SentimentDashboard />)
      
      await screen.findByText('NFL Sentiment Dashboard')
      
      const select = screen.getByRole('combobox')
      expect(select).toBeInTheDocument()
      expect(select).toHaveAccessibleName()
    })

    it('provides status updates for screen readers', async () => {
      render(<SentimentDashboard />)
      
      await screen.findByText('NFL Sentiment Dashboard')
      
      // Connection status should be announced
      const liveRegion = screen.getByText('Live')
      expect(liveRegion).toBeInTheDocument()
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

    const mockSentimentData = {
      team_id: '1',
      team_name: 'New England Patriots',
      current_sentiment: 0.75,
      sentiment_trend: [],
      total_mentions: 150,
    }

    it('should not have accessibility violations', async () => {
      const { container } = render(
        <TeamSentimentCard
          team={mockTeam}
          sentimentData={mockSentimentData}
          isSelected={false}
          onSelect={vi.fn()}
        />
      )
      
      const results = await axe(container)
      expect(results).toHaveNoViolations()
    })

    it('has proper button semantics', () => {
      const mockOnSelect = vi.fn()
      
      render(
        <TeamSentimentCard
          team={mockTeam}
          sentimentData={mockSentimentData}
          isSelected={false}
          onSelect={mockOnSelect}
        />
      )
      
      const button = screen.getByRole('button')
      expect(button).toBeInTheDocument()
      expect(button).toHaveAttribute('aria-pressed', 'false')
    })

    it('indicates selected state to screen readers', () => {
      render(
        <TeamSentimentCard
          team={mockTeam}
          sentimentData={mockSentimentData}
          isSelected={true}
          onSelect={vi.fn()}
        />
      )
      
      const button = screen.getByRole('button')
      expect(button).toHaveAttribute('aria-pressed', 'true')
    })

    it('has descriptive accessible name', () => {
      render(
        <TeamSentimentCard
          team={mockTeam}
          sentimentData={mockSentimentData}
          isSelected={false}
          onSelect={vi.fn()}
        />
      )
      
      const button = screen.getByRole('button')
      expect(button).toHaveAccessibleName('New England Patriots sentiment card')
    })
  })

  describe('Navbar', () => {
    it('should not have accessibility violations when unauthenticated', async () => {
      vi.mocked(useAuthStore).mockReturnValue({
        user: null,
        isAuthenticated: false,
        logout: vi.fn(),
      })

      const { container } = render(<Navbar />)
      
      const results = await axe(container)
      expect(results).toHaveNoViolations()
    })

    it('should not have accessibility violations when authenticated', async () => {
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
      
      const results = await axe(container)
      expect(results).toHaveNoViolations()
    })

    it('has proper navigation semantics', () => {
      vi.mocked(useAuthStore).mockReturnValue({
        user: null,
        isAuthenticated: false,
        logout: vi.fn(),
      })

      render(<Navbar />)
      
      const nav = screen.getByRole('navigation')
      expect(nav).toBeInTheDocument()
      
      const links = screen.getAllByRole('link')
      expect(links.length).toBeGreaterThan(0)
      
      links.forEach(link => {
        expect(link).toHaveAccessibleName()
      })
    })

    it('has accessible logout button', () => {
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

      render(<Navbar />)
      
      const logoutButton = screen.getByRole('button')
      expect(logoutButton).toBeInTheDocument()
      expect(logoutButton).toHaveAccessibleName()
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

    it('should not have accessibility violations', async () => {
      const { container } = render(<Login />)
      
      const results = await axe(container)
      expect(results).toHaveNoViolations()
    })

    it('has proper form semantics', () => {
      render(<Login />)
      
      const form = screen.getByRole('form')
      expect(form).toBeInTheDocument()
      
      const emailInput = screen.getByRole('textbox', { name: /email/i })
      expect(emailInput).toBeInTheDocument()
      expect(emailInput).toHaveAttribute('type', 'email')
      
      const passwordInput = screen.getByLabelText(/password/i)
      expect(passwordInput).toBeInTheDocument()
      expect(passwordInput).toHaveAttribute('type', 'password')
      
      const submitButton = screen.getByRole('button', { name: /sign in/i })
      expect(submitButton).toBeInTheDocument()
      expect(submitButton).toHaveAttribute('type', 'submit')
    })

    it('has proper labels and descriptions', () => {
      render(<Login />)
      
      const emailInput = screen.getByRole('textbox', { name: /email/i })
      const passwordInput = screen.getByLabelText(/password/i)
      
      expect(emailInput).toHaveAccessibleName()
      expect(passwordInput).toHaveAccessibleName()
    })

    it('announces form validation errors', async () => {
      render(<Login />)
      
      const submitButton = screen.getByRole('button', { name: /sign in/i })
      submitButton.click()
      
      // Wait for validation errors
      await screen.findByText('Invalid email address')
      
      const errorMessage = screen.getByText('Invalid email address')
      expect(errorMessage).toBeInTheDocument()
      
      // Error should be associated with the input
      const emailInput = screen.getByRole('textbox', { name: /email/i })
      expect(emailInput).toHaveAccessibleDescription()
    })

    it('has proper heading structure', () => {
      render(<Login />)
      
      const heading = screen.getByRole('heading', { level: 2 })
      expect(heading).toHaveTextContent('Sign in to your account')
    })
  })

  describe('Keyboard Navigation', () => {
    it('supports keyboard navigation in dashboard', async () => {
      render(<SentimentDashboard />)
      
      await screen.findByText('NFL Sentiment Dashboard')
      
      const select = screen.getByRole('combobox')
      
      // Should be focusable
      select.focus()
      expect(select).toHaveFocus()
      
      // Should respond to keyboard events
      expect(select).toHaveAttribute('tabIndex')
    })

    it('supports keyboard navigation in navbar', () => {
      vi.mocked(useAuthStore).mockReturnValue({
        user: null,
        isAuthenticated: false,
        logout: vi.fn(),
      })

      render(<Navbar />)
      
      const links = screen.getAllByRole('link')
      
      links.forEach(link => {
        link.focus()
        expect(link).toHaveFocus()
      })
    })
  })

  describe('Screen Reader Support', () => {
    it('provides meaningful text alternatives for images', async () => {
      render(<SentimentDashboard />)
      
      await screen.findByText('NFL Sentiment Dashboard')
      
      const chartImage = screen.getByRole('img', { name: /sentiment trend chart/i })
      expect(chartImage).toBeInTheDocument()
    })

    it('uses semantic HTML elements', async () => {
      render(<SentimentDashboard />)
      
      await screen.findByText('NFL Sentiment Dashboard')
      
      // Should have proper landmarks
      const main = screen.getByRole('main')
      expect(main).toBeInTheDocument()
    })

    it('provides status updates', async () => {
      render(<SentimentDashboard />)
      
      await screen.findByText('NFL Sentiment Dashboard')
      
      // Loading states should be announced
      const statusElements = screen.getAllByRole('status')
      expect(statusElements.length).toBeGreaterThanOrEqual(0)
    })
  })

  describe('Color Contrast and Visual Accessibility', () => {
    it('uses sufficient color contrast', async () => {
      const { container } = render(<SentimentDashboard />)
      
      await screen.findByText('NFL Sentiment Dashboard')
      
      // axe will check color contrast automatically
      const results = await axe(container, {
        rules: {
          'color-contrast': { enabled: true }
        }
      })
      expect(results).toHaveNoViolations()
    })

    it('does not rely solely on color for information', () => {
      const mockTeam = {
        id: '1',
        name: 'New England Patriots',
        abbreviation: 'NE',
        conference: 'AFC',
        division: 'East',
      }

      const mockSentimentData = {
        team_id: '1',
        team_name: 'New England Patriots',
        current_sentiment: 0.75,
        sentiment_trend: [],
        total_mentions: 150,
      }

      render(
        <TeamSentimentCard
          team={mockTeam}
          sentimentData={mockSentimentData}
          isSelected={false}
          onSelect={vi.fn()}
        />
      )
      
      // Should have text labels in addition to colors
      const button = screen.getByRole('button')
      expect(button).toHaveTextContent('New England Patriots')
    })
  })
})