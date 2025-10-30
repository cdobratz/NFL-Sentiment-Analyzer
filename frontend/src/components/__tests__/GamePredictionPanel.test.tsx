import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '../../test/utils'
import GamePredictionPanel from '../GamePredictionPanel'
import * as api from '../../services/api'

// Mock the API
vi.mock('../../services/api', () => ({
  dataApi: {
    getGames: vi.fn(),
  },
}))

// Mock LoadingSpinner
vi.mock('../LoadingSpinner', () => ({
  default: vi.fn(() => <div data-testid="loading-spinner">Loading...</div>),
}))

const mockGames = [
  {
    id: '1',
    home_team: {
      id: '1',
      name: 'New England Patriots',
      abbreviation: 'NE',
    },
    away_team: {
      id: '2',
      name: 'Dallas Cowboys',
      abbreviation: 'DAL',
    },
    game_date: '2024-01-15T20:00:00Z',
    week: 1,
    status: 'scheduled',
  },
  {
    id: '2',
    home_team: {
      id: '3',
      name: 'Green Bay Packers',
      abbreviation: 'GB',
    },
    away_team: {
      id: '4',
      name: 'Chicago Bears',
      abbreviation: 'CHI',
    },
    game_date: '2024-01-16T21:00:00Z',
    week: 1,
    status: 'scheduled',
  },
]

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
  {
    game_id: '2',
    home_team: 'Green Bay Packers',
    away_team: 'Chicago Bears',
    prediction: {
      home_win_probability: 0.45,
      away_win_probability: 0.55,
      sentiment_factor: 0.6,
    },
    betting_lines: {
      spread: 2.5,
      over_under: 42.0,
      moneyline: {
        home: 110,
        away: -130,
      },
    },
  },
]

describe('GamePredictionPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    
    // Mock successful API response
    vi.mocked(api.dataApi.getGames).mockResolvedValue({
      data: { data: mockGames },
      status: 200,
      statusText: 'OK',
      headers: {},
      config: {},
    } as any)
  })

  it('renders with upcoming games tab active by default', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} />)

    expect(screen.getByText('Games & Predictions')).toBeInTheDocument()
    expect(screen.getByText('Upcoming Games')).toBeInTheDocument()
    expect(screen.getByText('Predictions')).toBeInTheDocument()
    
    // Upcoming Games tab should be active
    const upcomingTab = screen.getByText('Upcoming Games')
    expect(upcomingTab.closest('button')).toHaveClass('bg-white', 'text-gray-900')
  })

  it('switches between tabs correctly', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} />)

    const predictionsTab = screen.getByText('Predictions')
    fireEvent.click(predictionsTab)

    expect(predictionsTab.closest('button')).toHaveClass('bg-white', 'text-gray-900')
  })

  it('displays upcoming games correctly', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} />)

    await waitFor(() => {
      expect(screen.getByText('New England Patriots')).toBeInTheDocument()
      expect(screen.getByText('Dallas Cowboys')).toBeInTheDocument()
      expect(screen.getByText('Week 1')).toBeInTheDocument()
    })
  })

  it('displays game predictions correctly', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} />)

    const predictionsTab = screen.getByText('Predictions')
    fireEvent.click(predictionsTab)

    await waitFor(() => {
      expect(screen.getByText('65.0%')).toBeInTheDocument() // Home win probability
      expect(screen.getByText('35.0%')).toBeInTheDocument() // Away win probability
      expect(screen.getByText('80.0%')).toBeInTheDocument() // Sentiment factor
    })
  })

  it('displays betting lines correctly', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} />)

    const predictionsTab = screen.getByText('Predictions')
    fireEvent.click(predictionsTab)

    await waitFor(() => {
      expect(screen.getByText('Spread: -3.5')).toBeInTheDocument()
      expect(screen.getByText('O/U: 45.5')).toBeInTheDocument()
      expect(screen.getByText('ML: -150 / +130')).toBeInTheDocument()
    })
  })

  it('filters predictions by selected team', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} selectedTeam="1" />)

    const predictionsTab = screen.getByText('Predictions')
    fireEvent.click(predictionsTab)

    await waitFor(() => {
      // Should only show predictions involving team 1
      expect(screen.getByText('New England Patriots')).toBeInTheDocument()
      expect(screen.queryByText('Green Bay Packers')).not.toBeInTheDocument()
    })
  })

  it('shows loading state for games', async () => {
    // Mock loading state
    vi.mocked(api.dataApi.getGames).mockImplementation(
      () => new Promise(() => {}) as any // Never resolves
    )

    render(<GamePredictionPanel predictions={mockPredictions} />)

    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument()
  })

  it('shows empty state when no games', async () => {
    vi.mocked(api.dataApi.getGames).mockResolvedValue({
      data: { data: [] },
      status: 200,
      statusText: 'OK',
      headers: {},
      config: {},
    } as any)

    render(<GamePredictionPanel predictions={mockPredictions} />)

    await waitFor(() => {
      expect(screen.getByText('No upcoming games found')).toBeInTheDocument()
    })
  })

  it('shows empty state when no predictions', async () => {
    render(<GamePredictionPanel predictions={[]} />)

    const predictionsTab = screen.getByText('Predictions')
    fireEvent.click(predictionsTab)

    expect(screen.getByText('No predictions available')).toBeInTheDocument()
    expect(screen.getByText('Predictions will appear as games approach')).toBeInTheDocument()
  })

  it('shows team-specific empty state for games', async () => {
    vi.mocked(api.dataApi.getGames).mockResolvedValue({
      data: { data: [] },
      status: 200,
      statusText: 'OK',
      headers: {},
      config: {},
    } as any)

    render(<GamePredictionPanel predictions={mockPredictions} selectedTeam="1" />)

    await waitFor(() => {
      expect(screen.getByText('No upcoming games found')).toBeInTheDocument()
      expect(screen.getByText('No games scheduled for selected team')).toBeInTheDocument()
    })
  })

  it('formats dates correctly', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} />)

    await waitFor(() => {
      // Should show formatted date and time
      expect(screen.getByText(/Jan 15/)).toBeInTheDocument()
      expect(screen.getByText(/8:00 PM/)).toBeInTheDocument()
    })
  })

  it('formats odds correctly', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} />)

    const predictionsTab = screen.getByText('Predictions')
    fireEvent.click(predictionsTab)

    await waitFor(() => {
      // Negative odds should show as-is, positive odds should have + prefix
      expect(screen.getByText('ML: -150 / +130')).toBeInTheDocument()
    })
  })

  it('applies correct win probability colors', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} />)

    const predictionsTab = screen.getByText('Predictions')
    fireEvent.click(predictionsTab)

    await waitFor(() => {
      const highProbability = screen.getByText('65.0%')
      expect(highProbability.closest('div')).toHaveClass('text-green-600', 'bg-green-50')

      const lowProbability = screen.getByText('35.0%')
      expect(lowProbability.closest('div')).toHaveClass('text-red-600', 'bg-red-50')
    })
  })

  it('shows correct title for selected team', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} selectedTeam="1" />)

    expect(screen.getByText('Team Games & Predictions')).toBeInTheDocument()
  })

  it('shows correct title for all teams', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} />)

    expect(screen.getByText('Games & Predictions')).toBeInTheDocument()
  })

  it('handles predictions without betting lines', async () => {
    const predictionsWithoutLines = [
      {
        ...mockPredictions[0],
        betting_lines: undefined,
      },
    ]

    render(<GamePredictionPanel predictions={predictionsWithoutLines} />)

    const predictionsTab = screen.getByText('Predictions')
    fireEvent.click(predictionsTab)

    await waitFor(() => {
      expect(screen.getByText('65.0%')).toBeInTheDocument()
      // Should not show betting lines section
      expect(screen.queryByText('Spread:')).not.toBeInTheDocument()
    })
  })

  it('displays team abbreviations correctly', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} />)

    await waitFor(() => {
      // Should show first 2 characters of abbreviation
      expect(screen.getByText('DA')).toBeInTheDocument() // DAL -> DA
      expect(screen.getByText('NE')).toBeInTheDocument() // NE -> NE
    })
  })

  it('calls API with correct parameters for selected team', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} selectedTeam="1" />)

    expect(api.dataApi.getGames).toHaveBeenCalledWith({
      status: 'scheduled',
      limit: 10,
      team_id: '1',
    })
  })

  it('calls API with correct parameters for all teams', async () => {
    render(<GamePredictionPanel predictions={mockPredictions} />)

    expect(api.dataApi.getGames).toHaveBeenCalledWith({
      status: 'scheduled',
      limit: 10,
    })
  })
})