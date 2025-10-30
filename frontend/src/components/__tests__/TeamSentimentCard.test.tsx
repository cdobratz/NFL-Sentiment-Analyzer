import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '../../test/utils'
import TeamSentimentCard from '../TeamSentimentCard'

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
  sentiment_trend: [
    { timestamp: '2024-01-15T09:00:00Z', sentiment: 0.7 },
    { timestamp: '2024-01-15T10:00:00Z', sentiment: 0.75 },
  ],
  total_mentions: 150,
}

describe('TeamSentimentCard', () => {
  const mockOnSelect = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders team information correctly', () => {
    render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={mockSentimentData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    expect(screen.getByText('New England Patriots')).toBeInTheDocument()
    expect(screen.getByText('AFC • East')).toBeInTheDocument()
    expect(screen.getByText('NE')).toBeInTheDocument()
  })

  it('displays positive sentiment correctly', () => {
    render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={mockSentimentData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    expect(screen.getByText('87.5%')).toBeInTheDocument() // (0.75 + 1) / 2 * 100
    expect(screen.getByText('Positive')).toBeInTheDocument()
    expect(screen.getByText('150 mentions')).toBeInTheDocument()
  })

  it('displays negative sentiment correctly', () => {
    const negativeSentimentData = {
      ...mockSentimentData,
      current_sentiment: -0.6,
    }

    render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={negativeSentimentData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    expect(screen.getByText('20.0%')).toBeInTheDocument() // (-0.6 + 1) / 2 * 100
    expect(screen.getByText('Negative')).toBeInTheDocument()
  })

  it('displays neutral sentiment correctly', () => {
    const neutralSentimentData = {
      ...mockSentimentData,
      current_sentiment: 0.05, // Within neutral range
    }

    render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={neutralSentimentData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    expect(screen.getByText('52.5%')).toBeInTheDocument() // (0.05 + 1) / 2 * 100
    expect(screen.getByText('Neutral')).toBeInTheDocument()
  })

  it('shows trend direction correctly', () => {
    const risingTrendData = {
      ...mockSentimentData,
      sentiment_trend: [
        { timestamp: '2024-01-15T09:00:00Z', sentiment: 0.5 },
        { timestamp: '2024-01-15T10:00:00Z', sentiment: 0.75 }, // Rising trend
      ],
    }

    render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={risingTrendData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    expect(screen.getByText('Rising')).toBeInTheDocument()
  })

  it('shows falling trend correctly', () => {
    const fallingTrendData = {
      ...mockSentimentData,
      sentiment_trend: [
        { timestamp: '2024-01-15T09:00:00Z', sentiment: 0.8 },
        { timestamp: '2024-01-15T10:00:00Z', sentiment: 0.5 }, // Falling trend
      ],
    }

    render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={fallingTrendData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    expect(screen.getByText('Falling')).toBeInTheDocument()
  })

  it('shows stable trend for small changes', () => {
    const stableTrendData = {
      ...mockSentimentData,
      sentiment_trend: [
        { timestamp: '2024-01-15T09:00:00Z', sentiment: 0.75 },
        { timestamp: '2024-01-15T10:00:00Z', sentiment: 0.76 }, // Small change
      ],
    }

    render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={stableTrendData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    expect(screen.getByText('Stable')).toBeInTheDocument()
  })

  it('handles click events', () => {
    render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={mockSentimentData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    const card = screen.getByRole('generic')
    fireEvent.click(card)

    expect(mockOnSelect).toHaveBeenCalledTimes(1)
  })

  it('shows selected state correctly', () => {
    const { container } = render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={mockSentimentData}
        isSelected={true}
        onSelect={mockOnSelect}
      />
    )

    expect(container.firstChild).toHaveClass('ring-2', 'ring-blue-500')
  })

  it('shows loading state when no sentiment data', () => {
    render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={undefined}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    expect(screen.getByText('Loading sentiment data...')).toBeInTheDocument()
    expect(screen.getByText('0.0%')).toBeInTheDocument() // Default sentiment
    expect(screen.getByText('0 mentions')).toBeInTheDocument() // Default mentions
  })

  it('renders mini trend chart when trend data available', () => {
    const { container } = render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={mockSentimentData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    // Check for trend chart container
    const trendChart = container.querySelector('.flex.items-end.space-x-1.h-8')
    expect(trendChart).toBeInTheDocument()
  })

  it('does not render trend chart when insufficient data', () => {
    const singlePointData = {
      ...mockSentimentData,
      sentiment_trend: [
        { timestamp: '2024-01-15T10:00:00Z', sentiment: 0.75 },
      ],
    }

    const { container } = render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={singlePointData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    const trendChart = container.querySelector('.flex.items-end.space-x-1.h-8')
    expect(trendChart).not.toBeInTheDocument()
  })

  it('formats timestamp correctly', () => {
    const recentData = {
      ...mockSentimentData,
      sentiment_trend: [
        { timestamp: '2024-01-15T14:30:00Z', sentiment: 0.75 },
      ],
    }

    render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={recentData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    // Should show formatted time (exact format depends on locale)
    expect(screen.getByText(/Updated/)).toBeInTheDocument()
  })

  it('handles zero mentions correctly', () => {
    const zeroMentionsData = {
      ...mockSentimentData,
      total_mentions: 0,
    }

    render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={zeroMentionsData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    expect(screen.getByText('0 mentions')).toBeInTheDocument()
  })

  it('formats large mention counts correctly', () => {
    const largeMentionsData = {
      ...mockSentimentData,
      total_mentions: 1500,
    }

    render(
      <TeamSentimentCard
        team={mockTeam}
        sentimentData={largeMentionsData}
        isSelected={false}
        onSelect={mockOnSelect}
      />
    )

    expect(screen.getByText('1,500 mentions')).toBeInTheDocument()
  })
})