import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen } from '../../test/utils'
import RealTimeChart from '../RealTimeChart'

// Mock Chart.js and react-chartjs-2
vi.mock('react-chartjs-2', () => ({
  Line: vi.fn(({ data, options }) => (
    <div data-testid="line-chart">
      <div data-testid="chart-data">{JSON.stringify(data)}</div>
      <div data-testid="chart-options">{JSON.stringify(options)}</div>
    </div>
  )),
}))

vi.mock('chart.js', () => ({
  Chart: {
    register: vi.fn(),
  },
  CategoryScale: {},
  LinearScale: {},
  PointElement: {},
  LineElement: {},
  Title: {},
  Tooltip: {},
  Legend: {},
  TimeScale: {},
}))

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
  {
    id: '3',
    text: 'Okay performance',
    sentiment: 'neutral' as const,
    confidence: 0.7,
    team_id: '1',
    source: 'twitter',
    timestamp: '2024-01-15T10:10:00Z',
  },
]

describe('RealTimeChart', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders chart with data', () => {
    render(<RealTimeChart data={mockSentimentData} />)

    expect(screen.getByTestId('line-chart')).toBeInTheDocument()
  })

  it('shows empty state when no data', () => {
    render(<RealTimeChart data={[]} />)

    expect(screen.getByText('No sentiment data available')).toBeInTheDocument()
    expect(screen.getByText('Waiting for real-time updates...')).toBeInTheDocument()
  })

  it('shows team-specific empty state when selected team has no data', () => {
    render(<RealTimeChart data={[]} selectedTeam="3" />)

    expect(screen.getByText('No sentiment data available')).toBeInTheDocument()
    expect(screen.getByText('No data for selected team')).toBeInTheDocument()
  })

  it('filters data by selected team', () => {
    render(<RealTimeChart data={mockSentimentData} selectedTeam="1" />)

    const chartData = screen.getByTestId('chart-data')
    const parsedData = JSON.parse(chartData.textContent || '{}')
    
    // Should have single dataset for selected team
    expect(parsedData.datasets).toHaveLength(1)
    expect(parsedData.datasets[0].label).toBe('Sentiment Score')
  })

  it('shows aggregated data when no team selected', () => {
    render(<RealTimeChart data={mockSentimentData} />)

    const chartData = screen.getByTestId('chart-data')
    const parsedData = JSON.parse(chartData.textContent || '{}')
    
    // Should have multiple datasets for overall view
    expect(parsedData.datasets.length).toBeGreaterThan(1)
    expect(parsedData.datasets[0].label).toBe('Overall Sentiment')
  })

  it('applies mobile-specific options', () => {
    render(<RealTimeChart data={mockSentimentData} isMobile={true} />)

    const chartOptions = screen.getByTestId('chart-options')
    const parsedOptions = JSON.parse(chartOptions.textContent || '{}')
    
    expect(parsedOptions.plugins.legend.position).toBe('bottom')
  })

  it('applies desktop options by default', () => {
    render(<RealTimeChart data={mockSentimentData} isMobile={false} />)

    const chartOptions = screen.getByTestId('chart-options')
    const parsedOptions = JSON.parse(chartOptions.textContent || '{}')
    
    expect(parsedOptions.plugins.legend.position).toBe('top')
  })

  it('sets custom height', () => {
    const { container } = render(
      <RealTimeChart data={mockSentimentData} height={400} />
    )

    const chartContainer = container.firstChild as HTMLElement
    expect(chartContainer.style.height).toBe('400px')
  })

  it('uses default height when not specified', () => {
    const { container } = render(<RealTimeChart data={mockSentimentData} />)

    const chartContainer = container.firstChild as HTMLElement
    expect(chartContainer.style.height).toBe('300px')
  })

  it('processes sentiment values correctly', () => {
    render(<RealTimeChart data={mockSentimentData} selectedTeam="1" />)

    const chartData = screen.getByTestId('chart-data')
    const parsedData = JSON.parse(chartData.textContent || '{}')
    
    const dataPoints = parsedData.datasets[0].data
    
    // First point: positive sentiment (1) * confidence (0.9) = 0.9
    expect(dataPoints[0].y).toBe(0.9)
    
    // Second point: neutral sentiment (0) * confidence (0.7) = 0
    expect(dataPoints[1].y).toBe(0)
  })

  it('sorts data by timestamp', () => {
    const unsortedData = [
      {
        ...mockSentimentData[2],
        timestamp: '2024-01-15T10:15:00Z', // Latest
      },
      {
        ...mockSentimentData[0],
        timestamp: '2024-01-15T10:00:00Z', // Earliest
      },
      {
        ...mockSentimentData[1],
        timestamp: '2024-01-15T10:10:00Z', // Middle
      },
    ]

    render(<RealTimeChart data={unsortedData} selectedTeam="1" />)

    const chartData = screen.getByTestId('chart-data')
    const parsedData = JSON.parse(chartData.textContent || '{}')
    
    const dataPoints = parsedData.datasets[0].data
    
    // Should be sorted by timestamp (x values should be in ascending order)
    expect(dataPoints[0].x).toBeLessThan(dataPoints[1].x)
  })

  it('handles empty sentiment data gracefully', () => {
    render(<RealTimeChart data={[]} />)

    expect(screen.getByText('No sentiment data available')).toBeInTheDocument()
  })

  it('configures chart options correctly for mobile', () => {
    render(<RealTimeChart data={mockSentimentData} isMobile={true} />)

    const chartOptions = screen.getByTestId('chart-options')
    const parsedOptions = JSON.parse(chartOptions.textContent || '{}')
    
    // Mobile-specific configurations
    expect(parsedOptions.plugins.legend.labels.font.size).toBe(10)
    expect(parsedOptions.scales.x.ticks.maxTicksLimit).toBe(4)
    expect(parsedOptions.scales.x.title.display).toBe(false)
    expect(parsedOptions.scales.y.title.display).toBe(false)
  })

  it('configures chart options correctly for desktop', () => {
    render(<RealTimeChart data={mockSentimentData} isMobile={false} />)

    const chartOptions = screen.getByTestId('chart-options')
    const parsedOptions = JSON.parse(chartOptions.textContent || '{}')
    
    // Desktop-specific configurations
    expect(parsedOptions.plugins.legend.labels.font.size).toBe(12)
    expect(parsedOptions.scales.x.ticks.maxTicksLimit).toBe(8)
    expect(parsedOptions.scales.x.title.display).toBe(true)
    expect(parsedOptions.scales.y.title.display).toBe(true)
  })

  it('sets correct y-axis range', () => {
    render(<RealTimeChart data={mockSentimentData} />)

    const chartOptions = screen.getByTestId('chart-options')
    const parsedOptions = JSON.parse(chartOptions.textContent || '{}')
    
    expect(parsedOptions.scales.y.min).toBe(-1)
    expect(parsedOptions.scales.y.max).toBe(1)
  })

  it('groups data by time for aggregated view', () => {
    const sameTimeData = [
      {
        id: '1',
        text: 'Great!',
        sentiment: 'positive' as const,
        confidence: 0.9,
        team_id: '1',
        source: 'twitter',
        timestamp: '2024-01-15T10:00:00Z',
      },
      {
        id: '2',
        text: 'Bad!',
        sentiment: 'negative' as const,
        confidence: 0.8,
        team_id: '2',
        source: 'twitter',
        timestamp: '2024-01-15T10:00:00Z', // Same minute
      },
    ]

    render(<RealTimeChart data={sameTimeData} />)

    const chartData = screen.getByTestId('chart-data')
    const parsedData = JSON.parse(chartData.textContent || '{}')
    
    // Should aggregate data points from the same time period
    expect(parsedData.datasets[0].data).toHaveLength(1)
  })
})