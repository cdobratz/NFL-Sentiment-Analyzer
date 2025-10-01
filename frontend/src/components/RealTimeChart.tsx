import React, { useMemo, useEffect, useRef } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale,
  ChartOptions,
  ChartData
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import 'chartjs-adapter-date-fns'
import { SentimentData } from '../hooks/useSentimentWebSocket'

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  TimeScale
)

interface RealTimeChartProps {
  data: SentimentData[]
  selectedTeam?: string | null
  height?: number
  isMobile?: boolean
}

const RealTimeChart: React.FC<RealTimeChartProps> = ({
  data,
  selectedTeam,
  height = 300,
  isMobile = false
}) => {
  const chartRef = useRef<ChartJS<'line'>>(null)

  // Process data for chart
  const chartData = useMemo(() => {
    // Filter data by selected team if specified
    const filteredData = selectedTeam 
      ? data.filter(item => item.team_id === selectedTeam)
      : data

    // Sort by timestamp
    const sortedData = [...filteredData].sort(
      (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    )

    // Convert sentiment to numeric values
    const sentimentToValue = (sentiment: string): number => {
      switch (sentiment.toLowerCase()) {
        case 'positive': return 1
        case 'negative': return -1
        default: return 0
      }
    }

    // Create datasets
    const datasets = []

    if (selectedTeam) {
      // Single team view - show individual points and trend
      datasets.push({
        label: 'Sentiment Score',
        data: sortedData.map(item => ({
          x: new Date(item.timestamp).getTime(),
          y: sentimentToValue(item.sentiment) * item.confidence
        })),
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.4,
        fill: true
      })
    } else {
      // All teams view - show aggregated sentiment over time
      const timeGroups = new Map<string, { positive: number, negative: number, neutral: number, total: number }>()
      
      sortedData.forEach(item => {
        const timeKey = new Date(item.timestamp).toISOString().slice(0, 16) // Group by minute
        
        if (!timeGroups.has(timeKey)) {
          timeGroups.set(timeKey, { positive: 0, negative: 0, neutral: 0, total: 0 })
        }
        
        const group = timeGroups.get(timeKey)!
        group.total++
        
        switch (item.sentiment.toLowerCase()) {
          case 'positive':
            group.positive++
            break
          case 'negative':
            group.negative++
            break
          default:
            group.neutral++
        }
      })

      // Convert to chart data
      const aggregatedData = Array.from(timeGroups.entries())
        .map(([timeKey, counts]) => ({
          x: new Date(timeKey).getTime(),
          y: counts.total > 0 ? (counts.positive - counts.negative) / counts.total : 0
        }))
        .sort((a, b) => a.x - b.x)

      datasets.push({
        label: 'Overall Sentiment',
        data: aggregatedData,
        borderColor: 'rgb(34, 197, 94)',
        backgroundColor: 'rgba(34, 197, 94, 0.1)',
        borderWidth: 2,
        pointRadius: 3,
        pointHoverRadius: 5,
        tension: 0.4,
        fill: true
      })

      // Add positive/negative trend lines
      const positiveData = Array.from(timeGroups.entries())
        .map(([timeKey, counts]) => ({
          x: new Date(timeKey).getTime(),
          y: counts.total > 0 ? counts.positive / counts.total : 0
        }))
        .sort((a, b) => a.x - b.x)

      const negativeData = Array.from(timeGroups.entries())
        .map(([timeKey, counts]) => ({
          x: new Date(timeKey).getTime(),
          y: counts.total > 0 ? -counts.negative / counts.total : 0
        }))
        .sort((a, b) => a.x - b.x)

      datasets.push(
        {
          label: 'Positive Sentiment',
          data: positiveData,
          borderColor: 'rgb(34, 197, 94)',
          backgroundColor: 'rgba(34, 197, 94, 0.05)',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.4,
          fill: false
        },
        {
          label: 'Negative Sentiment',
          data: negativeData,
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgba(239, 68, 68, 0.05)',
          borderWidth: 1,
          pointRadius: 0,
          tension: 0.4,
          fill: false
        }
      )
    }

    return {
      datasets
    } as ChartData<'line'>
  }, [data, selectedTeam])

  // Chart options
  const options: ChartOptions<'line'> = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        position: isMobile ? 'bottom' as const : 'top' as const,
        labels: {
          usePointStyle: true,
          padding: isMobile ? 10 : 20,
          boxWidth: isMobile ? 8 : 12,
          font: {
            size: isMobile ? 10 : 12
          }
        }
      },
      title: {
        display: false
      },
      tooltip: {
        callbacks: {
          label: (context) => {
            const value = context.parsed.y
            const sentiment = value > 0.1 ? 'Positive' : value < -0.1 ? 'Negative' : 'Neutral'
            return `${context.dataset.label}: ${(value * 100).toFixed(1)}% (${sentiment})`
          }
        },
        titleFont: {
          size: isMobile ? 10 : 12
        },
        bodyFont: {
          size: isMobile ? 10 : 12
        }
      }
    },
    scales: {
      x: {
        type: 'time' as const,
        time: {
          displayFormats: {
            minute: 'HH:mm',
            hour: 'HH:mm',
            day: 'MMM dd'
          }
        },
        title: {
          display: !isMobile,
          text: 'Time',
          font: {
            size: isMobile ? 10 : 12
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        },
        ticks: {
          font: {
            size: isMobile ? 9 : 11
          },
          maxTicksLimit: isMobile ? 4 : 8
        }
      },
      y: {
        min: -1,
        max: 1,
        title: {
          display: !isMobile,
          text: 'Sentiment Score',
          font: {
            size: isMobile ? 10 : 12
          }
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        },
        ticks: {
          font: {
            size: isMobile ? 9 : 11
          },
          callback: (value) => {
            const numValue = Number(value)
            if (isMobile) {
              // Shorter labels for mobile
              if (numValue > 0.5) return 'V+'
              if (numValue > 0.1) return '+'
              if (numValue > -0.1) return '~'
              if (numValue > -0.5) return '-'
              return 'V-'
            } else {
              if (numValue > 0.5) return 'Very Positive'
              if (numValue > 0.1) return 'Positive'
              if (numValue > -0.1) return 'Neutral'
              if (numValue > -0.5) return 'Negative'
              return 'Very Negative'
            }
          }
        }
      }
    },
    elements: {
      point: {
        hoverBackgroundColor: 'white',
        hoverBorderWidth: 2
      }
    },
    animation: {
      duration: 750,
      easing: 'easeInOutQuart'
    }
  }), [])

  // Auto-update chart when new data arrives
  useEffect(() => {
    if (chartRef.current) {
      chartRef.current.update('none')
    }
  }, [data])

  if (data.length === 0) {
    return (
      <div 
        className="flex items-center justify-center bg-gray-50 rounded-lg"
        style={{ height: `${height}px` }}
      >
        <div className="text-center">
          <div className="w-12 h-12 bg-gray-200 rounded-full flex items-center justify-center mx-auto mb-3">
            <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <p className="text-gray-500 text-sm">No sentiment data available</p>
          <p className="text-gray-400 text-xs mt-1">
            {selectedTeam ? 'No data for selected team' : 'Waiting for real-time updates...'}
          </p>
        </div>
      </div>
    )
  }

  return (
    <div style={{ height: `${height}px` }}>
      <Line 
        ref={chartRef}
        data={chartData} 
        options={options} 
      />
    </div>
  )
}

export default RealTimeChart