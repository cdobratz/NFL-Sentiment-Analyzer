import React from 'react'
import { TrendingUp, TrendingDown, Minus, MessageCircle } from 'lucide-react'
import { TeamSentiment } from '../hooks/useSentimentWebSocket'

interface Team {
  id: string
  name: string
  abbreviation: string
  conference: string
  division: string
}

interface TeamSentimentCardProps {
  team: Team
  sentimentData?: TeamSentiment
  isSelected: boolean
  onSelect: () => void
}

const TeamSentimentCard: React.FC<TeamSentimentCardProps> = ({
  team,
  sentimentData,
  isSelected,
  onSelect
}) => {
  // Default sentiment data if not available
  const sentiment = sentimentData?.current_sentiment ?? 0
  const totalMentions = sentimentData?.total_mentions ?? 0
  const trend = sentimentData?.sentiment_trend ?? []

  // Calculate trend direction
  const getTrendDirection = () => {
    if (trend.length < 2) return 'neutral'
    const recent = trend[trend.length - 1]?.sentiment ?? 0
    const previous = trend[trend.length - 2]?.sentiment ?? 0
    
    if (recent > previous + 0.1) return 'up'
    if (recent < previous - 0.1) return 'down'
    return 'neutral'
  }

  const trendDirection = getTrendDirection()

  // Get sentiment color and icon
  const getSentimentDisplay = (value: number) => {
    if (value > 0.1) {
      return {
        color: 'text-green-600',
        bgColor: 'bg-green-50',
        borderColor: 'border-green-200',
        icon: <TrendingUp className="w-4 h-4" />,
        label: 'Positive'
      }
    } else if (value < -0.1) {
      return {
        color: 'text-red-600',
        bgColor: 'bg-red-50',
        borderColor: 'border-red-200',
        icon: <TrendingDown className="w-4 h-4" />,
        label: 'Negative'
      }
    } else {
      return {
        color: 'text-gray-600',
        bgColor: 'bg-gray-50',
        borderColor: 'border-gray-200',
        icon: <Minus className="w-4 h-4" />,
        label: 'Neutral'
      }
    }
  }

  const sentimentDisplay = getSentimentDisplay(sentiment)

  // Get trend icon
  const getTrendIcon = () => {
    switch (trendDirection) {
      case 'up':
        return <TrendingUp className="w-3 h-3 text-green-500" />
      case 'down':
        return <TrendingDown className="w-3 h-3 text-red-500" />
      default:
        return <Minus className="w-3 h-3 text-gray-400" />
    }
  }

  // Format sentiment score as percentage
  const formatSentiment = (value: number) => {
    const percentage = ((value + 1) / 2) * 100 // Convert from -1,1 to 0,100
    return `${percentage.toFixed(1)}%`
  }

  return (
    <div
      className={`card cursor-pointer transition-all duration-200 hover:shadow-md ${
        isSelected 
          ? `ring-2 ring-blue-500 ${sentimentDisplay.bgColor}` 
          : 'hover:bg-gray-50'
      }`}
      onClick={onSelect}
    >
      {/* Team Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">
              {team.abbreviation}
            </span>
          </div>
          <div>
            <h3 className="font-semibold text-gray-900 text-sm">
              {team.name}
            </h3>
            <p className="text-xs text-gray-500">
              {team.conference} • {team.division}
            </p>
          </div>
        </div>
        
        {/* Selection indicator */}
        {isSelected && (
          <div className="w-2 h-2 bg-blue-500 rounded-full" />
        )}
      </div>

      {/* Sentiment Score */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <div className={`p-1 rounded ${sentimentDisplay.bgColor}`}>
            {sentimentDisplay.icon}
          </div>
          <div>
            <p className={`font-semibold ${sentimentDisplay.color}`}>
              {formatSentiment(sentiment)}
            </p>
            <p className="text-xs text-gray-500">
              {sentimentDisplay.label}
            </p>
          </div>
        </div>

        {/* Trend indicator */}
        <div className="flex items-center space-x-1">
          {getTrendIcon()}
          <span className="text-xs text-gray-500">
            {trendDirection === 'up' ? 'Rising' : 
             trendDirection === 'down' ? 'Falling' : 'Stable'}
          </span>
        </div>
      </div>

      {/* Stats */}
      <div className="flex items-center justify-between text-xs text-gray-500">
        <div className="flex items-center space-x-1">
          <MessageCircle className="w-3 h-3" />
          <span>{totalMentions.toLocaleString()} mentions</span>
        </div>
        
        {trend.length > 0 && (
          <div className="text-xs text-gray-400">
            Updated {new Date(trend[trend.length - 1]?.timestamp || Date.now()).toLocaleTimeString([], {
              hour: '2-digit',
              minute: '2-digit'
            })}
          </div>
        )}
      </div>

      {/* Mini trend chart */}
      {trend.length > 1 && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <div className="flex items-end space-x-1 h-8">
            {trend.slice(-10).map((point, index) => {
              const height = Math.max(2, Math.abs(point.sentiment) * 20 + 4)
              const isPositive = point.sentiment > 0
              
              return (
                <div
                  key={index}
                  className={`flex-1 rounded-sm ${
                    isPositive ? 'bg-green-300' : 'bg-red-300'
                  }`}
                  style={{ height: `${height}px` }}
                  title={`${formatSentiment(point.sentiment)} at ${new Date(point.timestamp).toLocaleTimeString()}`}
                />
              )
            })}
          </div>
        </div>
      )}

      {/* Loading state */}
      {!sentimentData && (
        <div className="mt-3 pt-3 border-t border-gray-100">
          <div className="flex items-center space-x-2 text-xs text-gray-400">
            <div className="w-2 h-2 bg-gray-300 rounded-full animate-pulse" />
            <span>Loading sentiment data...</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default TeamSentimentCard