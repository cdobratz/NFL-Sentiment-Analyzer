import React, { useState, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { dataApi } from '../services/api'
import useSentimentWebSocket from '../hooks/useSentimentWebSocket'
import TeamSentimentCard from './TeamSentimentCard'
import RealTimeChart from './RealTimeChart'
import GamePredictionPanel from './GamePredictionPanel'
import LoadingSpinner from './LoadingSpinner'
import { Wifi, WifiOff, AlertCircle, RefreshCw } from 'lucide-react'

// Hook to detect mobile screen size
const useIsMobile = () => {
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkIsMobile = () => {
      setIsMobile(window.innerWidth < 768)
    }

    checkIsMobile()
    window.addEventListener('resize', checkIsMobile)

    return () => window.removeEventListener('resize', checkIsMobile)
  }, [])

  return isMobile
}

interface Team {
  id: string
  name: string
  abbreviation: string
  conference: string
  division: string
}

const SentimentDashboard: React.FC = () => {
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null)
  const [selectedConference, setSelectedConference] = useState<string>('all')
  const isMobile = useIsMobile()

  // Fetch teams data
  const { data: teamsResponse, isLoading: teamsLoading, error: teamsError } = useQuery({
    queryKey: ['teams'],
    queryFn: () => dataApi.getTeams(),
  })

  // WebSocket connection for real-time data
  const {
    recentSentiments,
    teamSentiments,
    gamePredictions,
    connectionStatus,
    subscribeToTeam,
    reconnect
  } = useSentimentWebSocket()

  const teams: Team[] = teamsResponse?.data?.data || []

  // Filter teams by conference
  const filteredTeams = selectedConference === 'all' 
    ? teams 
    : teams.filter(team => team.conference === selectedConference)

  // Subscribe to team sentiment updates when team is selected
  useEffect(() => {
    if (selectedTeam) {
      subscribeToTeam(selectedTeam)
    }
  }, [selectedTeam, subscribeToTeam])

  const ConnectionStatus = () => {
    const { isConnected, isConnecting, error, reconnectAttempts } = connectionStatus

    if (isConnected) {
      return (
        <div className="flex items-center space-x-2 text-green-600">
          <Wifi className="w-4 h-4" />
          <span className="text-sm">Live</span>
        </div>
      )
    }

    if (isConnecting) {
      return (
        <div className="flex items-center space-x-2 text-yellow-600">
          <RefreshCw className="w-4 h-4 animate-spin" />
          <span className="text-sm">Connecting...</span>
        </div>
      )
    }

    if (error) {
      return (
        <div className="flex items-center space-x-2 text-red-600">
          <WifiOff className="w-4 h-4" />
          <span className="text-sm">
            Disconnected {reconnectAttempts > 0 && `(${reconnectAttempts} attempts)`}
          </span>
          <button
            onClick={reconnect}
            className="text-xs bg-red-100 hover:bg-red-200 px-2 py-1 rounded"
          >
            Retry
          </button>
        </div>
      )
    }

    return (
      <div className="flex items-center space-x-2 text-gray-500">
        <WifiOff className="w-4 h-4" />
        <span className="text-sm">Offline</span>
      </div>
    )
  }

  if (teamsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  if (teamsError) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-900 mb-2">Error Loading Data</h3>
          <p className="text-gray-600">Failed to load teams data. Please try again.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header with connection status */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">NFL Sentiment Dashboard</h1>
          <p className="text-gray-600 mt-2">
            Real-time sentiment analysis and betting insights
          </p>
        </div>
        <ConnectionStatus />
      </div>

      {/* Conference Filter */}
      <div className="flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-4">
        <select
          value={selectedConference}
          onChange={(e) => setSelectedConference(e.target.value)}
          className="w-full sm:w-auto px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          <option value="all">All Conferences</option>
          <option value="AFC">AFC</option>
          <option value="NFC">NFC</option>
        </select>
      </div>

      {/* Real-time Chart */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Sentiment Trends
        </h2>
        <RealTimeChart 
          data={recentSentiments}
          selectedTeam={selectedTeam}
          height={isMobile ? 250 : 300}
          isMobile={isMobile}
        />
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Team Sentiment Cards */}
        <div className="xl:col-span-2">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Team Sentiment
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-2 gap-4">
            {filteredTeams.map((team) => (
              <TeamSentimentCard
                key={team.id}
                team={team}
                sentimentData={teamSentiments[team.id]}
                isSelected={selectedTeam === team.id}
                onSelect={() => setSelectedTeam(
                  selectedTeam === team.id ? null : team.id
                )}
              />
            ))}
          </div>
          
          {filteredTeams.length === 0 && (
            <div className="text-center py-8">
              <p className="text-gray-500">No teams found for the selected conference.</p>
            </div>
          )}
        </div>

        {/* Game Predictions Panel */}
        <div className="xl:col-span-1">
          <GamePredictionPanel 
            predictions={Object.values(gamePredictions)}
            selectedTeam={selectedTeam}
          />
        </div>
      </div>

      {/* Recent Activity */}
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">
          Recent Sentiment Activity
        </h2>
        <div className="space-y-3 max-h-96 overflow-y-auto">
          {recentSentiments.length > 0 ? (
            recentSentiments.map((sentiment) => (
              <div
                key={sentiment.id}
                className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg"
              >
                <div className={`w-3 h-3 rounded-full mt-2 ${
                  sentiment.sentiment === 'positive' ? 'bg-green-500' :
                  sentiment.sentiment === 'negative' ? 'bg-red-500' :
                  'bg-gray-500'
                }`} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-gray-900 line-clamp-2">
                    {sentiment.text}
                  </p>
                  <div className="flex items-center mt-1 space-x-2 text-xs text-gray-500">
                    <span className="capitalize">{sentiment.sentiment}</span>
                    <span>•</span>
                    <span>{(sentiment.confidence * 100).toFixed(1)}% confidence</span>
                    <span>•</span>
                    <span>{sentiment.source}</span>
                    <span>•</span>
                    <span>{new Date(sentiment.timestamp).toLocaleTimeString()}</span>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center py-8">
              <p className="text-gray-500">No recent sentiment data available.</p>
              {!connectionStatus.isConnected && (
                <p className="text-sm text-gray-400 mt-2">
                  Connect to see real-time updates.
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default SentimentDashboard