import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { dataApi } from '../services/api'
import { GamePrediction } from '../hooks/useSentimentWebSocket'
import { Calendar, DollarSign, Clock, Target } from 'lucide-react'
import LoadingSpinner from './LoadingSpinner'

interface GamePredictionPanelProps {
  predictions: GamePrediction[]
  selectedTeam?: string | null
}

interface Game {
  id: string
  home_team: {
    id: string
    name: string
    abbreviation: string
  }
  away_team: {
    id: string
    name: string
    abbreviation: string
  }
  game_date: string
  week: number
  status: string
}

const GamePredictionPanel: React.FC<GamePredictionPanelProps> = ({
  predictions,
  selectedTeam
}) => {
  const [activeTab, setActiveTab] = useState<'upcoming' | 'predictions'>('upcoming')

  // Fetch upcoming games
  const { data: gamesResponse, isLoading: gamesLoading } = useQuery({
    queryKey: ['upcoming-games', selectedTeam],
    queryFn: () => dataApi.getGames({ 
      status: 'scheduled', 
      limit: 10,
      ...(selectedTeam && { team_id: selectedTeam })
    }),
  })

  const games: Game[] = gamesResponse?.data?.data || []

  // Filter predictions for selected team if specified
  const filteredPredictions = selectedTeam 
    ? predictions.filter(p => 
        p.home_team.includes(selectedTeam) || p.away_team.includes(selectedTeam)
      )
    : predictions

  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return {
      date: date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric' 
      }),
      time: date.toLocaleTimeString('en-US', { 
        hour: 'numeric', 
        minute: '2-digit',
        hour12: true 
      })
    }
  }

  const formatOdds = (odds: number) => {
    return odds > 0 ? `+${odds}` : `${odds}`
  }

  const getWinProbabilityColor = (probability: number) => {
    if (probability > 0.6) return 'text-green-600 bg-green-50'
    if (probability > 0.4) return 'text-yellow-600 bg-yellow-50'
    return 'text-red-600 bg-red-50'
  }

  const PredictionCard: React.FC<{ prediction: GamePrediction }> = ({ prediction }) => (
    <div className="border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-shadow">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <Target className="w-4 h-4 text-blue-500" />
          <span className="font-medium text-sm text-gray-900">Game Prediction</span>
        </div>
        <div className="text-xs text-gray-500">
          Sentiment Factor: {(prediction.prediction.sentiment_factor * 100).toFixed(1)}%
        </div>
      </div>

      <div className="space-y-3">
        {/* Teams and Probabilities */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 bg-blue-500 rounded text-white text-xs flex items-center justify-center">
                H
              </div>
              <span className="text-sm font-medium">{prediction.home_team}</span>
            </div>
            <div className={`px-2 py-1 rounded text-xs font-medium ${
              getWinProbabilityColor(prediction.prediction.home_win_probability)
            }`}>
              {(prediction.prediction.home_win_probability * 100).toFixed(1)}%
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 bg-gray-500 rounded text-white text-xs flex items-center justify-center">
                A
              </div>
              <span className="text-sm font-medium">{prediction.away_team}</span>
            </div>
            <div className={`px-2 py-1 rounded text-xs font-medium ${
              getWinProbabilityColor(prediction.prediction.away_win_probability)
            }`}>
              {(prediction.prediction.away_win_probability * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Betting Lines */}
        {prediction.betting_lines && (
          <div className="pt-3 border-t border-gray-100">
            <div className="flex items-center space-x-4 text-xs text-gray-600">
              <div className="flex items-center space-x-1">
                <DollarSign className="w-3 h-3" />
                <span>Spread: {prediction.betting_lines.spread > 0 ? '+' : ''}{prediction.betting_lines.spread}</span>
              </div>
              <div>O/U: {prediction.betting_lines.over_under}</div>
            </div>
            <div className="flex items-center space-x-4 text-xs text-gray-600 mt-1">
              <span>ML: {formatOdds(prediction.betting_lines.moneyline.home)} / {formatOdds(prediction.betting_lines.moneyline.away)}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )

  const GameCard: React.FC<{ game: Game }> = ({ game }) => {
    const { date, time } = formatDate(game.game_date)
    
    return (
      <div className="border border-gray-200 rounded-lg p-4 hover:shadow-sm transition-shadow">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center space-x-2">
            <Calendar className="w-4 h-4 text-blue-500" />
            <span className="font-medium text-sm text-gray-900">Week {game.week}</span>
          </div>
          <div className="text-xs text-gray-500">
            {date} • {time}
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 bg-gray-500 rounded text-white text-xs flex items-center justify-center">
                {game.away_team.abbreviation.slice(0, 2)}
              </div>
              <span className="text-sm font-medium">{game.away_team.name}</span>
            </div>
            <span className="text-xs text-gray-500">@</span>
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className="w-6 h-6 bg-blue-500 rounded text-white text-xs flex items-center justify-center">
                {game.home_team.abbreviation.slice(0, 2)}
              </div>
              <span className="text-sm font-medium">{game.home_team.name}</span>
            </div>
          </div>
        </div>

        <div className="mt-3 pt-3 border-t border-gray-100">
          <div className="flex items-center justify-between text-xs text-gray-500">
            <span className="capitalize">{game.status}</span>
            <div className="flex items-center space-x-1">
              <Clock className="w-3 h-3" />
              <span>Upcoming</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-gray-900">
          {selectedTeam ? 'Team Games & Predictions' : 'Games & Predictions'}
        </h2>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
        <button
          onClick={() => { setActiveTab('upcoming') }}
          className={`flex-1 px-3 py-2 text-sm font-medium rounded-md transition-colors ${
            activeTab === 'upcoming'
              ? 'bg-white text-gray-900 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Upcoming Games
        </button>
        <button
          onClick={() => { setActiveTab('predictions') }}
          className={`flex-1 px-3 py-2 text-sm font-medium rounded-md transition-colors ${
            activeTab === 'predictions'
              ? 'bg-white text-gray-900 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          Predictions
        </button>
      </div>

      {/* Content */}
      <div className="space-y-3 max-h-96 overflow-y-auto">
        {activeTab === 'upcoming' && (
          <>
            {gamesLoading ? (
              <div className="flex items-center justify-center py-8">
                <LoadingSpinner size="sm" />
              </div>
            ) : games.length > 0 ? (
              games.map((game) => (
                <GameCard key={game.id} game={game} />
              ))
            ) : (
              <div className="text-center py-8">
                <Calendar className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-500 text-sm">No upcoming games found</p>
                {selectedTeam && (
                  <p className="text-gray-400 text-xs mt-1">
                    No games scheduled for selected team
                  </p>
                )}
              </div>
            )}
          </>
        )}

        {activeTab === 'predictions' && (
          <>
            {filteredPredictions.length > 0 ? (
              filteredPredictions.map((prediction) => (
                <PredictionCard key={prediction.game_id} prediction={prediction} />
              ))
            ) : (
              <div className="text-center py-8">
                <Target className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-500 text-sm">No predictions available</p>
                <p className="text-gray-400 text-xs mt-1">
                  Predictions will appear as games approach
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default GamePredictionPanel