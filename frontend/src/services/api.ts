import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor to add auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth-storage')
  if (token) {
    try {
      const parsed = JSON.parse(token)
      if (parsed.state?.token) {
        config.headers.Authorization = `Bearer ${parsed.state.token}`
      }
    } catch (error) {
      console.error('Error parsing auth token:', error)
    }
  }
  return config
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid, clear auth state
      localStorage.removeItem('auth-storage')
      
      // Show toast notification
      import('react-hot-toast').then(({ default: toast }) => {
        toast.error('Session expired. Please log in again.')
      })
      
      // Redirect to login after a short delay
      setTimeout(() => {
        window.location.href = '/login'
      }, 1000)
    }
    return Promise.reject(error)
  }
)

// Auth API
export const authApi = {
  login: (email: string, password: string) =>
    api.post('/auth/login', new URLSearchParams({ username: email, password }), {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    }),
  
  register: (userData: { email: string; username: string; password: string }) =>
    api.post('/auth/register', userData),
  
  getProfile: (token?: string) =>
    api.get('/auth/profile', token ? { headers: { Authorization: `Bearer ${token}` } } : {}),
  
  refreshToken: (token: string) =>
    api.post('/auth/refresh', {}, { headers: { Authorization: `Bearer ${token}` } }),
  
  logout: (token: string) =>
    api.delete('/auth/logout', { headers: { Authorization: `Bearer ${token}` } }),
  
  updateProfile: (userData: { username: string; email: string }, token: string) =>
    api.put('/auth/profile', userData, { headers: { Authorization: `Bearer ${token}` } }),
  
  changePassword: (passwordData: { currentPassword: string; newPassword: string }, token: string) =>
    api.put('/auth/change-password', passwordData, { headers: { Authorization: `Bearer ${token}` } }),
}

// Sentiment API
export const sentimentApi = {
  analyze: (text: string, context?: unknown) =>
    api.post('/sentiment/analyze', { text, context }),
  
  analyzeBatch: (texts: string[], context?: unknown) =>
    api.post('/sentiment/analyze/batch', { texts, context }),
  
  getTeamSentiment: (teamId: string, days = 7) =>
    api.get(`/sentiment/team/${teamId}?days=${days}`),
  
  getPlayerSentiment: (playerId: string, days = 7) =>
    api.get(`/sentiment/player/${playerId}?days=${days}`),
  
  getGameSentiment: (gameId: string) =>
    api.get(`/sentiment/game/${gameId}`),
  
  getTrends: (params?: { team_id?: string; player_id?: string; days?: number }) =>
    api.get('/sentiment/trends', { params }),
  
  getRecent: (params?: { limit?: number; team_id?: string; player_id?: string }) =>
    api.get('/sentiment/recent', { params }),
}

// Data API
export const dataApi = {
  getTeams: (params?: { conference?: string; division?: string }) =>
    api.get('/data/teams', { params }),
  
  getTeam: (teamId: string) =>
    api.get(`/data/teams/${teamId}`),
  
  getPlayers: (params?: { team_id?: string; position?: string; limit?: number }) =>
    api.get('/data/players', { params }),
  
  getPlayer: (playerId: string) =>
    api.get(`/data/players/${playerId}`),
  
  getGames: (params?: { 
    week?: number; 
    season?: number; 
    team_id?: string; 
    status?: string; 
    limit?: number 
  }) =>
    api.get('/data/games', { params }),
  
  getGame: (gameId: string) =>
    api.get(`/data/games/${gameId}`),
  
  getBettingLines: (params?: { game_id?: string; sportsbook?: string }) =>
    api.get('/data/betting-lines', { params }),
  
  getSchedule: (params?: { week?: number; season?: number; team_id?: string }) =>
    api.get('/data/schedule', { params }),
}

// Admin API
export const adminApi = {
  // User management
  getUsers: (params?: { limit?: number; skip?: number }) =>
    api.get('/admin/users', { params }),
  
  getUser: (userId: string) =>
    api.get(`/admin/users/${userId}`),
  
  deactivateUser: (userId: string) =>
    api.put(`/admin/users/${userId}/deactivate`),
  
  activateUser: (userId: string) =>
    api.put(`/admin/users/${userId}/activate`),
  
  // System monitoring
  getSystemHealth: () =>
    api.get('/admin/system-health'),
  
  getAnalytics: (days = 7) =>
    api.get(`/admin/analytics?days=${days}`),
  
  // Model management
  getModels: () =>
    api.get('/admin/models'),
  
  getModelDetails: (modelId: string) =>
    api.get(`/admin/models/${modelId}`),
  
  getModelPerformance: (params?: { model_id?: string; days?: number }) =>
    api.get('/admin/model-performance', { params }),
  
  retrainModels: (data?: { 
    model_name?: string; 
    trigger_reason?: string; 
    training_config?: unknown; 
    auto_deploy?: boolean 
  }) =>
    api.post('/admin/retrain-models', data),
  
  getMlJobs: (params?: { limit?: number; status?: string; model_name?: string }) =>
    api.get('/admin/ml-jobs', { params }),
  
  // Alerts and monitoring
  getAlerts: (params?: { severity?: string; status?: string; limit?: number }) =>
    api.get('/admin/alerts', { params }),
  
  acknowledgeAlert: (alertId: string) =>
    api.put(`/admin/alerts/${alertId}/acknowledge`),
  
  // Cache management
  clearCache: (cacheType?: string) =>
    api.delete('/admin/cache/clear', { params: cacheType ? { cache_type: cacheType } : {} }),
}

export default api