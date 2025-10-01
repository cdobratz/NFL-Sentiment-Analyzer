import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { adminApi } from '../services/api'
import LoadingSpinner from '../components/LoadingSpinner'
import { 
  Users, 
  Activity, 
  BarChart3, 
  RefreshCw, 
  Server, 
  AlertTriangle, 
  CheckCircle, 
  XCircle, 
  Clock, 
  Cpu, 
  Database, 
  Trash2,
  Eye,
  Settings,
  TrendingUp,
  Brain,
  Zap
} from 'lucide-react'
import toast from 'react-hot-toast'

export default function AdminPanel() {
  const [activeTab, setActiveTab] = useState<'overview' | 'models' | 'users' | 'alerts' | 'jobs'>('overview')
  const [selectedDays, setSelectedDays] = useState(7)
  const queryClient = useQueryClient()

  // Queries
  const { data: systemHealth, isLoading: healthLoading } = useQuery({
    queryKey: ['system-health'],
    queryFn: () => adminApi.getSystemHealth(),
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const { data: analytics, isLoading: analyticsLoading } = useQuery({
    queryKey: ['admin-analytics', selectedDays],
    queryFn: () => adminApi.getAnalytics(selectedDays),
  })

  const { data: users, isLoading: usersLoading } = useQuery({
    queryKey: ['admin-users'],
    queryFn: () => adminApi.getUsers({ limit: 20 }),
  })

  const { data: models, isLoading: modelsLoading } = useQuery({
    queryKey: ['admin-models'],
    queryFn: () => adminApi.getModels(),
    enabled: activeTab === 'models',
  })

  const { data: modelPerformance, isLoading: performanceLoading } = useQuery({
    queryKey: ['model-performance', selectedDays],
    queryFn: () => adminApi.getModelPerformance({ days: selectedDays }),
    enabled: activeTab === 'models',
  })

  const { data: alerts, isLoading: alertsLoading } = useQuery({
    queryKey: ['admin-alerts'],
    queryFn: () => adminApi.getAlerts({ limit: 50 }),
    enabled: activeTab === 'alerts',
  })

  const { data: mlJobs, isLoading: jobsLoading } = useQuery({
    queryKey: ['ml-jobs'],
    queryFn: () => adminApi.getMlJobs({ limit: 20 }),
    enabled: activeTab === 'jobs',
  })

  // Mutations
  const retrainModelsMutation = useMutation({
    mutationFn: (data?: any) => adminApi.retrainModels(data),
    onSuccess: () => {
      toast.success('Model retraining started successfully')
      queryClient.invalidateQueries({ queryKey: ['ml-jobs'] })
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to start model retraining')
    },
  })

  const clearCacheMutation = useMutation({
    mutationFn: (cacheType?: string) => adminApi.clearCache(cacheType),
    onSuccess: (data) => {
      toast.success(data.data.message)
      queryClient.invalidateQueries({ queryKey: ['system-health'] })
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to clear cache')
    },
  })

  const acknowledgeAlertMutation = useMutation({
    mutationFn: (alertId: string) => adminApi.acknowledgeAlert(alertId),
    onSuccess: () => {
      toast.success('Alert acknowledged')
      queryClient.invalidateQueries({ queryKey: ['admin-alerts'] })
    },
    onError: (error: any) => {
      toast.error(error.response?.data?.detail || 'Failed to acknowledge alert')
    },
  })

  const isLoading = healthLoading || analyticsLoading || usersLoading

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  const getHealthStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'text-green-600 bg-green-50'
      case 'degraded':
        return 'text-yellow-600 bg-yellow-50'
      case 'unhealthy':
        return 'text-red-600 bg-red-50'
      default:
        return 'text-gray-600 bg-gray-50'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle className="w-4 h-4 text-green-600" />
      case 'degraded':
        return <AlertTriangle className="w-4 h-4 text-yellow-600" />
      case 'unhealthy':
        return <XCircle className="w-4 h-4 text-red-600" />
      default:
        return <Clock className="w-4 h-4 text-gray-600" />
    }
  }

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'text-red-600 bg-red-50 border-red-200'
      case 'high':
        return 'text-orange-600 bg-orange-50 border-orange-200'
      case 'medium':
        return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'low':
        return 'text-blue-600 bg-blue-50 border-blue-200'
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }



  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Admin Panel</h1>
          <p className="text-gray-600 mt-2">System administration and monitoring</p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            value={selectedDays}
            onChange={(e) => setSelectedDays(Number(e.target.value))}
            className="px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value={1}>Last 24 hours</option>
            <option value={7}>Last 7 days</option>
            <option value={14}>Last 14 days</option>
            <option value={30}>Last 30 days</option>
          </select>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {[
            { id: 'overview', label: 'Overview', icon: BarChart3 },
            { id: 'models', label: 'Models', icon: Brain },
            { id: 'users', label: 'Users', icon: Users },
            { id: 'alerts', label: 'Alerts', icon: AlertTriangle },
            { id: 'jobs', label: 'ML Jobs', icon: Settings },
          ].map((tab) => {
            const Icon = tab.icon
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center space-x-2 py-2 px-1 border-b-2 font-medium text-sm ${
                  activeTab === tab.id
                    ? 'border-blue-500 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span>{tab.label}</span>
              </button>
            )
          })}
        </nav>
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* System Health */}
          <div className="card">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold text-gray-900">System Health</h2>
              <div className="flex items-center space-x-2">
                {getStatusIcon(systemHealth?.data?.overall_status || 'unknown')}
                <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                  getHealthStatusColor(systemHealth?.data?.overall_status || 'unknown')
                }`}>
                  {systemHealth?.data?.overall_status || 'Unknown'}
                </span>
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {systemHealth?.data?.services && Object.entries(systemHealth.data.services).map(([service, status]: [string, any]) => (
                <div key={service} className="p-4 bg-gray-50 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <h3 className="text-sm font-medium text-gray-900 capitalize flex items-center space-x-2">
                      {service === 'mongodb' && <Database className="w-4 h-4" />}
                      {service === 'redis' && <Zap className="w-4 h-4" />}
                      {service === 'mlops' && <Brain className="w-4 h-4" />}
                      <span>{service}</span>
                    </h3>
                    <div className="flex items-center space-x-1">
                      {getStatusIcon(status.status)}
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        getHealthStatusColor(status.status)
                      }`}>
                        {status.status}
                      </span>
                    </div>
                  </div>
                  {status.response_time_ms && (
                    <p className="text-xs text-gray-600">Response: {status.response_time_ms}ms</p>
                  )}
                  {status.error && (
                    <p className="text-xs text-red-600 mt-1">{status.error}</p>
                  )}
                  {status.database_size_mb && (
                    <p className="text-xs text-gray-600">Size: {status.database_size_mb}MB</p>
                  )}
                  {status.memory_used_mb && (
                    <p className="text-xs text-gray-600">Memory: {status.memory_used_mb}MB</p>
                  )}
                </div>
              ))}
            </div>

            {/* System Resources */}
            {systemHealth?.data?.system_resources && (
              <div className="mt-6">
                <h3 className="text-lg font-medium text-gray-900 mb-3">System Resources</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Cpu className="w-4 h-4 text-blue-600" />
                        <span className="text-sm font-medium">CPU Usage</span>
                      </div>
                      <span className="text-sm font-bold">
                        {systemHealth.data.system_resources.cpu_usage_percent?.toFixed(1)}%
                      </span>
                    </div>
                    <div className="mt-2 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-600 h-2 rounded-full" 
                        style={{ width: `${systemHealth.data.system_resources.cpu_usage_percent}%` }}
                      />
                    </div>
                  </div>
                  
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Server className="w-4 h-4 text-green-600" />
                        <span className="text-sm font-medium">Memory</span>
                      </div>
                      <span className="text-sm font-bold">
                        {systemHealth.data.system_resources.memory_usage_percent?.toFixed(1)}%
                      </span>
                    </div>
                    <div className="mt-2 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-600 h-2 rounded-full" 
                        style={{ width: `${systemHealth.data.system_resources.memory_usage_percent}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-600 mt-1">
                      {systemHealth.data.system_resources.memory_available_gb?.toFixed(1)}GB available
                    </p>
                  </div>
                  
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Database className="w-4 h-4 text-purple-600" />
                        <span className="text-sm font-medium">Disk</span>
                      </div>
                      <span className="text-sm font-bold">
                        {systemHealth.data.system_resources.disk_usage_percent?.toFixed(1)}%
                      </span>
                    </div>
                    <div className="mt-2 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-purple-600 h-2 rounded-full" 
                        style={{ width: `${systemHealth.data.system_resources.disk_usage_percent}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-600 mt-1">
                      {systemHealth.data.system_resources.disk_free_gb?.toFixed(1)}GB free
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Analytics Overview */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div className="card">
              <div className="flex items-center">
                <div className="p-2 bg-blue-50 rounded-lg">
                  <Users className="w-6 h-6 text-blue-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Total Users</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {analytics?.data?.users?.total || 0}
                  </p>
                  <p className="text-xs text-gray-500">
                    {analytics?.data?.users?.new || 0} new in period
                  </p>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="flex items-center">
                <div className="p-2 bg-green-50 rounded-lg">
                  <Activity className="w-6 h-6 text-green-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Active Users</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {analytics?.data?.users?.active || 0}
                  </p>
                  <p className="text-xs text-gray-500">Currently active</p>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="flex items-center">
                <div className="p-2 bg-purple-50 rounded-lg">
                  <BarChart3 className="w-6 h-6 text-purple-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Analyses</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {analytics?.data?.sentiment_analyses?.recent || 0}
                  </p>
                  <p className="text-xs text-gray-500">In period</p>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="flex items-center">
                <div className="p-2 bg-orange-50 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-orange-600" />
                </div>
                <div className="ml-4">
                  <p className="text-sm font-medium text-gray-600">Accuracy</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {analytics?.data?.model_performance?.[0]?.avg_accuracy 
                      ? `${(analytics.data.model_performance[0].avg_accuracy * 100).toFixed(1)}%`
                      : 'N/A'
                    }
                  </p>
                  <p className="text-xs text-gray-500">Model performance</p>
                </div>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="card">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Quick Actions</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <button 
                onClick={() => retrainModelsMutation.mutate(undefined)}
                disabled={retrainModelsMutation.isPending}
                className="flex items-center space-x-3 p-4 text-left bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors disabled:opacity-50"
              >
                <RefreshCw className={`w-5 h-5 text-blue-600 ${retrainModelsMutation.isPending ? 'animate-spin' : ''}`} />
                <div>
                  <p className="text-sm font-medium text-gray-900">Retrain Models</p>
                  <p className="text-xs text-gray-500">Trigger ML model retraining</p>
                </div>
              </button>
              
              <button 
                onClick={() => clearCacheMutation.mutate(undefined)}
                disabled={clearCacheMutation.isPending}
                className="flex items-center space-x-3 p-4 text-left bg-green-50 hover:bg-green-100 rounded-lg transition-colors disabled:opacity-50"
              >
                <Trash2 className="w-5 h-5 text-green-600" />
                <div>
                  <p className="text-sm font-medium text-gray-900">Clear Cache</p>
                  <p className="text-xs text-gray-500">Clear application cache</p>
                </div>
              </button>
              
              <button 
                onClick={() => queryClient.invalidateQueries({ queryKey: ['system-health'] })}
                className="flex items-center space-x-3 p-4 text-left bg-purple-50 hover:bg-purple-100 rounded-lg transition-colors"
              >
                <RefreshCw className="w-5 h-5 text-purple-600" />
                <div>
                  <p className="text-sm font-medium text-gray-900">Refresh Data</p>
                  <p className="text-xs text-gray-500">Reload all dashboard data</p>
                </div>
              </button>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'models' && (
        <div className="space-y-6">
          {modelsLoading || performanceLoading ? (
            <div className="flex items-center justify-center h-64">
              <LoadingSpinner size="lg" />
            </div>
          ) : (
            <>
              {/* Model Performance Summary */}
              {modelPerformance?.data?.summary && (
                <div className="card">
                  <h2 className="text-xl font-semibold text-gray-900 mb-4">Model Performance Summary</h2>
                  <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                    <div className="text-center">
                      <p className="text-2xl font-bold text-blue-600">
                        {(modelPerformance.data.summary.avg_accuracy * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-gray-600">Accuracy</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-green-600">
                        {(modelPerformance.data.summary.avg_f1_score * 100).toFixed(1)}%
                      </p>
                      <p className="text-sm text-gray-600">F1 Score</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-purple-600">
                        {modelPerformance.data.summary.avg_response_time_ms.toFixed(0)}ms
                      </p>
                      <p className="text-sm text-gray-600">Avg Response</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-orange-600">
                        {modelPerformance.data.summary.total_predictions.toLocaleString()}
                      </p>
                      <p className="text-sm text-gray-600">Predictions</p>
                    </div>
                    <div className="text-center">
                      <p className="text-2xl font-bold text-red-600">
                        {(modelPerformance.data.summary.avg_error_rate * 100).toFixed(2)}%
                      </p>
                      <p className="text-sm text-gray-600">Error Rate</p>
                    </div>
                  </div>
                </div>
              )}

              {/* Available Models */}
              <div className="card">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">Available Models</h2>
                {models?.data?.models && models.data.models.length > 0 ? (
                  <div className="space-y-4">
                    {models.data.models.map((model: any, index: number) => (
                      <div key={index} className="p-4 border border-gray-200 rounded-lg">
                        <div className="flex items-center justify-between">
                          <div>
                            <h3 className="text-lg font-medium text-gray-900">{model.modelId || model.id}</h3>
                            <p className="text-sm text-gray-600">{model.pipeline_tag || 'Sentiment Analysis'}</p>
                          </div>
                          <div className="flex items-center space-x-2">
                            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                              Active
                            </span>
                          </div>
                        </div>
                        {model.downloads && (
                          <p className="text-xs text-gray-500 mt-2">
                            {model.downloads.toLocaleString()} downloads
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500 text-center py-8">No models found</p>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {activeTab === 'users' && (
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">User Management</h2>
          {usersLoading ? (
            <div className="flex items-center justify-center h-32">
              <LoadingSpinner />
            </div>
          ) : (
            <div className="space-y-3">
              {users?.data && users.data.length > 0 ? (
                users.data.map((user: any) => (
                  <div key={user.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className="w-10 h-10 bg-gray-200 rounded-full flex items-center justify-center">
                        <Users className="w-5 h-5 text-gray-600" />
                      </div>
                      <div>
                        <p className="text-sm font-medium text-gray-900">{user.username}</p>
                        <p className="text-xs text-gray-500">{user.email}</p>
                        <p className="text-xs text-gray-400">
                          Created: {new Date(user.created_at).toLocaleDateString()}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-3">
                      <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                        user.is_active 
                          ? 'bg-green-100 text-green-800' 
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {user.is_active ? 'Active' : 'Inactive'}
                      </span>
                      <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800 capitalize">
                        {user.role}
                      </span>
                      <button className="p-1 text-gray-400 hover:text-gray-600">
                        <Eye className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-gray-500 text-center py-8">No users found</p>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === 'alerts' && (
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">System Alerts</h2>
          {alertsLoading ? (
            <div className="flex items-center justify-center h-32">
              <LoadingSpinner />
            </div>
          ) : (
            <div className="space-y-3">
              {alerts?.data?.alerts && alerts.data.alerts.length > 0 ? (
                alerts.data.alerts.map((alert: any) => (
                  <div key={alert.alert_id} className={`p-4 border rounded-lg ${getSeverityColor(alert.severity)}`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <AlertTriangle className="w-5 h-5" />
                        <div>
                          <p className="text-sm font-medium">{alert.message}</p>
                          <p className="text-xs opacity-75">{alert.description}</p>
                          <p className="text-xs opacity-60 mt-1">
                            {new Date(alert.created_at).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getSeverityColor(alert.severity)}`}>
                          {alert.severity}
                        </span>
                        {alert.status === 'active' && (
                          <button
                            onClick={() => acknowledgeAlertMutation.mutate(alert.alert_id)}
                            className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700"
                          >
                            Acknowledge
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-center py-8">
                  <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-2" />
                  <p className="text-gray-500">No active alerts</p>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {activeTab === 'jobs' && (
        <div className="card">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">ML Jobs</h2>
          {jobsLoading ? (
            <div className="flex items-center justify-center h-32">
              <LoadingSpinner />
            </div>
          ) : (
            <div className="space-y-3">
              {mlJobs?.data?.jobs && mlJobs.data.jobs.length > 0 ? (
                mlJobs.data.jobs.map((job: any) => (
                  <div key={job.job_id} className="p-4 border border-gray-200 rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="w-10 h-10 bg-blue-50 rounded-full flex items-center justify-center">
                          <Settings className="w-5 h-5 text-blue-600" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-gray-900">
                            {job.model_name || 'Model Retraining'}
                          </p>
                          <p className="text-xs text-gray-500">{job.trigger_reason}</p>
                          <p className="text-xs text-gray-400">
                            Started: {new Date(job.created_at).toLocaleString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center space-x-3">
                        <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                          job.status === 'completed' 
                            ? 'bg-green-100 text-green-800'
                            : job.status === 'running'
                            ? 'bg-blue-100 text-blue-800'
                            : job.status === 'failed'
                            ? 'bg-red-100 text-red-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}>
                          {job.status}
                        </span>
                        <button className="p-1 text-gray-400 hover:text-gray-600">
                          <Eye className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-gray-500 text-center py-8">No ML jobs found</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}