import { useEffect, useState } from 'react'
import { useAuth } from '../hooks/useAuth'
import toast from 'react-hot-toast'

interface SessionTimeoutProps {
  warningTime?: number // Minutes before expiry to show warning
  children: React.ReactNode
}

export default function SessionTimeout({ warningTime = 5, children }: SessionTimeoutProps) {
  const { isAuthenticated, logout, refreshToken } = useAuth()
  const [showWarning, setShowWarning] = useState(false)
  const [timeLeft, setTimeLeft] = useState(0)

  useEffect(() => {
    if (!isAuthenticated) return

    let warningTimer: NodeJS.Timeout
    let countdownTimer: NodeJS.Timeout
    let logoutTimer: NodeJS.Timeout

    const setupTimers = () => {
      // Clear existing timers
      clearTimeout(warningTimer)
      clearTimeout(countdownTimer)
      clearTimeout(logoutTimer)

      // Token expires in 30 minutes (from backend config)
      const tokenExpiryTime = 30 * 60 * 1000 // 30 minutes in ms
      const warningTimeMs = warningTime * 60 * 1000 // Warning time in ms

      // Show warning before expiry
      warningTimer = setTimeout(() => {
        setShowWarning(true)
        setTimeLeft(warningTime * 60) // Set countdown in seconds

        // Start countdown
        countdownTimer = setInterval(() => {
          setTimeLeft((prev) => {
            if (prev <= 1) {
              clearInterval(countdownTimer)
              return 0
            }
            return prev - 1
          })
        }, 1000)

        // Auto logout after warning period
        logoutTimer = setTimeout(() => {
          setShowWarning(false)
          logout()
          toast.error('Session expired. Please log in again.')
        }, warningTimeMs)
      }, tokenExpiryTime - warningTimeMs)
    }

    setupTimers()

    return () => {
      clearTimeout(warningTimer)
      clearTimeout(countdownTimer)
      clearTimeout(logoutTimer)
    }
  }, [isAuthenticated, warningTime, logout])

  const handleExtendSession = async () => {
    const success = await refreshToken()
    if (success) {
      setShowWarning(false)
      setTimeLeft(0)
      toast.success('Session extended successfully')
    } else {
      toast.error('Failed to extend session. Please log in again.')
    }
  }

  const handleLogoutNow = () => {
    setShowWarning(false)
    logout()
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <>
      {children}
      
      {showWarning && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <div className="flex items-center space-x-3 mb-4">
              <div className="w-10 h-10 bg-yellow-100 rounded-full flex items-center justify-center">
                <svg className="w-6 h-6 text-yellow-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.268 18.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Session Expiring</h3>
                <p className="text-sm text-gray-600">Your session will expire in {formatTime(timeLeft)}</p>
              </div>
            </div>
            
            <p className="text-gray-700 mb-6">
              Would you like to extend your session or log out now?
            </p>
            
            <div className="flex space-x-3">
              <button
                onClick={handleExtendSession}
                className="flex-1 btn-primary"
              >
                Extend Session
              </button>
              <button
                onClick={handleLogoutNow}
                className="flex-1 btn-secondary"
              >
                Log Out
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}