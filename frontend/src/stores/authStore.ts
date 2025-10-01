import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { authApi } from '../services/api'
import toast from 'react-hot-toast'

interface User {
  id: string
  email: string
  username: string
  role: 'user' | 'admin'
  is_active: boolean
  created_at: string
  last_login?: string
  preferences: Record<string, any>
}

interface AuthState {
  user: User | null
  token: string | null
  isLoading: boolean
  isAuthenticated: boolean
  login: (email: string, password: string) => Promise<boolean>
  register: (userData: {
    email: string
    username: string
    password: string
  }) => Promise<boolean>
  logout: () => Promise<void>
  checkAuth: () => Promise<void>
  refreshToken: () => Promise<boolean>
  updateProfile: (userData: {
    username: string
    email: string
  }) => Promise<boolean>
  changePassword: (passwordData: {
    currentPassword: string
    newPassword: string
  }) => Promise<boolean>
  setupTokenRefresh: (expiresIn: number) => void
}

let refreshTimer: NodeJS.Timeout | null = null

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      token: null,
      isLoading: false,
      isAuthenticated: false,

      login: async (email: string, password: string) => {
        set({ isLoading: true })
        try {
          const response = await authApi.login(email, password)
          const { access_token, expires_in } = response.data
          
          // Get user profile
          const profileResponse = await authApi.getProfile(access_token)
          const user = profileResponse.data
          
          set({
            user,
            token: access_token,
            isAuthenticated: true,
            isLoading: false,
          })
          
          // Set up automatic token refresh
          get().setupTokenRefresh(expires_in)
          
          toast.success('Login successful!')
          return true
        } catch (error: any) {
          set({ isLoading: false })
          const message = error.response?.data?.detail || 'Login failed'
          toast.error(message)
          return false
        }
      },

      register: async (userData) => {
        set({ isLoading: true })
        try {
          await authApi.register(userData)
          set({ isLoading: false })
          toast.success('Registration successful! Please log in.')
          return true
        } catch (error: any) {
          set({ isLoading: false })
          const message = error.response?.data?.detail || 'Registration failed'
          toast.error(message)
          return false
        }
      },

      logout: async () => {
        const { token } = get()
        
        // Call logout API to blacklist token
        if (token) {
          try {
            await authApi.logout(token)
          } catch (error) {
            // Ignore errors during logout
            console.warn('Logout API call failed:', error)
          }
        }
        
        // Clear refresh timer
        if (refreshTimer) {
          clearTimeout(refreshTimer)
          refreshTimer = null
        }
        
        set({
          user: null,
          token: null,
          isAuthenticated: false,
        })
        toast.success('Logged out successfully')
      },

      checkAuth: async () => {
        const { token } = get()
        if (!token) {
          set({ isLoading: false })
          return
        }

        set({ isLoading: true })
        try {
          const response = await authApi.getProfile(token)
          const user = response.data
          
          set({
            user,
            isAuthenticated: true,
            isLoading: false,
          })
        } catch (error) {
          // Token is invalid, clear auth state
          set({
            user: null,
            token: null,
            isAuthenticated: false,
            isLoading: false,
          })
        }
      },

      refreshToken: async () => {
        const { token } = get()
        if (!token) return false

        try {
          const response = await authApi.refreshToken(token)
          const { access_token, expires_in } = response.data
          
          set({ token: access_token })
          
          // Set up next refresh
          get().setupTokenRefresh(expires_in)
          
          return true
        } catch (error) {
          // Refresh failed, logout user
          get().logout()
          return false
        }
      },

      setupTokenRefresh: (expiresIn: number) => {
        // Clear existing timer
        if (refreshTimer) {
          clearTimeout(refreshTimer)
        }
        
        // Refresh token 5 minutes before expiry
        const refreshTime = (expiresIn - 300) * 1000
        
        if (refreshTime > 0) {
          refreshTimer = setTimeout(() => {
            get().refreshToken()
          }, refreshTime)
        }
      },

      updateProfile: async (userData) => {
        const { token } = get()
        if (!token) return false

        try {
          const response = await authApi.updateProfile(userData, token)
          const updatedUser = response.data
          
          set({ user: updatedUser })
          toast.success('Profile updated successfully!')
          return true
        } catch (error: any) {
          const message = error.response?.data?.detail || 'Failed to update profile'
          toast.error(message)
          return false
        }
      },

      changePassword: async (passwordData) => {
        const { token } = get()
        if (!token) return false

        try {
          await authApi.changePassword(passwordData, token)
          toast.success('Password changed successfully!')
          return true
        } catch (error: any) {
          const message = error.response?.data?.detail || 'Failed to change password'
          toast.error(message)
          return false
        }
      },
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        token: state.token,
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    }
  )
)