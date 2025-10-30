import { createContext, useContext, useEffect, ReactNode } from 'react'
import { useAuthStore } from '../stores/authStore'

interface User {
  id: string
  email: string
  username: string
  role: 'user' | 'admin'
  is_active: boolean
  created_at: string
  last_login?: string
  preferences: Record<string, unknown>
}

interface RegisterData {
  email: string
  username: string
  password: string
}

interface UpdateProfileData {
  username: string
  email: string
}

interface ChangePasswordData {
  currentPassword: string
  newPassword: string
}

interface AuthContextType {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  login: (email: string, password: string) => Promise<boolean>
  register: (userData: RegisterData) => Promise<boolean>
  logout: () => Promise<void>
  updateProfile: (userData: UpdateProfileData) => Promise<boolean>
  changePassword: (passwordData: ChangePasswordData) => Promise<boolean>
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps) {
  const authStore = useAuthStore()

  useEffect(() => {
    // Check authentication status on app load
    authStore.checkAuth()
  }, [])

  const contextValue: AuthContextType = {
    user: authStore.user,
    isAuthenticated: authStore.isAuthenticated,
    isLoading: authStore.isLoading,
    login: authStore.login,
    register: authStore.register,
    logout: authStore.logout,
    updateProfile: authStore.updateProfile,
    changePassword: authStore.changePassword,
  }

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}