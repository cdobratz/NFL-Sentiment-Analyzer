import { useAuthStore } from '../stores/authStore'

export function useAuth() {
  const {
    user,
    token,
    isLoading,
    isAuthenticated,
    login,
    register,
    logout,
    checkAuth,
    refreshToken,
    updateProfile,
    changePassword,
  } = useAuthStore()

  return {
    user,
    token,
    isLoading,
    isAuthenticated,
    login,
    register,
    logout,
    checkAuth,
    refreshToken,
    updateProfile,
    changePassword,
    isAdmin: user?.role === 'admin',
    isUser: user?.role === 'user',
  }
}

export function useRequireAuth() {
  const auth = useAuth()
  
  if (!auth.isAuthenticated) {
    throw new Error('Authentication required')
  }
  
  return auth
}

export function useRequireAdmin() {
  const auth = useRequireAuth()
  
  if (!auth.isAdmin) {
    throw new Error('Admin privileges required')
  }
  
  return auth
}