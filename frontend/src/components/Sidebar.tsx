import { Link, useLocation } from 'react-router-dom'
import { 
  BarChart3, 
  Users, 
  Calendar, 
  TrendingUp, 
  Settings,
  Home,
  Menu,
  X
} from 'lucide-react'
import { useAuthStore } from '../stores/authStore'
import { cn } from '../utils/cn'
import { useState, useEffect } from 'react'

const navigation = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'Teams', href: '/teams', icon: Users },
  { name: 'Games', href: '/games', icon: Calendar },
  { name: 'Trends', href: '/trends', icon: TrendingUp },
  { name: 'Analytics', href: '/analytics', icon: BarChart3 },
]

const adminNavigation = [
  { name: 'Admin Panel', href: '/admin', icon: Settings },
]

export default function Sidebar() {
  const location = useLocation()
  const { user } = useAuthStore()
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkIsMobile = () => {
      setIsMobile(window.innerWidth < 768)
      if (window.innerWidth >= 768) {
        setIsMobileMenuOpen(false)
      }
    }

    checkIsMobile()
    window.addEventListener('resize', checkIsMobile)

    return () => { window.removeEventListener('resize', checkIsMobile) }
  }, [])

  const NavigationItems = () => (
    <nav className="space-y-2">
      {navigation.map((item) => {
        const isActive = location.pathname === item.href
        return (
          <Link
            key={item.name}
            to={item.href}
            onClick={() => { isMobile && setIsMobileMenuOpen(false) }}
            className={cn(
              'flex items-center space-x-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
              isActive
                ? 'bg-primary-50 text-primary-700 border-r-2 border-primary-700'
                : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
            )}
          >
            <item.icon size={18} />
            <span>{item.name}</span>
          </Link>
        )
      })}
      
      {user?.role === 'admin' && (
        <>
          <div className="border-t border-gray-200 my-4"></div>
          {adminNavigation.map((item) => {
            const isActive = location.pathname === item.href
            return (
              <Link
                key={item.name}
                to={item.href}
                onClick={() => { isMobile && setIsMobileMenuOpen(false) }}
                className={cn(
                  'flex items-center space-x-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors',
                  isActive
                    ? 'bg-primary-50 text-primary-700 border-r-2 border-primary-700'
                    : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                )}
              >
                <item.icon size={18} />
                <span>{item.name}</span>
              </Link>
            )
          })}
        </>
      )}
    </nav>
  )

  if (isMobile) {
    return (
      <>
        {/* Mobile menu button */}
        <button
          onClick={() => { setIsMobileMenuOpen(!isMobileMenuOpen) }}
          className="fixed top-4 left-4 z-50 p-2 bg-white rounded-lg shadow-md border border-gray-200 md:hidden"
        >
          {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>

        {/* Mobile menu overlay */}
        {isMobileMenuOpen && (
          <div
            className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden"
            onClick={() => { setIsMobileMenuOpen(false) }}
          />
        )}

        {/* Mobile sidebar */}
        <div
          className={cn(
            'fixed left-0 top-0 h-full w-64 bg-white shadow-lg transform transition-transform duration-300 ease-in-out z-40 md:hidden',
            isMobileMenuOpen ? 'translate-x-0' : '-translate-x-full'
          )}
        >
          <div className="p-6 pt-16">
            <NavigationItems />
          </div>
        </div>
      </>
    )
  }

  // Desktop sidebar
  return (
    <div className="w-64 bg-white shadow-sm border-r border-gray-200 min-h-screen">
      <div className="p-6">
        <NavigationItems />
      </div>
    </div>
  )
}