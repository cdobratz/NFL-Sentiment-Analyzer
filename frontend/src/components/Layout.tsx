import { Outlet } from 'react-router-dom'
import { useState, useEffect } from 'react'
import Navbar from './Navbar'
import Sidebar from './Sidebar'

export default function Layout() {
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkIsMobile = () => {
      setIsMobile(window.innerWidth < 768)
    }

    checkIsMobile()
    window.addEventListener('resize', checkIsMobile)

    return () => { window.removeEventListener('resize', checkIsMobile) }
  }, [])

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <div className="flex">
        <Sidebar />
        <main className={`flex-1 ${isMobile ? 'p-4' : 'p-6'}`}>
          <Outlet />
        </main>
      </div>
    </div>
  )
}