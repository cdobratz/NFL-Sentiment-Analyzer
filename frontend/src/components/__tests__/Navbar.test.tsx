import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '../../test/utils'
import Navbar from '../Navbar'
import { useAuthStore } from '../../stores/authStore'

// Mock the auth store
vi.mock('../../stores/authStore', () => ({
  useAuthStore: vi.fn(),
}))

describe('Navbar', () => {
  const mockLogout = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders unauthenticated state correctly', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: null,
      isAuthenticated: false,
      logout: mockLogout,
    })

    render(<Navbar />)

    expect(screen.getByText('NFL')).toBeInTheDocument()
    expect(screen.getByText('Sentiment Analyzer')).toBeInTheDocument()
    expect(screen.getByText('Login')).toBeInTheDocument()
    expect(screen.getByText('Sign Up')).toBeInTheDocument()
  })

  it('renders authenticated state correctly', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'USER',
      },
      isAuthenticated: true,
      logout: mockLogout,
    })

    render(<Navbar />)

    expect(screen.getByText('Welcome, testuser')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /logout/i })).toBeInTheDocument()
    expect(screen.queryByText('Login')).not.toBeInTheDocument()
    expect(screen.queryByText('Sign Up')).not.toBeInTheDocument()
  })

  it('shows admin link for admin users', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: {
        id: '1',
        username: 'admin',
        email: 'admin@example.com',
        role: 'admin',
      },
      isAuthenticated: true,
      logout: mockLogout,
    })

    render(<Navbar />)

    expect(screen.getByRole('link', { name: /settings/i })).toBeInTheDocument()
  })

  it('does not show admin link for regular users', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'USER',
      },
      isAuthenticated: true,
      logout: mockLogout,
    })

    render(<Navbar />)

    expect(screen.queryByRole('link', { name: /settings/i })).not.toBeInTheDocument()
  })

  it('handles logout click', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'USER',
      },
      isAuthenticated: true,
      logout: mockLogout,
    })

    render(<Navbar />)

    const logoutButton = screen.getByRole('button', { name: /logout/i })
    fireEvent.click(logoutButton)

    expect(mockLogout).toHaveBeenCalledTimes(1)
  })

  it('has correct navigation links', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: null,
      isAuthenticated: false,
      logout: mockLogout,
    })

    render(<Navbar />)

    const homeLink = screen.getByRole('link', { name: /nfl sentiment analyzer/i })
    expect(homeLink).toHaveAttribute('href', '/')

    const loginLink = screen.getByRole('link', { name: /login/i })
    expect(loginLink).toHaveAttribute('href', '/login')

    const signUpLink = screen.getByRole('link', { name: /sign up/i })
    expect(signUpLink).toHaveAttribute('href', '/register')
  })

  it('has correct authenticated navigation links', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'USER',
      },
      isAuthenticated: true,
      logout: mockLogout,
    })

    render(<Navbar />)

    const profileLink = screen.getByRole('link', { name: /user/i })
    expect(profileLink).toHaveAttribute('href', '/profile')
  })

  it('has correct admin navigation links', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: {
        id: '1',
        username: 'admin',
        email: 'admin@example.com',
        role: 'admin',
      },
      isAuthenticated: true,
      logout: mockLogout,
    })

    render(<Navbar />)

    const adminLink = screen.getByRole('link', { name: /settings/i })
    expect(adminLink).toHaveAttribute('href', '/admin')
  })

  it('has proper accessibility attributes', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'USER',
      },
      isAuthenticated: true,
      logout: mockLogout,
    })

    render(<Navbar />)

    const nav = screen.getByRole('navigation')
    expect(nav).toBeInTheDocument()

    const logoutButton = screen.getByRole('button', { name: /logout/i })
    expect(logoutButton).toBeInTheDocument()

    const profileLink = screen.getByRole('link', { name: /user/i })
    expect(profileLink).toBeInTheDocument()
  })

  it('applies correct CSS classes for styling', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: null,
      isAuthenticated: false,
      logout: mockLogout,
    })

    const { container } = render(<Navbar />)

    const nav = container.querySelector('nav')
    expect(nav).toHaveClass('bg-white', 'shadow-sm', 'border-b', 'border-gray-200')
  })

  it('shows correct logo and branding', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: null,
      isAuthenticated: false,
      logout: mockLogout,
    })

    render(<Navbar />)

    const logo = screen.getByText('NFL')
    expect(logo).toBeInTheDocument()
    expect(logo.closest('div')).toHaveClass('bg-primary-600')

    const brandText = screen.getByText('Sentiment Analyzer')
    expect(brandText).toBeInTheDocument()
  })

  it('handles keyboard navigation', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: {
        id: '1',
        username: 'testuser',
        email: 'test@example.com',
        role: 'USER',
      },
      isAuthenticated: true,
      logout: mockLogout,
    })

    render(<Navbar />)

    const logoutButton = screen.getByRole('button', { name: /logout/i })
    
    // Focus the button
    logoutButton.focus()
    expect(logoutButton).toHaveFocus()

    // Press Enter
    fireEvent.keyDown(logoutButton, { key: 'Enter', code: 'Enter' })
    fireEvent.click(logoutButton)

    expect(mockLogout).toHaveBeenCalled()
  })

  it('displays user information correctly', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: {
        id: '1',
        username: 'john_doe',
        email: 'john@example.com',
        role: 'USER',
      },
      isAuthenticated: true,
      logout: mockLogout,
    })

    render(<Navbar />)

    expect(screen.getByText('Welcome, john_doe')).toBeInTheDocument()
  })

  it('handles missing user data gracefully', () => {
    vi.mocked(useAuthStore).mockReturnValue({
      user: {
        id: '1',
        username: '',
        email: 'test@example.com',
        role: 'USER',
      },
      isAuthenticated: true,
      logout: mockLogout,
    })

    render(<Navbar />)

    expect(screen.getByText('Welcome, ')).toBeInTheDocument()
  })
})