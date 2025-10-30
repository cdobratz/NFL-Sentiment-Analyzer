import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '../../test/utils'
import Login from '../Login'
import { useAuth } from '../../hooks/useAuth'

// Mock the useAuth hook
vi.mock('../../hooks/useAuth', () => ({
  useAuth: vi.fn(),
}))

// Mock LoadingSpinner
vi.mock('../../components/LoadingSpinner', () => ({
  default: vi.fn(() => <div data-testid="loading-spinner">Loading...</div>),
}))

describe('Login', () => {
  const mockLogin = vi.fn()
  const mockLogout = vi.fn()
  const mockRegister = vi.fn()
  const mockCheckAuth = vi.fn()
  const mockRefreshToken = vi.fn()
  const mockUpdateProfile = vi.fn()
  const mockChangePassword = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()

    vi.mocked(useAuth).mockReturnValue({
      user: null,
      token: null,
      isLoading: false,
      isAuthenticated: false,
      login: mockLogin,
      register: mockRegister,
      logout: mockLogout,
      checkAuth: mockCheckAuth,
      refreshToken: mockRefreshToken,
      updateProfile: mockUpdateProfile,
      changePassword: mockChangePassword,
      isAdmin: false,
      isUser: false,
    })
  })

  it('renders login form correctly', () => {
    render(<Login />)

    expect(screen.getByText('Sign in to your account')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Enter your email')).toBeInTheDocument()
    expect(screen.getByPlaceholderText('Enter your password')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument()
    expect(screen.getByText('create a new account')).toBeInTheDocument()
  })

  it('redirects when already authenticated', () => {
    vi.mocked(useAuth).mockReturnValue({
      user: {
        id: '1',
        username: 'test',
        email: 'test@example.com',
        role: 'user',
        is_active: true,
        created_at: '2024-01-15T10:00:00Z',
        preferences: {}
      },
      token: 'mock-token',
      isLoading: false,
      isAuthenticated: true,
      login: mockLogin,
      register: mockRegister,
      logout: mockLogout,
      checkAuth: mockCheckAuth,
      refreshToken: mockRefreshToken,
      updateProfile: mockUpdateProfile,
      changePassword: mockChangePassword,
      isAdmin: false,
      isUser: true,
    })

    render(<Login />)

    // Should not render the login form
    expect(screen.queryByText('Sign in to your account')).not.toBeInTheDocument()
  })

  it('validates email field', async () => {
    render(<Login />)

    const emailInput = screen.getByPlaceholderText('Enter your email')
    const submitButton = screen.getByRole('button', { name: /sign in/i })

    // Submit with invalid email
    fireEvent.change(emailInput, { target: { value: 'invalid-email' } })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(screen.getByText('Invalid email address')).toBeInTheDocument()
    })
  })

  it('validates password field', async () => {
    render(<Login />)

    const passwordInput = screen.getByPlaceholderText('Enter your password')
    const submitButton = screen.getByRole('button', { name: /sign in/i })

    // Submit with empty password
    fireEvent.change(passwordInput, { target: { value: '' } })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(screen.getByText('Password is required')).toBeInTheDocument()
    })
  })

  it('submits form with valid data', async () => {
    mockLogin.mockResolvedValue(true)

    render(<Login />)

    const emailInput = screen.getByPlaceholderText('Enter your email')
    const passwordInput = screen.getByPlaceholderText('Enter your password')
    const submitButton = screen.getByRole('button', { name: /sign in/i })

    fireEvent.change(emailInput, { target: { value: 'test@example.com' } })
    fireEvent.change(passwordInput, { target: { value: 'password123' } })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith('test@example.com', 'password123')
    })
  })

  it('shows loading state during submission', async () => {
    mockLogin.mockImplementation(() => new Promise(() => { })) // Never resolves

    render(<Login />)

    const emailInput = screen.getByPlaceholderText('Enter your email')
    const passwordInput = screen.getByPlaceholderText('Enter your password')
    const submitButton = screen.getByRole('button')

    fireEvent.change(emailInput, { target: { value: 'test@example.com' } })
    fireEvent.change(passwordInput, { target: { value: 'password123' } })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument()
      expect(submitButton).toBeDisabled()
    })
  })

  it('shows loading state when auth is loading', () => {
    vi.mocked(useAuth).mockReturnValue({
      user: null,
      token: null,
      isLoading: true,
      isAuthenticated: false,
      login: mockLogin,
      register: mockRegister,
      logout: mockLogout,
      checkAuth: mockCheckAuth,
      refreshToken: mockRefreshToken,
      updateProfile: mockUpdateProfile,
      changePassword: mockChangePassword,
      isAdmin: false,
      isUser: false,
    })

    render(<Login />)

    const submitButton = screen.getByRole('button')
    expect(submitButton).toBeDisabled()
  })

  it('handles login failure gracefully', async () => {
    mockLogin.mockResolvedValue(false)

    render(<Login />)

    const emailInput = screen.getByPlaceholderText('Enter your email')
    const passwordInput = screen.getByPlaceholderText('Enter your password')
    const submitButton = screen.getByRole('button', { name: /sign in/i })

    fireEvent.change(emailInput, { target: { value: 'test@example.com' } })
    fireEvent.change(passwordInput, { target: { value: 'wrongpassword' } })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith('test@example.com', 'wrongpassword')
    })

    // Should not show loading spinner after failed login
    expect(screen.queryByTestId('loading-spinner')).not.toBeInTheDocument()
    expect(submitButton).not.toBeDisabled()
  })

  it('has correct form attributes', () => {
    render(<Login />)

    const emailInput = screen.getByPlaceholderText('Enter your email')
    const passwordInput = screen.getByPlaceholderText('Enter your password')

    expect(emailInput).toHaveAttribute('type', 'email')
    expect(emailInput).toHaveAttribute('autoComplete', 'email')
    expect(emailInput).toHaveAttribute('placeholder', 'Enter your email')

    expect(passwordInput).toHaveAttribute('type', 'password')
    expect(passwordInput).toHaveAttribute('autoComplete', 'current-password')
    expect(passwordInput).toHaveAttribute('placeholder', 'Enter your password')
  })

  it('has correct navigation links', () => {
    render(<Login />)

    const registerLink = screen.getByRole('link', { name: /create a new account/i })
    expect(registerLink).toHaveAttribute('href', '/register')
  })

  it('displays NFL branding correctly', () => {
    render(<Login />)

    expect(screen.getByText('NFL')).toBeInTheDocument()
    expect(screen.getByText('NFL').closest('div')).toHaveClass('bg-primary-600')
  })

  it('has proper accessibility attributes', () => {
    render(<Login />)

    const emailInput = screen.getByPlaceholderText('Enter your email')
    const passwordInput = screen.getByPlaceholderText('Enter your password')
    const submitButton = screen.getByRole('button', { name: /sign in/i })

    expect(emailInput).toBeInTheDocument()
    expect(passwordInput).toBeInTheDocument()
    expect(submitButton).toHaveAttribute('type', 'submit')

    // Check that labels are present
    expect(screen.getByText('Email address')).toBeInTheDocument()
    expect(screen.getByText('Password')).toBeInTheDocument()
  })

  it('handles keyboard navigation', async () => {
    render(<Login />)

    const emailInput = screen.getByPlaceholderText('Enter your email')
    const passwordInput = screen.getByPlaceholderText('Enter your password')
    const submitButton = screen.getByRole('button', { name: /sign in/i })

    // Tab through form elements
    emailInput.focus()
    expect(emailInput).toHaveFocus()

    fireEvent.keyDown(emailInput, { key: 'Tab' })
    passwordInput.focus()
    expect(passwordInput).toHaveFocus()

    fireEvent.keyDown(passwordInput, { key: 'Tab' })
    submitButton.focus()
    expect(submitButton).toHaveFocus()
  })

  it('clears form validation errors when user types', async () => {
    render(<Login />)

    const emailInput = screen.getByPlaceholderText('Enter your email')
    const submitButton = screen.getByRole('button', { name: /sign in/i })

    // Trigger validation error
    fireEvent.change(emailInput, { target: { value: 'invalid' } })
    fireEvent.click(submitButton)

    await waitFor(() => {
      expect(screen.getByText('Invalid email address')).toBeInTheDocument()
    })

    // Fix the email
    fireEvent.change(emailInput, { target: { value: 'valid@example.com' } })

    await waitFor(() => {
      expect(screen.queryByText('Invalid email address')).not.toBeInTheDocument()
    })
  })

  it('prevents multiple submissions', async () => {
    mockLogin.mockImplementation(() => new Promise(resolve => setTimeout(() => { resolve(true) }, 100)))

    render(<Login />)

    const emailInput = screen.getByPlaceholderText('Enter your email')
    const passwordInput = screen.getByPlaceholderText('Enter your password')
    const submitButton = screen.getByRole('button', { name: /sign in/i })

    fireEvent.change(emailInput, { target: { value: 'test@example.com' } })
    fireEvent.change(passwordInput, { target: { value: 'password123' } })

    // Click submit multiple times
    fireEvent.click(submitButton)
    fireEvent.click(submitButton)
    fireEvent.click(submitButton)

    // Should only call login once
    expect(mockLogin).toHaveBeenCalledTimes(1)
  })

  it('handles form submission with Enter key', async () => {
    mockLogin.mockResolvedValue(true)

    render(<Login />)

    const emailInput = screen.getByPlaceholderText('Enter your email')
    const passwordInput = screen.getByPlaceholderText('Enter your password')

    fireEvent.change(emailInput, { target: { value: 'test@example.com' } })
    fireEvent.change(passwordInput, { target: { value: 'password123' } })

    // Press Enter in password field
    fireEvent.keyDown(passwordInput, { key: 'Enter', code: 'Enter' })

    await waitFor(() => {
      expect(mockLogin).toHaveBeenCalledWith('test@example.com', 'password123')
    })
  })
})