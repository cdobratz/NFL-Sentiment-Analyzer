import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { renderHook, act } from '@testing-library/react'
import useWebSocket from '../useWebSocket'
import { useAuthStore } from '../../stores/authStore'

// Mock the auth store
vi.mock('../../stores/authStore', () => ({
  useAuthStore: vi.fn(),
}))

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3

  readyState = MockWebSocket.CONNECTING
  onopen: ((event: Event) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null

  constructor(public url: string) {
    // Simulate async connection
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN
      this.onopen?.(new Event('open'))
    }, 10)
  }

  send(_data: string) {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('WebSocket is not open')
    }
  }

  close() {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.(new CloseEvent('close'))
  }

  // Helper methods for testing
  simulateMessage(data: any) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }))
    }
  }

  simulateError() {
    this.onerror?.(new Event('error'))
  }

  simulateClose(code = 1000, reason = '') {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.(new CloseEvent('close', { code, reason }))
  }
}

// Replace global WebSocket
global.WebSocket = MockWebSocket as any

describe('useWebSocket', () => {
  let mockAuthStore: any

  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()

    mockAuthStore = {
      token: 'test-token',
    }
    vi.mocked(useAuthStore).mockReturnValue(mockAuthStore)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('initializes with correct default state', () => {
    const { result } = renderHook(() => useWebSocket('/test'))

    expect(result.current.isConnected).toBe(false)
    expect(result.current.isConnecting).toBe(true)
    expect(result.current.error).toBe(null)
    expect(result.current.reconnectAttempts).toBe(0)
  })

  it('connects successfully', async () => {
    const onConnect = vi.fn()
    const { result } = renderHook(() => 
      useWebSocket('/test', { onConnect })
    )

    // Fast-forward timers to simulate connection
    act(() => {
      vi.advanceTimersByTime(20)
    })

    expect(result.current.isConnected).toBe(true)
    expect(result.current.isConnecting).toBe(false)
    expect(onConnect).toHaveBeenCalled()
  })

  it('handles connection with auth token', () => {
    renderHook(() => useWebSocket('/test'))

    // Check that WebSocket was created with token
    expect(MockWebSocket).toHaveBeenCalledWith(
      'ws://localhost:8000/test?token=test-token'
    )
  })

  it('handles connection without auth token', () => {
    mockAuthStore.token = null
    vi.mocked(useAuthStore).mockReturnValue(mockAuthStore)

    renderHook(() => useWebSocket('/test'))

    expect(MockWebSocket).toHaveBeenCalledWith('ws://localhost:8000/test')
  })

  it('handles incoming messages', async () => {
    const onMessage = vi.fn()
    renderHook(() => 
      useWebSocket('/test', { onMessage })
    )

    act(() => {
      vi.advanceTimersByTime(20)
    })

    // Simulate receiving a message
    const mockMessage = { type: 'test', data: 'hello' }
    const mockWs = (global.WebSocket as any).mock.instances[0]
    
    act(() => {
      mockWs.simulateMessage(mockMessage)
    })

    expect(onMessage).toHaveBeenCalledWith(mockMessage)
  })

  it('handles malformed messages gracefully', async () => {
    const onMessage = vi.fn()
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    
    renderHook(() => 
      useWebSocket('/test', { onMessage })
    )

    act(() => {
      vi.advanceTimersByTime(20)
    })

    const mockWs = (global.WebSocket as any).mock.instances[0]
    
    act(() => {
      // Simulate malformed JSON
      if (mockWs.onmessage) {
        mockWs.onmessage(new MessageEvent('message', { data: 'invalid json' }))
      }
    })

    expect(onMessage).not.toHaveBeenCalled()
    expect(consoleSpy).toHaveBeenCalledWith(
      'Failed to parse WebSocket message:',
      expect.any(Error)
    )

    consoleSpy.mockRestore()
  })

  it('sends messages when connected', async () => {
    const { result } = renderHook(() => useWebSocket('/test'))

    act(() => {
      vi.advanceTimersByTime(20)
    })

    const mockWs = (global.WebSocket as any).mock.instances[0]
    const sendSpy = vi.spyOn(mockWs, 'send')

    const message = { type: 'test', data: 'hello' }
    
    act(() => {
      const success = result.current.sendMessage(message)
      expect(success).toBe(true)
    })

    expect(sendSpy).toHaveBeenCalledWith(JSON.stringify(message))
  })

  it('fails to send messages when not connected', () => {
    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const { result } = renderHook(() => useWebSocket('/test'))

    const message = { type: 'test', data: 'hello' }
    
    act(() => {
      const success = result.current.sendMessage(message)
      expect(success).toBe(false)
    })

    expect(consoleSpy).toHaveBeenCalledWith('WebSocket is not connected')
    consoleSpy.mockRestore()
  })

  it('handles disconnection', async () => {
    const onDisconnect = vi.fn()
    const { result } = renderHook(() => 
      useWebSocket('/test', { onDisconnect })
    )

    act(() => {
      vi.advanceTimersByTime(20)
    })

    expect(result.current.isConnected).toBe(true)

    const mockWs = (global.WebSocket as any).mock.instances[0]
    
    act(() => {
      mockWs.simulateClose()
    })

    expect(result.current.isConnected).toBe(false)
    expect(onDisconnect).toHaveBeenCalled()
  })

  it('attempts reconnection after disconnection', async () => {
    const { result } = renderHook(() => 
      useWebSocket('/test', { reconnectInterval: 1000, maxReconnectAttempts: 3 })
    )

    act(() => {
      vi.advanceTimersByTime(20)
    })

    const mockWs = (global.WebSocket as any).mock.instances[0]
    
    act(() => {
      mockWs.simulateClose()
    })

    expect(result.current.reconnectAttempts).toBe(1)

    // Fast-forward to trigger reconnection
    act(() => {
      vi.advanceTimersByTime(1000)
    })

    // Should create a new WebSocket instance
    expect(MockWebSocket).toHaveBeenCalledTimes(2)
  })

  it('stops reconnecting after max attempts', async () => {
    const { result } = renderHook(() => 
      useWebSocket('/test', { reconnectInterval: 1000, maxReconnectAttempts: 2 })
    )

    act(() => {
      vi.advanceTimersByTime(20)
    })

    // Simulate multiple disconnections
    for (let i = 0; i < 3; i++) {
      const mockWs = (global.WebSocket as any).mock.instances[i]
      
      act(() => {
        mockWs.simulateClose()
        vi.advanceTimersByTime(1000)
      })
    }

    expect(result.current.error).toBe('Max reconnection attempts reached')
    expect(result.current.reconnectAttempts).toBe(2)
  })

  it('handles manual disconnect', async () => {
    const { result } = renderHook(() => useWebSocket('/test'))

    act(() => {
      vi.advanceTimersByTime(20)
    })

    expect(result.current.isConnected).toBe(true)

    act(() => {
      result.current.disconnect()
    })

    expect(result.current.isConnected).toBe(false)
    expect(result.current.reconnectAttempts).toBe(0)
  })

  it('handles manual reconnect', async () => {
    const { result } = renderHook(() => useWebSocket('/test'))

    act(() => {
      vi.advanceTimersByTime(20)
    })

    act(() => {
      result.current.disconnect()
    })

    expect(result.current.isConnected).toBe(false)

    act(() => {
      result.current.reconnect()
      vi.advanceTimersByTime(120) // Wait for disconnect + reconnect
    })

    expect(result.current.isConnected).toBe(true)
  })

  it('handles WebSocket errors', async () => {
    const onError = vi.fn()
    const { result } = renderHook(() => 
      useWebSocket('/test', { onError })
    )

    act(() => {
      vi.advanceTimersByTime(20)
    })

    const mockWs = (global.WebSocket as any).mock.instances[0]
    
    act(() => {
      mockWs.simulateError()
    })

    expect(result.current.error).toBe('WebSocket connection error')
    expect(result.current.isConnecting).toBe(false)
    expect(onError).toHaveBeenCalled()
  })

  it('cleans up on unmount', async () => {
    const { unmount } = renderHook(() => useWebSocket('/test'))

    act(() => {
      vi.advanceTimersByTime(20)
    })

    const mockWs = (global.WebSocket as any).mock.instances[0]
    const closeSpy = vi.spyOn(mockWs, 'close')

    unmount()

    expect(closeSpy).toHaveBeenCalled()
  })

  it('uses custom WebSocket URL from environment', () => {
    vi.stubEnv('VITE_WS_URL', 'ws://custom-url:9000')

    renderHook(() => useWebSocket('/test'))

    expect(MockWebSocket).toHaveBeenCalledWith(
      'ws://custom-url:9000/test?token=test-token'
    )

    vi.unstubAllEnvs()
  })

  it('does not reconnect when manually closed', async () => {
    const { result } = renderHook(() => 
      useWebSocket('/test', { maxReconnectAttempts: 5 })
    )

    act(() => {
      vi.advanceTimersByTime(20)
    })

    // Manually disconnect
    act(() => {
      result.current.disconnect()
    })

    // Fast-forward past reconnect interval
    act(() => {
      vi.advanceTimersByTime(5000)
    })

    // Should not have attempted reconnection
    expect(MockWebSocket).toHaveBeenCalledTimes(1)
    expect(result.current.reconnectAttempts).toBe(0)
  })
})