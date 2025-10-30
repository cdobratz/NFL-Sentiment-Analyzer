/* 
 Test framework: Vitest + jsdom
 Project uses: vitest (^0.34.x), jsdom, @testing-library/jest-dom (configured in src/test/setup.ts)
*/
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  registerServiceWorker,
  unregisterServiceWorker,
  initializePWAPrompt,
  showInstallPrompt,
  isPWAInstalled,
  canInstallPWA,
  isMobileDevice,
  isTouchDevice,
  getDeviceOrientation,
  optimizeViewport,
  preventZoomOnInputFocus,
} from '../pwa'

describe('PWA Utilities (Vitest)', () => {
  let logSpy: ReturnType<typeof vi.spyOn>
  let errorSpy: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    vi.clearAllMocks()
    logSpy = vi.spyOn(console, 'log').mockImplementation(() => {})
    errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('registerServiceWorker', () => {
    it('registers when supported and logs success', async () => {
      const mockRegistration = {
        addEventListener: vi.fn(),
        installing: null as unknown,
      }
      const mockRegister = vi.fn().mockResolvedValue(mockRegistration)
      Object.defineProperty(navigator, 'serviceWorker', {
        value: { register: mockRegister },
        configurable: true,
        writable: true,
      })

      await registerServiceWorker()

      expect(mockRegister).toHaveBeenCalledWith('/sw.js')
      expect(logSpy).toHaveBeenCalledWith(
        'Service Worker registered successfully:',
        mockRegistration
      )
    })

    it('does nothing if service worker unsupported', async () => {
      Object.defineProperty(navigator, 'serviceWorker', {
        value: undefined,
        configurable: true,
        writable: true,
      })

      await registerServiceWorker()
      expect(logSpy).not.toHaveBeenCalledWith(expect.stringContaining('registered successfully'))
    })

    it('logs error on registration failure', async () => {
      const err = new Error('Registration failed')
      const mockRegister = vi.fn().mockRejectedValue(err)
      Object.defineProperty(navigator, 'serviceWorker', {
        value: { register: mockRegister },
        configurable: true,
        writable: true,
      })

      await registerServiceWorker()
      expect(errorSpy).toHaveBeenCalledWith('Service Worker registration failed:', err)
    })

    it('wires updatefound listener', async () => {
      const addListener = vi.fn()
      const mockRegistration = { addEventListener: addListener, installing: null as unknown }
      const mockRegister = vi.fn().mockResolvedValue(mockRegistration)
      Object.defineProperty(navigator, 'serviceWorker', {
        value: { register: mockRegister },
        configurable: true,
        writable: true,
      })

      await registerServiceWorker()
      expect(addListener).toHaveBeenCalledWith('updatefound', expect.any(Function))
    })

    it('logs when new content available after install and controller exists', async () => {
      let updateFoundCb: unknown
      let stateChangeCb: unknown
      const newWorker = {
        state: 'installed',
        addEventListener: vi.fn((ev, cb) => {
          if (ev === 'statechange') stateChangeCb = cb
        }),
      }
      const mockRegistration = {
        installing: newWorker as unknown,
        addEventListener: vi.fn((ev, cb) => {
          if (ev === 'updatefound') updateFoundCb = cb
        }),
      }
      const mockRegister = vi.fn().mockResolvedValue(mockRegistration)
      Object.defineProperty(navigator, 'serviceWorker', {
        value: { register: mockRegister, controller: {} },
        configurable: true,
        writable: true,
      })

      await registerServiceWorker()
      ;(updateFoundCb as () => void)()
      ;(stateChangeCb as () => void)()
      expect(logSpy).toHaveBeenCalledWith('New content available, please refresh')
    })

    it('does not log new content when no controller', async () => {
      let updateFoundCb: unknown
      let stateChangeCb: unknown
      const newWorker = {
        state: 'installed',
        addEventListener: vi.fn((ev, cb) => {
          if (ev === 'statechange') stateChangeCb = cb
        }),
      }
      const mockRegistration = {
        installing: newWorker as unknown,
        addEventListener: vi.fn((ev, cb) => {
          if (ev === 'updatefound') updateFoundCb = cb
        }),
      }
      const mockRegister = vi.fn().mockResolvedValue(mockRegistration)
      Object.defineProperty(navigator, 'serviceWorker', {
        value: { register: mockRegister, controller: null },
        configurable: true,
        writable: true,
      })

      await registerServiceWorker()
      logSpy.mockClear()
      ;(updateFoundCb as () => void)()
      ;(stateChangeCb as () => void)()
      expect(logSpy).not.toHaveBeenCalledWith('New content available, please refresh')
    })

    it('handles null installing worker', async () => {
      let updateFoundCb: unknown
      const mockRegistration = {
        installing: null as unknown,
        addEventListener: vi.fn((ev, cb) => {
          if (ev === 'updatefound') updateFoundCb = cb
        }),
      }
      const mockRegister = vi.fn().mockResolvedValue(mockRegistration)
      Object.defineProperty(navigator, 'serviceWorker', {
        value: { register: mockRegister },
        configurable: true,
        writable: true,
      })
      await registerServiceWorker()
      expect(() => (updateFoundCb as () => void)()).not.toThrow()
    })
  })

  describe('unregisterServiceWorker', () => {
    it('unregisters all registrations and logs', async () => {
      const u1 = vi.fn().mockResolvedValue(true)
      const u2 = vi.fn().mockResolvedValue(true)
      const getRegs = vi.fn().mockResolvedValue([{ unregister: u1 }, { unregister: u2 }])
      Object.defineProperty(navigator, 'serviceWorker', {
        value: { getRegistrations: getRegs },
        configurable: true,
        writable: true,
      })
      await unregisterServiceWorker()
      expect(getRegs).toHaveBeenCalled()
      expect(u1).toHaveBeenCalled()
      expect(u2).toHaveBeenCalled()
      expect(logSpy).toHaveBeenCalledWith('Service Worker unregistered')
    })

    it('handles empty registrations array', async () => {
      const getRegs = vi.fn().mockResolvedValue([])
      Object.defineProperty(navigator, 'serviceWorker', {
        value: { getRegistrations: getRegs },
        configurable: true,
        writable: true,
      })
      await unregisterServiceWorker()
      expect(logSpy).toHaveBeenCalledWith('Service Worker unregistered')
    })

    it('no-op if unsupported', async () => {
      Object.defineProperty(navigator, 'serviceWorker', {
        value: undefined,
        configurable: true,
        writable: true,
      })
      await unregisterServiceWorker()
      expect(logSpy).not.toHaveBeenCalledWith('Service Worker unregistered')
    })

    it('logs error if getRegistrations fails', async () => {
      const err = new Error('getRegistrations failed')
      const getRegs = vi.fn().mockRejectedValue(err)
      Object.defineProperty(navigator, 'serviceWorker', {
        value: { getRegistrations: getRegs },
        configurable: true,
        writable: true,
      })
      await unregisterServiceWorker()
      expect(errorSpy).toHaveBeenCalledWith('Service Worker unregistration failed:', err)
    })

    it('logs error when a registration.unregister rejects (and stops further unregisters)', async () => {
      const u1 = vi.fn().mockRejectedValue(new Error('fail one'))
      const u2 = vi.fn().mockResolvedValue(true)
      const getRegs = vi.fn().mockResolvedValue([{ unregister: u1 }, { unregister: u2 }])
      Object.defineProperty(navigator, 'serviceWorker', {
        value: { getRegistrations: getRegs },
        configurable: true,
        writable: true,
      })
      await unregisterServiceWorker()
      expect(u1).toHaveBeenCalled()
      expect(errorSpy).toHaveBeenCalled()
      expect(logSpy).not.toHaveBeenCalledWith('Service Worker unregistered')
    })
  })

  describe('initializePWAPrompt', () => {
    it('registers beforeinstallprompt and appinstalled listeners', () => {
      const spy = vi.spyOn(window, 'addEventListener')
      initializePWAPrompt()
      expect(spy).toHaveBeenCalledWith('beforeinstallprompt', expect.any(Function))
      expect(spy).toHaveBeenCalledWith('appinstalled', expect.any(Function))
    })

    it('prevents default and logs when beforeinstallprompt fires', () => {
      const spy = vi.spyOn(window, 'addEventListener')
      let beforeCb: unknown
      spy.mockImplementation((evt: unknown, cb: unknown) => {
        if (evt === 'beforeinstallprompt') beforeCb = cb
      })
      initializePWAPrompt()
      const ev = { preventDefault: vi.fn() }
      ;(beforeCb as (ev: any) => void)(ev)
      expect(ev.preventDefault).toHaveBeenCalled()
      expect(logSpy).toHaveBeenCalledWith('PWA install prompt available')
    })

    it('logs on appinstalled', () => {
      const spy = vi.spyOn(window, 'addEventListener')
      let appCb: unknown
      spy.mockImplementation((evt: unknown, cb: unknown) => {
        if (evt === 'appinstalled') appCb = cb
      })
      initializePWAPrompt()
      ;(appCb as () => void)()
      expect(logSpy).toHaveBeenCalledWith('PWA was installed')
    })
  })

  describe('showInstallPrompt', () => {
    it('returns false when no deferred prompt', async () => {
      const res = await showInstallPrompt()
      expect(res).toBe(false)
    })

    it('prompts and returns true on accept', async () => {
      let beforeCb: unknown
      vi.spyOn(window, 'addEventListener').mockImplementation((evt: unknown, cb: unknown) => {
        if (evt === 'beforeinstallprompt') beforeCb = cb
      })
      initializePWAPrompt()
      const prompt = vi.fn()
      const event = { preventDefault: vi.fn(), prompt, userChoice: Promise.resolve({ outcome: 'accepted' }) }
      ;(beforeCb as (event: any) => void)(event)

      const res = await showInstallPrompt()
      expect(prompt).toHaveBeenCalled()
      expect(res).toBe(true)
      expect(logSpy).toHaveBeenCalledWith('User response to the install prompt: accepted')
    })

    it('prompts and returns false on dismissed', async () => {
      let beforeCb: unknown
      vi.spyOn(window, 'addEventListener').mockImplementation((evt: unknown, cb: unknown) => {
        if (evt === 'beforeinstallprompt') beforeCb = cb
      })
      initializePWAPrompt()
      const prompt = vi.fn()
      const event = { preventDefault: vi.fn(), prompt, userChoice: Promise.resolve({ outcome: 'dismissed' }) }
      ;(beforeCb as (event: any) => void)(event)

      const res = await showInstallPrompt()
      expect(prompt).toHaveBeenCalled()
      expect(res).toBe(false)
      expect(logSpy).toHaveBeenCalledWith('User response to the install prompt: dismissed')
    })

    it('logs error and returns false when userChoice rejects', async () => {
      let beforeCb: unknown
      vi.spyOn(window, 'addEventListener').mockImplementation((evt: unknown, cb: unknown) => {
        if (evt === 'beforeinstallprompt') beforeCb = cb
      })
      initializePWAPrompt()
      const err = new Error('choice error')
      const event = { preventDefault: vi.fn(), prompt: vi.fn(), userChoice: Promise.reject(err) }
      ;(beforeCb as (event: any) => void)(event)

      const res = await showInstallPrompt()
      expect(res).toBe(false)
      expect(errorSpy).toHaveBeenCalledWith('Error showing install prompt:', err)
    })

    it('clears deferred prompt after showing', async () => {
      let beforeCb: unknown
      vi.spyOn(window, 'addEventListener').mockImplementation((evt: unknown, cb: unknown) => {
        if (evt === 'beforeinstallprompt') beforeCb = cb
      })
      initializePWAPrompt()
      const event = { preventDefault: vi.fn(), prompt: vi.fn(), userChoice: Promise.resolve({ outcome: 'accepted' }) }
      ;(beforeCb as (event: any) => void)(event)
      await showInstallPrompt()
      const res2 = await showInstallPrompt()
      expect(res2).toBe(false)
    })
  })

  describe('isPWAInstalled', () => {
    it('true when display-mode standalone matches', () => {
      Object.defineProperty(window, 'matchMedia', {
        value: vi.fn().mockReturnValue({ matches: true }),
        configurable: true,
        writable: true,
      })
      expect(isPWAInstalled()).toBe(true)
    })

    it('true when navigator.standalone is true', () => {
      Object.defineProperty(window, 'matchMedia', {
        value: vi.fn().mockReturnValue({ matches: false }),
        configurable: true,
        writable: true,
      })
      Object.defineProperty(window.navigator as unknown, 'standalone', {
        value: true,
        configurable: true,
        writable: true,
      })
      expect(isPWAInstalled()).toBe(true)
    })

    it('false otherwise', () => {
      Object.defineProperty(window, 'matchMedia', {
        value: vi.fn().mockReturnValue({ matches: false }),
        configurable: true,
        writable: true,
      })
      Object.defineProperty(window.navigator as unknown, 'standalone', {
        value: false,
        configurable: true,
        writable: true,
      })
      expect(isPWAInstalled()).toBe(false)
    })
  })

  describe('canInstallPWA', () => {
    it('false initially, true after beforeinstallprompt, false after appinstalled', () => {
      let beforeCb: unknown, appCb: unknown
      vi.spyOn(window, 'addEventListener').mockImplementation((evt: unknown, cb: unknown) => {
        if (evt === 'beforeinstallprompt') beforeCb = cb
        if (evt === 'appinstalled') appCb = cb
      })
      expect(canInstallPWA()).toBe(false)
      initializePWAPrompt()
      ;(beforeCb as (ev: any) => void)({ preventDefault: vi.fn() })
      expect(canInstallPWA()).toBe(true)
      ;(appCb as () => void)()
      expect(canInstallPWA()).toBe(false)
    })
  })

  describe('isMobileDevice', () => {
    const mobileUAs = [
      'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)',
      'Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X)',
      'Mozilla/5.0 (Linux; Android 10)',
      'Opera/9.80 (J2ME/MIDP; Opera Mini/9.80)',
    ]
    const desktopUAs = [
      'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
    ]

    mobileUAs.forEach((ua) => {
      it(`returns true for mobile UA: ${ua.slice(0, 30)}...`, () => {
        Object.defineProperty(navigator, 'userAgent', { value: ua, configurable: true, writable: true })
        expect(isMobileDevice()).toBe(true)
      })
    })
    desktopUAs.forEach((ua) => {
      it(`returns false for desktop UA: ${ua.slice(0, 30)}...`, () => {
        Object.defineProperty(navigator, 'userAgent', { value: ua, configurable: true, writable: true })
        expect(isMobileDevice()).toBe(false)
      })
    })
  })

  describe('isTouchDevice', () => {
    it('true when ontouchstart exists', () => {
      Object.defineProperty(window as unknown, 'ontouchstart', { value: {}, configurable: true, writable: true })
      Object.defineProperty(navigator as unknown, 'maxTouchPoints', { value: 0, configurable: true, writable: true })
      expect(isTouchDevice()).toBe(true)
    })
    it('true when maxTouchPoints > 0', () => {
      delete (window as any).ontouchstart
      Object.defineProperty(navigator as unknown, 'maxTouchPoints', { value: 3, configurable: true, writable: true })
      expect(isTouchDevice()).toBe(true)
    })
    it('false when no touch support', () => {
      delete (window as any).ontouchstart
      Object.defineProperty(navigator as unknown, 'maxTouchPoints', { value: 0, configurable: true, writable: true })
      expect(isTouchDevice()).toBe(false)
    })
  })

  describe('getDeviceOrientation', () => {
    it('portrait when height > width', () => {
      Object.defineProperty(window, 'innerHeight', { value: 800, configurable: true, writable: true })
      Object.defineProperty(window, 'innerWidth', { value: 600, configurable: true, writable: true })
      expect(getDeviceOrientation()).toBe('portrait')
    })
    it('landscape when width >= height', () => {
      Object.defineProperty(window, 'innerHeight', { value: 600, configurable: true, writable: true })
      Object.defineProperty(window, 'innerWidth', { value: 800, configurable: true, writable: true })
      expect(getDeviceOrientation()).toBe('landscape')
    })
  })

  describe('optimizeViewport', () => {
    it('sets viewport content when meta exists', () => {
      const setAttribute = vi.fn()
      vi.spyOn(document, 'querySelector').mockReturnValue({ setAttribute } as unknown as Element)
      optimizeViewport()
      expect(document.querySelector).toHaveBeenCalledWith('meta[name="viewport"]')
      expect(setAttribute).toHaveBeenCalledWith(
        'content',
        'width=device-width, initial-scale=1.0, user-scalable=no, viewport-fit=cover'
      )
    })
    it('no-ops when meta not found', () => {
      vi.spyOn(document, 'querySelector').mockReturnValue(null)
      optimizeViewport()
      expect(document.querySelector).toHaveBeenCalledWith('meta[name="viewport"]')
    })
    it('throws if meta element lacks setAttribute', () => {
      vi.spyOn(document, 'querySelector').mockReturnValue({} as Element)
      expect(() => optimizeViewport()).toThrow()
    })
  })

  describe('preventZoomOnInputFocus', () => {
    beforeEach(() => {
      Object.defineProperty(navigator, 'userAgent', {
        value: 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)',
        configurable: true,
        writable: true,
      })
    })

    it('adds focus/blur listeners on mobile', () => {
      const el1 = { addEventListener: vi.fn() }
      const el2 = { addEventListener: vi.fn() }
      const el3 = { addEventListener: vi.fn() }
      vi.spyOn(document, 'querySelectorAll').mockReturnValue([el1, el2, el3] as unknown as NodeListOf<Element>)
      preventZoomOnInputFocus()
      expect(document.querySelectorAll).toHaveBeenCalledWith('input, select, textarea')
      ;[el1, el2, el3].forEach((el: any) => {
        expect(el.addEventListener).toHaveBeenCalledWith('focus', expect.any(Function))
        expect(el.addEventListener).toHaveBeenCalledWith('blur', expect.any(Function))
      })
    })

    it('does not add listeners on non-mobile', () => {
      Object.defineProperty(navigator, 'userAgent', {
        value: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        configurable: true,
        writable: true,
      })
      const qsAll = vi.spyOn(document, 'querySelectorAll')
      preventZoomOnInputFocus()
      expect(qsAll).not.toHaveBeenCalled()
    })

    it('sets maximum-scale on focus', () => {
      let focusCb: unknown
      const el = { addEventListener: vi.fn((ev, cb) => ev === 'focus' && (focusCb = cb)) }
      vi.spyOn(document, 'querySelectorAll').mockReturnValue([el] as unknown as NodeListOf<Element>)
      const setAttribute = vi.fn()
      vi.spyOn(document, 'querySelector').mockReturnValue({ setAttribute } as unknown as Element)
      preventZoomOnInputFocus()
      ;(focusCb as () => void)()
      expect(document.querySelector).toHaveBeenCalledWith('meta[name="viewport"]')
      expect(setAttribute).toHaveBeenCalledWith(
        'content',
        'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no'
      )
    })

    it('removes maximum-scale on blur', () => {
      let blurCb: unknown
      const el = { addEventListener: vi.fn((ev, cb) => ev === 'blur' && (blurCb = cb)) }
      vi.spyOn(document, 'querySelectorAll').mockReturnValue([el] as unknown as NodeListOf<Element>)
      const setAttribute = vi.fn()
      vi.spyOn(document, 'querySelector').mockReturnValue({ setAttribute } as unknown as Element)
      preventZoomOnInputFocus()
      ;(blurCb as () => void)()
      expect(setAttribute).toHaveBeenCalledWith(
        'content',
        'width=device-width, initial-scale=1.0, user-scalable=no'
      )
    })

    it('gracefully handles missing viewport meta', () => {
      let focusCb: unknown
      const el = { addEventListener: vi.fn((ev, cb) => ev === 'focus' && (focusCb = cb)) }
      vi.spyOn(document, 'querySelectorAll').mockReturnValue([el] as unknown as NodeListOf<Element>)
      vi.spyOn(document, 'querySelector').mockReturnValue(null)
      preventZoomOnInputFocus()
      expect(() => (focusCb as () => void)()).not.toThrow()
    })

    it('handles empty input list', () => {
      vi.spyOn(document, 'querySelectorAll').mockReturnValue([] as unknown as NodeListOf<Element>)
      expect(() => preventZoomOnInputFocus()).not.toThrow()
    })
  })

  describe('Integration smoke', () => {
    it('PWA install flow end-to-end', async () => {
      let beforeCb: unknown
      vi.spyOn(window, 'addEventListener').mockImplementation((evt: unknown, cb: unknown) => {
        if (evt === 'beforeinstallprompt') beforeCb = cb
      })
      initializePWAPrompt()
      expect(canInstallPWA()).toBe(false)
      const ev = { preventDefault: vi.fn(), prompt: vi.fn(), userChoice: Promise.resolve({ outcome: 'accepted' }) }
      ;(beforeCb as (ev: any) => void)(ev)
      expect(canInstallPWA()).toBe(true)
      const res = await showInstallPrompt()
      expect(res).toBe(true)
      expect(ev.prompt).toHaveBeenCalled()
      expect(canInstallPWA()).toBe(false)
    })

    it('Service Worker register + unregister', async () => {
      const mockRegistration = { addEventListener: vi.fn(), installing: null as unknown }
      const register = vi.fn().mockResolvedValue(mockRegistration)
      const unregister = vi.fn().mockResolvedValue(true)
      Object.defineProperty(navigator, 'serviceWorker', {
        value: { register, getRegistrations: vi.fn().mockResolvedValue([{ unregister }]) },
        configurable: true,
        writable: true,
      })
      await registerServiceWorker()
      expect(register).toHaveBeenCalled()
      expect(logSpy).toHaveBeenCalledWith('Service Worker registered successfully:', mockRegistration)
      await unregisterServiceWorker()
      expect(unregister).toHaveBeenCalled()
      expect(logSpy).toHaveBeenCalledWith('Service Worker unregistered')
    })
  })
})