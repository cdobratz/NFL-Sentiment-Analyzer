// PWA utilities for service worker registration and app installation

export const registerServiceWorker = async (): Promise<void> => {
  if ('serviceWorker' in navigator) {
    try {
      const registration = await navigator.serviceWorker.register('/sw.js')
      console.log('Service Worker registered successfully:', registration)
      
      // Check for updates
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing
        if (newWorker) {
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              // New content is available, notify user
              console.log('New content available, please refresh')
              // You could show a toast notification here
            }
          })
        }
      })
    } catch (error) {
      console.error('Service Worker registration failed:', error)
    }
  }
}

export const unregisterServiceWorker = async (): Promise<void> => {
  if ('serviceWorker' in navigator) {
    try {
      const registrations = await navigator.serviceWorker.getRegistrations()
      for (const registration of registrations) {
        await registration.unregister()
      }
      console.log('Service Worker unregistered')
    } catch (error) {
      console.error('Service Worker unregistration failed:', error)
    }
  }
}

// PWA installation prompt
let deferredPrompt: any = null

export const initializePWAPrompt = (): void => {
  window.addEventListener('beforeinstallprompt', (e) => {
    // Prevent the mini-infobar from appearing on mobile
    e.preventDefault()
    // Stash the event so it can be triggered later
    deferredPrompt = e
    console.log('PWA install prompt available')
  })

  window.addEventListener('appinstalled', () => {
    console.log('PWA was installed')
    deferredPrompt = null
  })
}

export const showInstallPrompt = async (): Promise<boolean> => {
  if (!deferredPrompt) {
    return false
  }

  try {
    // Show the install prompt
    deferredPrompt.prompt()
    
    // Wait for the user to respond to the prompt
    const { outcome } = await deferredPrompt.userChoice
    
    console.log(`User response to the install prompt: ${outcome}`)
    
    // Clear the deferredPrompt
    deferredPrompt = null
    
    return outcome === 'accepted'
  } catch (error) {
    console.error('Error showing install prompt:', error)
    return false
  }
}

export const isPWAInstalled = (): boolean => {
  // Check if running in standalone mode (PWA)
  return window.matchMedia('(display-mode: standalone)').matches ||
         (window.navigator as any).standalone === true
}

export const canInstallPWA = (): boolean => {
  return deferredPrompt !== null
}

// Detect if device is mobile
export const isMobileDevice = (): boolean => {
  return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent)
}

// Detect if device supports touch
export const isTouchDevice = (): boolean => {
  return 'ontouchstart' in window || navigator.maxTouchPoints > 0
}

// Get device orientation
export const getDeviceOrientation = (): 'portrait' | 'landscape' => {
  return window.innerHeight > window.innerWidth ? 'portrait' : 'landscape'
}

// Add viewport meta tag for better mobile experience
export const optimizeViewport = (): void => {
  const viewport = document.querySelector('meta[name="viewport"]')
  if (viewport) {
    viewport.setAttribute('content', 'width=device-width, initial-scale=1.0, user-scalable=no, viewport-fit=cover')
  }
}

// Prevent zoom on input focus (iOS Safari)
export const preventZoomOnInputFocus = (): void => {
  if (isMobileDevice()) {
    const inputs = document.querySelectorAll('input, select, textarea')
    inputs.forEach(input => {
      input.addEventListener('focus', () => {
        const viewport = document.querySelector('meta[name="viewport"]')
        if (viewport) {
          viewport.setAttribute('content', 'width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no')
        }
      })
      
      input.addEventListener('blur', () => {
        const viewport = document.querySelector('meta[name="viewport"]')
        if (viewport) {
          viewport.setAttribute('content', 'width=device-width, initial-scale=1.0, user-scalable=no')
        }
      })
    })
  }
}