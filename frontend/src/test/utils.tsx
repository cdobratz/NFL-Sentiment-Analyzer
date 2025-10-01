import React, { ReactElement } from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

// Create a custom render function that includes providers
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  })

  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {children}
      </BrowserRouter>
    </QueryClientProvider>
  )
}

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
) => render(ui, { wrapper: AllTheProviders, ...options })

export * from '@testing-library/react'
export { customRender as render }

// Mock data generators
export const mockUser = {
  id: '1',
  email: 'test@example.com',
  username: 'testuser',
  role: 'USER' as const,
}

export const mockTeam = {
  id: '1',
  name: 'Test Team',
  abbreviation: 'TT',
  conference: 'NFC',
  division: 'North',
  current_sentiment: 0.75,
}

export const mockGame = {
  id: '1',
  home_team: 'Test Team 1',
  away_team: 'Test Team 2',
  game_date: '2024-01-15T20:00:00Z',
  week: 1,
  betting_lines: [],
}

export const mockSentimentData = {
  sentiment: 'POSITIVE' as const,
  confidence: 0.85,
  team_sentiment: 0.75,
  player_sentiment: 0.65,
}