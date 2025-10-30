declare module 'jest-axe' {
  export interface AxeResults {
    violations: unknown[]
  }

  export function axe(container: Element, options?: unknown): Promise<AxeResults>
  export function toHaveNoViolations(received: AxeResults): {
    message(): string
    pass: boolean
  }
}

declare global {
  namespace Vi {
    interface AsymmetricMatchersContaining {
      toHaveNoViolations(): unknown
    }
  }
}