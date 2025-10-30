declare module 'jest-axe' {
  export interface AxeResults {
    violations: any[]
  }

  export function axe(container: Element, options?: any): Promise<AxeResults>
  export function toHaveNoViolations(received: AxeResults): {
    message(): string
    pass: boolean
  }
}

declare global {
  namespace Vi {
    interface AsymmetricMatchersContaining {
      toHaveNoViolations(): any
    }
  }
}