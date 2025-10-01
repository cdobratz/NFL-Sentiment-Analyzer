import { describe, it, expect } from 'vitest'
import { render } from '../../test/utils'
import LoadingSpinner from '../LoadingSpinner'

describe('LoadingSpinner', () => {
  it('renders loading spinner with default size', () => {
    const { container } = render(<LoadingSpinner />)
    
    const spinner = container.querySelector('.animate-spin')
    expect(spinner).toBeInTheDocument()
    expect(spinner).toHaveClass('w-6', 'h-6') // default md size
  })

  it('renders with small size when specified', () => {
    const { container } = render(<LoadingSpinner size="sm" />)
    
    const spinner = container.querySelector('.animate-spin')
    expect(spinner).toBeInTheDocument()
    expect(spinner).toHaveClass('w-4', 'h-4')
  })

  it('renders with large size when specified', () => {
    const { container } = render(<LoadingSpinner size="lg" />)
    
    const spinner = container.querySelector('.animate-spin')
    expect(spinner).toBeInTheDocument()
    expect(spinner).toHaveClass('w-8', 'h-8')
  })

  it('applies custom className', () => {
    const { container } = render(<LoadingSpinner className="custom-class" />)
    
    const wrapper = container.firstChild
    expect(wrapper).toHaveClass('custom-class')
  })
})