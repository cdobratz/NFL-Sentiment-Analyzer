# Testing Infrastructure

This document describes the testing infrastructure for the NFL Sentiment Analyzer project.

## Overview

The project uses a comprehensive testing setup with:
- **Frontend**: Vitest + React Testing Library
- **Backend**: pytest + FastAPI TestClient
- **Coverage**: Integrated coverage reporting for both frontend and backend
- **CI/CD**: Automated testing in GitHub Actions

## Directory Structure

```
tests/
├── README.md                 # This file
├── __init__.py              # Python package marker
├── conftest.py              # Pytest configuration and fixtures
├── test_config.py           # Test configuration and utilities
├── test_main.py             # Basic functionality tests
└── ...                      # Additional test files

frontend/src/
├── test/
│   ├── setup.ts             # Vitest setup and global mocks
│   └── utils.tsx            # Testing utilities and custom render
└── components/__tests__/
    └── LoadingSpinner.test.tsx  # Example component test
```

## Backend Testing

### Setup

The backend uses pytest with the following key dependencies:
- `pytest`: Test framework
- `pytest-asyncio`: Async test support
- `pytest-cov`: Coverage reporting
- `httpx`: HTTP client for API testing
- `motor`: MongoDB async driver (for integration tests)
- `redis`: Redis client (for integration tests)

### Running Backend Tests

```bash
# Run all backend tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html --cov-report=term-missing

# Run specific test file
pytest tests/test_main.py -v

# Run tests with specific markers
pytest -m unit
pytest -m integration
```

### Test Configuration

The `pytest.ini` file configures:
- Test discovery patterns
- Coverage settings (80% minimum)
- Async test mode
- Test markers for categorization

### Fixtures

The `conftest.py` file provides:
- Mock database and Redis clients
- Test data factories
- External API mocks
- Authentication helpers

### Test Categories

Tests are organized using pytest markers:
- `@pytest.mark.unit`: Unit tests (fast, isolated)
- `@pytest.mark.integration`: Integration tests (require services)
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.external`: Tests requiring external services

## Frontend Testing

### Setup

The frontend uses Vitest with React Testing Library:
- `vitest`: Fast test runner
- `@testing-library/react`: React component testing utilities
- `@testing-library/jest-dom`: Additional matchers
- `jsdom`: DOM environment for tests

### Running Frontend Tests

```bash
cd frontend

# Run all tests
npm run test

# Run with coverage
npm run test:coverage

# Run in watch mode (development)
npm run test:watch
```

### Test Utilities

The `frontend/src/test/utils.tsx` provides:
- Custom render function with providers
- Mock data generators
- Common test utilities

### Global Mocks

The `frontend/src/test/setup.ts` includes:
- IntersectionObserver mock
- ResizeObserver mock
- matchMedia mock
- WebSocket mock
- fetch mock

## Coverage Requirements

### Backend Coverage
- **Target**: 85% for core business logic
- **Minimum**: 80% overall
- **Reports**: HTML, XML, and terminal output

### Frontend Coverage
- **Target**: 80% for critical components
- **Thresholds**: 80% for branches, functions, lines, and statements
- **Reports**: HTML, LCOV, JSON, and text

## CI/CD Integration

### GitHub Actions

The `.github/workflows/ci.yml` includes:
- Separate jobs for backend and frontend testing
- Service containers (MongoDB, Redis) for integration tests
- Coverage reporting to Codecov
- Linting and formatting checks
- Docker image testing

### Test Services

For integration tests, the CI provides:
- MongoDB 7.0 container
- Redis 7.2 container
- Health checks for service readiness

## Local Development

### Prerequisites

1. **Backend**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Frontend**:
   ```bash
   cd frontend && npm install
   ```

3. **Services** (for integration tests):
   ```bash
   # Using Docker
   docker run -d -p 27017:27017 mongo:7.0
   docker run -d -p 6379:6379 redis:7.2-alpine
   ```

### Test Scripts

Use the provided scripts for easy testing:

```bash
# Run all tests
make test

# Run backend tests only
make test-backend

# Run frontend tests only
make test-frontend

# Run with coverage
make test-coverage

# Use the comprehensive test script
./scripts/run-tests.sh --help
```

### Test Script Options

The `scripts/run-tests.sh` script supports:
- `--backend-only`: Run only backend tests
- `--frontend-only`: Run only frontend tests
- `--coverage`: Include coverage reporting
- `--integration`: Run only integration tests
- `--unit`: Run only unit tests

## Writing Tests

### Backend Test Example

```python
import pytest
from unittest.mock import MagicMock

class TestSentimentAnalysis:
    def test_analyze_positive_sentiment(self):
        # Arrange
        text = "This team is amazing!"
        
        # Act
        result = analyze_sentiment(text)
        
        # Assert
        assert result.sentiment == "POSITIVE"
        assert result.confidence > 0.7

    @pytest.mark.asyncio
    async def test_async_analysis(self):
        # Test async functionality
        result = await async_analyze_sentiment("Great game!")
        assert result is not None
```

### Frontend Test Example

```tsx
import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '../test/utils'
import MyComponent from '../MyComponent'

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />)
    expect(screen.getByText('Hello World')).toBeInTheDocument()
  })

  it('handles user interaction', async () => {
    render(<MyComponent />)
    const button = screen.getByRole('button')
    
    fireEvent.click(button)
    
    expect(screen.getByText('Clicked!')).toBeInTheDocument()
  })
})
```

## Best Practices

### General
- Write tests before or alongside implementation (TDD)
- Keep tests simple and focused
- Use descriptive test names
- Follow the AAA pattern (Arrange, Act, Assert)

### Backend
- Mock external dependencies
- Use fixtures for common test data
- Test both success and error cases
- Separate unit and integration tests

### Frontend
- Test user interactions, not implementation details
- Use semantic queries (getByRole, getByText)
- Mock external API calls
- Test accessibility features

### Performance
- Keep unit tests fast (< 100ms each)
- Use mocks to avoid external dependencies
- Run integration tests separately
- Parallelize test execution when possible

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Service Connection**: Check if MongoDB/Redis are running for integration tests
3. **Coverage Failures**: Review coverage thresholds in configuration
4. **Async Test Issues**: Ensure proper async/await usage and pytest-asyncio setup

### Debug Commands

```bash
# Verbose test output
pytest -v -s

# Debug specific test
pytest tests/test_main.py::TestClass::test_method -v -s

# Show test coverage details
pytest --cov=app --cov-report=html
open htmlcov/index.html
```

## Future Enhancements

- [ ] Add visual regression testing
- [ ] Implement load testing
- [ ] Add mutation testing
- [ ] Enhance test data factories
- [ ] Add contract testing for APIs
- [ ] Implement snapshot testing for components