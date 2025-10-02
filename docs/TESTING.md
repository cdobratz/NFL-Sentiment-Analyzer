# Testing Guide

This document outlines the testing setup and prerequisites for the NFL Sentiment Analyzer project.

## Prerequisites

### Backend Testing
- **Python 3.11+** with `uv` package manager (recommended) or `pip`
- **pytest** with configured markers for unit and integration tests
- **MongoDB** and **Redis** services for integration tests

### Frontend Testing  
- **Node.js 18+** with `npm`
- **Vitest** for running tests and coverage

## Test Configuration

### Pytest Markers
The project uses pytest markers to categorize tests:
- `unit`: Fast unit tests that don't require external services
- `integration`: Tests that require MongoDB, Redis, or other services
- `performance`: Performance benchmarking tests
- `slow`: Long-running tests

These markers are configured in both `pytest.ini` and `pyproject.toml`.

### Frontend Test Scripts
The frontend uses Vitest with the following npm scripts:
- `test`: Run tests once
- `test:watch`: Run tests in watch mode
- `test:coverage`: Run tests with coverage reporting

## Available Make Targets

### Test Commands
```bash
make test                # Run all tests (backend + frontend)
make test-backend        # Run backend tests with coverage
make test-frontend       # Run frontend tests
make test-unit           # Run unit tests only
make test-integration    # Run integration tests only
make test-coverage       # Run tests with coverage reports
```

### Code Quality Commands
```bash
make lint                # Run linting (flake8, mypy, eslint)
make format              # Format code (black, prettier)
```

### Development Commands
```bash
make dev-backend         # Start backend development server
make dev-frontend        # Start frontend development server
make dev                 # Start full development environment with Docker
```

## Validation Script

Run the validation script to check if all prerequisites are met:

```bash
./scripts/validate-test-setup.sh
```

This script will:
- Check Python, Node.js, and package manager installations
- Verify pytest markers are configured
- Validate frontend test scripts exist
- Install dependencies if needed
- Test all Makefile targets

## CI/CD Integration

The project includes GitHub Actions workflows that:
- Install all dependencies using `uv` for Python and `npm` for frontend
- Run the same test commands as the Makefile targets
- Ensure consistent behavior between local development and CI

### Workflow Files
- `.github/workflows/ci.yml`: Main CI pipeline with comprehensive testing
- `.github/workflows/ci-cd.yml`: Extended CI/CD with security scanning and deployment

## Troubleshooting

### Common Issues

1. **"pytest markers not found"**
   - Ensure `pytest.ini` or `pyproject.toml` contains marker definitions
   - Run `pytest --markers` to see available markers

2. **"Frontend test scripts missing"**
   - Check `frontend/package.json` has `test` and `test:coverage` scripts
   - Install frontend dependencies with `cd frontend && npm install`

3. **"MongoDB/Redis connection failed"**
   - Start services with `make db-setup` or use Docker Compose
   - Check connection strings in environment variables

4. **"uv command not found"**
   - Install uv: `pip install uv` or use system package manager
   - Alternative: Use pip with `pip install -e ".[dev]"`

### Environment Variables

For testing, ensure these environment variables are set:
```bash
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=test_nfl_sentiment
REDIS_URL=redis://localhost:6379/1
JWT_SECRET_KEY=test-secret-key-for-testing
TESTING=true
```

## Best Practices

1. **Run tests before committing**: Use `make test` to ensure all tests pass
2. **Use appropriate markers**: Mark tests as `unit` or `integration` appropriately
3. **Keep tests fast**: Unit tests should run quickly without external dependencies
4. **Use fixtures**: Leverage pytest fixtures for test data and setup
5. **Mock external services**: Use mocks for unit tests, real services for integration tests

## Coverage Reports

Coverage reports are generated in multiple formats:
- **Terminal**: Summary displayed after test runs
- **HTML**: Detailed reports in `htmlcov/` directory
- **XML**: For CI/CD integration in `coverage.xml`

View HTML coverage reports:
```bash
make test-coverage
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```