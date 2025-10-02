# Development Setup Guide

This project uses [uv](https://docs.astral.sh/uv/) for fast, reliable Python dependency management.

## Quick Start

### 1. Install uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

### 2. Setup the project

```bash
# Clone and setup
git clone <repository-url>
cd NFL-Sentiment-Analyzer

# Run the setup script
./scripts/setup.sh

# Or manually:
uv sync --dev
```

### 3. Activate the environment

```bash
# Activate the virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Or run commands directly with uv
uv run <command>
```

## Development Commands

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_main.py -v

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run tests in parallel
uv run pytest -n auto
```

### Code Quality
```bash
# Format code
uv run black app/

# Lint code
uv run flake8 app/

# Type checking
uv run mypy app/

# Security scanning
uv run bandit -r app/

# Run all quality checks
uv run black app/ && uv run flake8 app/ && uv run mypy app/
```

### Running the Application
```bash
# Development server with hot reload
uv run uvicorn app.main:app --reload

# Production server
uv run gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Dependency Management

### Adding Dependencies
```bash
# Add a runtime dependency
uv add fastapi

# Add a development dependency
uv add --dev pytest

# Add with version constraints
uv add "pandas>=2.0.0,<3.0.0"

# Add optional dependencies
uv add --optional mlops hopsworks
```

### Updating Dependencies
```bash
# Update all dependencies
uv sync --upgrade

# Update specific dependency
uv add fastapi@latest
```

### Managing Environments
```bash
# Create fresh environment
uv sync --reinstall

# Remove environment
rm -rf .venv
uv sync
```

## Project Structure

```
├── app/                    # Main application code
│   ├── api/               # FastAPI routes
│   ├── core/              # Core functionality
│   ├── models/            # Pydantic models
│   └── services/          # Business logic
├── tests/                 # Test files
├── scripts/               # Development scripts
├── pyproject.toml         # Project configuration
├── uv.lock               # Dependency lock file
└── .python-version       # Python version specification
```

## Environment Variables

Create a `.env` file in the project root:

```env
# Database
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=nfl_sentiment

# Redis
REDIS_URL=redis://localhost:6379

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here

# API Keys
TWITTER_API_KEY=your-twitter-api-key
OPENAI_API_KEY=your-openai-api-key
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're using `uv run` or have activated the virtual environment
2. **Dependency conflicts**: Try `uv sync --reinstall` to recreate the environment
3. **Python version issues**: Check `.python-version` file and ensure you have Python 3.11+

### Performance Tips

- Use `uv run` for one-off commands
- Activate the virtual environment for interactive development
- Use `uv sync --frozen` in CI/CD for faster, reproducible builds

## CI/CD

The project uses GitHub Actions with uv for fast, reliable builds:

- **Dependencies**: Cached and installed with `uv sync --dev`
- **Tests**: Run with `uv run pytest`
- **Linting**: Run with `uv run flake8` and `uv run black --check`
- **Type checking**: Run with `uv run mypy`

## Migration from pip

If you're migrating from the old pip-based setup:

1. Remove old files: `rm requirements.txt requirements-dev.txt`
2. Install uv: Follow installation instructions above
3. Run setup: `uv sync --dev`
4. Update scripts to use `uv run` instead of direct command calls

## Benefits of uv

- **Speed**: 10-100x faster than pip
- **Reliability**: Better dependency resolution
- **Reproducibility**: Lock files ensure consistent environments
- **Simplicity**: Single tool for all Python package management
- **Compatibility**: Works with existing Python ecosystem