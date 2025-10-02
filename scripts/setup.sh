#!/bin/bash
# Setup script for NFL Sentiment Analyzer

set -e

echo "🚀 Setting up NFL Sentiment Analyzer development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "✅ uv found"

# Install dependencies
echo "📦 Installing dependencies..."
uv sync --dev

echo "🧪 Running tests to verify setup..."
uv run pytest tests/test_main.py -v

echo "🎉 Setup complete! You can now:"
echo "   - Run tests: uv run pytest"
echo "   - Start the server: uv run uvicorn app.main:app --reload"
echo "   - Run linting: uv run black app/ && uv run flake8 app/"
echo "   - Run type checking: uv run mypy app/"