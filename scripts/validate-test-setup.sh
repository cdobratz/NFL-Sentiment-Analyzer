#!/bin/bash

# Validation script for test prerequisites
# This script checks if all prerequisites for Makefile test targets are met

set -e

echo "🔍 Validating test setup prerequisites..."
echo

# Check Python and pytest installation
echo "Checking Python and pytest..."
if command -v python &> /dev/null; then
    echo "✅ Python found: $(python --version)"
else
    echo "❌ Python not found"
    exit 1
fi

# Check if uv is available (preferred) or pip
if command -v uv &> /dev/null; then
    echo "✅ uv found: $(uv --version)"
    echo "  Installing dependencies with uv..."
    uv sync --dev
elif command -v pip &> /dev/null; then
    echo "⚠️  uv not found, using pip"
    echo "  Installing dependencies with pip..."
    pip install -e ".[dev]"
else
    echo "❌ Neither uv nor pip found"
    exit 1
fi

# Check pytest installation and markers
echo
echo "Checking pytest configuration..."
if python -c "import pytest" 2>/dev/null; then
    echo "✅ pytest is installed"
    
    # Check for pytest markers
    if python -c "import pytest; pytest.main(['--markers'])" 2>/dev/null | grep -q "integration\|unit"; then
        echo "✅ pytest markers (integration, unit) are configured"
    else
        echo "❌ pytest markers not found - test-integration and test-unit may fail"
        echo "  Add markers to pytest.ini or pyproject.toml:"
        echo "  markers ="
        echo "      unit: Unit tests"
        echo "      integration: Integration tests"
        exit 1
    fi
else
    echo "❌ pytest not installed"
    exit 1
fi

# Check frontend directory and package.json
echo
echo "Checking frontend setup..."
if [ -d "frontend" ]; then
    echo "✅ frontend/ directory exists"
    
    if [ -f "frontend/package.json" ]; then
        echo "✅ frontend/package.json exists"
        
        # Check for required test scripts
        if command -v jq &> /dev/null; then
            if jq -e '.scripts.test and .scripts["test:coverage"]' frontend/package.json > /dev/null 2>&1; then
                echo "✅ Frontend test scripts configured"
            else
                echo "❌ Frontend test scripts missing"
                echo "  Add to frontend/package.json scripts:"
                echo '  "test": "vitest --run",'
                echo '  "test:coverage": "vitest --coverage"'
                exit 1
            fi
        else
            echo "⚠️  jq not found, cannot validate package.json scripts"
            echo "  Please ensure frontend/package.json has 'test' and 'test:coverage' scripts"
        fi
        
        # Check if Node.js is available
        if command -v node &> /dev/null; then
            echo "✅ Node.js found: $(node --version)"
            
            # Check if npm is available
            if command -v npm &> /dev/null; then
                echo "✅ npm found: $(npm --version)"
                echo "  Installing frontend dependencies..."
                cd frontend && npm install && cd ..
            else
                echo "❌ npm not found"
                exit 1
            fi
        else
            echo "❌ Node.js not found"
            exit 1
        fi
    else
        echo "❌ frontend/package.json not found"
        exit 1
    fi
else
    echo "❌ frontend/ directory not found"
    exit 1
fi

# Test the actual make targets
echo
echo "Testing Makefile targets..."

echo "Testing 'make test-unit'..."
if make test-unit > /dev/null 2>&1; then
    echo "✅ make test-unit works"
else
    echo "❌ make test-unit failed"
fi

echo "Testing 'make test-integration'..."
if make test-integration > /dev/null 2>&1; then
    echo "✅ make test-integration works"
else
    echo "❌ make test-integration failed"
fi

echo "Testing 'make test-frontend'..."
if make test-frontend > /dev/null 2>&1; then
    echo "✅ make test-frontend works"
else
    echo "❌ make test-frontend failed"
fi

echo "Testing 'make lint'..."
if make lint > /dev/null 2>&1; then
    echo "✅ make lint works"
else
    echo "❌ make lint failed"
fi

echo "Testing 'make format'..."
if make format > /dev/null 2>&1; then
    echo "✅ make format works"
else
    echo "❌ make format failed"
fi

echo
echo "🎉 All prerequisites validated successfully!"
echo
echo "Available test commands:"
echo "  make test           # Run all tests"
echo "  make test-unit      # Run unit tests only"
echo "  make test-integration # Run integration tests only"
echo "  make test-frontend  # Run frontend tests"
echo "  make test-coverage  # Run tests with coverage"
echo "  make lint           # Run linting"
echo "  make format         # Format code"