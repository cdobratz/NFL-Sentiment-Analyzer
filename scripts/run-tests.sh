#!/bin/bash

# NFL Sentiment Analyzer - Test Runner Script
# This script runs all tests for the application

set -e

echo "🧪 NFL Sentiment Analyzer - Running Tests"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "requirements.txt" ] || [ ! -d "frontend" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Parse command line arguments
BACKEND_ONLY=false
FRONTEND_ONLY=false
COVERAGE=false
INTEGRATION=false
UNIT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --backend-only)
            BACKEND_ONLY=true
            shift
            ;;
        --frontend-only)
            FRONTEND_ONLY=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --integration)
            INTEGRATION=true
            shift
            ;;
        --unit)
            UNIT=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --backend-only    Run only backend tests"
            echo "  --frontend-only   Run only frontend tests"
            echo "  --coverage        Run tests with coverage reporting"
            echo "  --integration     Run only integration tests"
            echo "  --unit           Run only unit tests"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if required services are running
check_services() {
    print_status "Checking required services..."
    
    # Check MongoDB
    if ! nc -z localhost 27017 2>/dev/null; then
        print_warning "MongoDB not running on localhost:27017"
        print_status "Starting MongoDB with Docker..."
        docker run -d --name test-mongodb -p 27017:27017 mongo:7.0 || true
        sleep 5
    fi
    
    # Check Redis
    if ! nc -z localhost 6379 2>/dev/null; then
        print_warning "Redis not running on localhost:6379"
        print_status "Starting Redis with Docker..."
        docker run -d --name test-redis -p 6379:6379 redis:7.2-alpine || true
        sleep 2
    fi
}

# Run backend tests
run_backend_tests() {
    print_status "Running backend tests..."
    
    if [ "$COVERAGE" = true ]; then
        pytest --cov=app --cov-report=html --cov-report=xml --cov-report=term-missing
    elif [ "$INTEGRATION" = true ]; then
        pytest -m integration
    elif [ "$UNIT" = true ]; then
        pytest -m unit
    else
        pytest
    fi
}

# Run frontend tests
run_frontend_tests() {
    print_status "Running frontend tests..."
    
    cd frontend
    
    if [ "$COVERAGE" = true ]; then
        npm run test:coverage
    else
        npm run test
    fi
    
    cd ..
}

# Main execution
main() {
    # Check services unless running frontend only
    if [ "$FRONTEND_ONLY" != true ]; then
        check_services
    fi
    
    # Run tests based on options
    if [ "$BACKEND_ONLY" = true ]; then
        run_backend_tests
    elif [ "$FRONTEND_ONLY" = true ]; then
        run_frontend_tests
    else
        run_backend_tests
        run_frontend_tests
    fi
    
    print_status "All tests completed successfully! ✅"
}

# Cleanup function
cleanup() {
    print_status "Cleaning up test containers..."
    docker rm -f test-mongodb test-redis 2>/dev/null || true
}

# Set up trap for cleanup
trap cleanup EXIT

# Run main function
main