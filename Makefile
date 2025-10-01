.PHONY: help dev prod build clean test lint format install

# Default target
help:
	@echo "Available commands:"
	@echo "  dev      - Start development environment"
	@echo "  prod     - Start production environment"
	@echo "  build    - Build all services"
	@echo "  clean    - Clean up containers and volumes"
	@echo "  test     - Run tests"
	@echo "  lint     - Run linting"
	@echo "  format   - Format code"
	@echo "  install  - Install dependencies"

# Development environment
dev:
	docker-compose -f docker-compose.dev.yml up --build

# Production environment
prod:
	docker-compose up --build -d

# Build all services
build:
	docker-compose build
	cd frontend && npm run build

# Clean up
clean:
	docker-compose down -v
	docker-compose -f docker-compose.dev.yml down -v
	docker system prune -f

# Run tests
test:
	# Backend tests
	python -m pytest
	# Frontend tests
	cd frontend && npm run test

# Run backend tests with coverage
test-backend:
	python -m pytest --cov=app --cov-report=html --cov-report=term-missing

# Run frontend tests
test-frontend:
	cd frontend && npm run test

# Run tests with coverage
test-coverage:
	python -m pytest --cov=app --cov-report=html --cov-report=xml --cov-report=term-missing
	cd frontend && npm run test:coverage

# Run integration tests
test-integration:
	python -m pytest -m integration

# Run unit tests only
test-unit:
	python -m pytest -m unit

# Run linting
lint:
	# Backend linting
	cd app && python -m flake8 .
	cd app && python -m mypy .
	# Frontend linting
	cd frontend && npm run lint

# Format code
format:
	# Backend formatting
	cd app && python -m black .
	# Frontend formatting
	cd frontend && npm run format

# Install dependencies
install:
	# Backend dependencies
	pip install -r requirements.txt
	# Frontend dependencies
	cd frontend && npm install

# Database setup
db-setup:
	docker-compose -f docker-compose.dev.yml up mongodb redis -d

# Stop services
stop:
	docker-compose down
	docker-compose -f docker-compose.dev.yml down

# View logs
logs:
	docker-compose logs -f

# Development backend only
dev-backend:
	cd app && uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Development frontend only
dev-frontend:
	cd frontend && npm run dev