#!/bin/bash

# Railway startup script for NFL Sentiment API

set -e

echo "Starting NFL Sentiment API..."
echo "Environment: ${ENVIRONMENT:-production}"
echo "Port: ${PORT:-8000}"
echo "Workers: ${WORKERS:-2}"

# Wait a moment for any system setup
sleep 2

# Check if we can import the app
echo "Testing app import..."
uv run python -c "from app.main import app; print('App imported successfully')"

# Start the server
echo "Starting Gunicorn server..."
exec uv run gunicorn app.main:app \
    -w ${WORKERS:-2} \
    -k uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:${PORT:-8000} \
    --access-logfile - \
    --error-logfile - \
    --log-level ${LOG_LEVEL:-info} \
    --timeout 120 \
    --keepalive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload