#!/bin/sh

# Railway startup script for NFL Sentiment Frontend

set -e

echo "Starting NFL Sentiment Frontend..."
echo "Environment: ${ENVIRONMENT:-production}"
echo "Port: ${PORT:-3000}"

# Start the frontend server
echo "Starting frontend server on port ${PORT:-3000}..."
exec serve -s dist -l ${PORT:-3000}