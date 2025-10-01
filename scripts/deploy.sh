#!/bin/bash

# Production Deployment Script for NFL Sentiment Analyzer
# This script handles the complete deployment process

set -e

# Configuration
ENVIRONMENT=${ENVIRONMENT:-production}
STACK_NAME="nfl-sentiment"
BACKUP_BEFORE_DEPLOY=${BACKUP_BEFORE_DEPLOY:-true}

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -e ENV      Environment (production|staging) [default: production]"
    echo "  -s STACK    Stack name [default: nfl-sentiment]"
    echo "  -b          Skip backup before deployment"
    echo "  -h          Show this help message"
    exit 1
}

# Health check function
health_check() {
    local service_url=$1
    local max_attempts=30
    local attempt=1
    
    log "Performing health check for $service_url..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f -s "$service_url/health" > /dev/null; then
            log "Health check passed for $service_url"
            return 0
        fi
        
        log "Health check attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    log "ERROR: Health check failed for $service_url after $max_attempts attempts"
    return 1
}

# Rollback function
rollback() {
    log "Rolling back deployment..."
    
    if docker stack ls | grep -q "$STACK_NAME"; then
        docker stack rm "$STACK_NAME"
        log "Stack removed, waiting for cleanup..."
        sleep 30
    fi
    
    # Restore from backup if available
    if [ -f "./scripts/restore.sh" ]; then
        log "Attempting to restore from backup..."
        ./scripts/restore.sh -d "$(ls -1 /backups/mongodb_backup_*.gz 2>/dev/null | sort -r | head -n1 | sed 's/.*mongodb_backup_\(.*\)\.gz/\1/')" || true
    fi
    
    log "Rollback completed"
}

# Parse command line arguments
while getopts "e:s:bh" opt; do
    case $opt in
        e)
            ENVIRONMENT="$OPTARG"
            ;;
        s)
            STACK_NAME="$OPTARG"
            ;;
        b)
            BACKUP_BEFORE_DEPLOY=false
            ;;
        h)
            usage
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
    esac
done

log "Starting deployment process for environment: $ENVIRONMENT"

# Validate environment
if [ "$ENVIRONMENT" != "production" ] && [ "$ENVIRONMENT" != "staging" ]; then
    log "ERROR: Invalid environment. Must be 'production' or 'staging'"
    exit 1
fi

# Check if Docker Swarm is initialized
if ! docker info | grep -q "Swarm: active"; then
    log "Initializing Docker Swarm..."
    docker swarm init
fi

# Load environment variables
ENV_FILE=".env.$ENVIRONMENT"
if [ -f "$ENV_FILE" ]; then
    log "Loading environment variables from $ENV_FILE"
    set -a
    source "$ENV_FILE"
    set +a
else
    log "WARNING: Environment file $ENV_FILE not found"
fi

# Validate required environment variables
REQUIRED_VARS=(
    "MONGO_ROOT_PASSWORD"
    "REDIS_PASSWORD"
    "JWT_SECRET_KEY"
)

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        log "ERROR: Required environment variable $var is not set"
        exit 1
    fi
done

# Create backup before deployment
if [ "$BACKUP_BEFORE_DEPLOY" = true ]; then
    log "Creating backup before deployment..."
    if [ -f "./scripts/backup.sh" ]; then
        ./scripts/backup.sh || log "WARNING: Backup failed, continuing with deployment"
    else
        log "WARNING: Backup script not found, skipping backup"
    fi
fi

# Setup secrets
log "Setting up Docker secrets..."
if [ -f "./scripts/setup-secrets.sh" ]; then
    ./scripts/setup-secrets.sh
else
    log "WARNING: Secrets setup script not found"
fi

# Pull latest images
log "Pulling latest Docker images..."
docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" pull

# Deploy the stack
log "Deploying Docker stack: $STACK_NAME"
docker stack deploy \
    -c docker-compose.yml \
    -c "docker-compose.$ENVIRONMENT.yml" \
    "$STACK_NAME"

# Wait for services to start
log "Waiting for services to start..."
sleep 60

# Perform health checks
log "Performing health checks..."
HEALTH_CHECK_FAILED=false

# Check API health
if ! health_check "http://localhost:8000"; then
    HEALTH_CHECK_FAILED=true
fi

# Check frontend health (if applicable)
if ! health_check "http://localhost:3000"; then
    log "WARNING: Frontend health check failed, but continuing..."
fi

# Check database connectivity
log "Checking database connectivity..."
if ! docker exec $(docker ps -q -f name="${STACK_NAME}_mongodb") mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
    log "ERROR: Database connectivity check failed"
    HEALTH_CHECK_FAILED=true
fi

# Check Redis connectivity
log "Checking Redis connectivity..."
if ! docker exec $(docker ps -q -f name="${STACK_NAME}_redis") redis-cli ping > /dev/null 2>&1; then
    log "ERROR: Redis connectivity check failed"
    HEALTH_CHECK_FAILED=true
fi

# Handle health check failures
if [ "$HEALTH_CHECK_FAILED" = true ]; then
    log "ERROR: Health checks failed, initiating rollback..."
    rollback
    exit 1
fi

# Run smoke tests
log "Running smoke tests..."
SMOKE_TESTS_PASSED=true

# Test API endpoints
if ! curl -f -s "http://localhost:8000/" > /dev/null; then
    log "ERROR: API root endpoint test failed"
    SMOKE_TESTS_PASSED=false
fi

# Test database operations
if ! curl -f -s -X POST "http://localhost:8000/analyze" \
    -H "Content-Type: application/json" \
    -d '{"text": "Test sentiment analysis"}' > /dev/null; then
    log "ERROR: API functionality test failed"
    SMOKE_TESTS_PASSED=false
fi

if [ "$SMOKE_TESTS_PASSED" = false ]; then
    log "ERROR: Smoke tests failed, initiating rollback..."
    rollback
    exit 1
fi

# Update monitoring dashboards
log "Updating monitoring configuration..."
if docker service ls | grep -q "${STACK_NAME}_prometheus"; then
    docker service update --force "${STACK_NAME}_prometheus"
fi

if docker service ls | grep -q "${STACK_NAME}_grafana"; then
    docker service update --force "${STACK_NAME}_grafana"
fi

# Run security scan
log "Running post-deployment security scan..."
if [ -f "./scripts/security-scan.sh" ]; then
    ./scripts/security-scan.sh || log "WARNING: Security scan failed"
fi

# Display deployment status
log "Deployment completed successfully!"
echo ""
echo "=== Deployment Summary ==="
echo "Environment: $ENVIRONMENT"
echo "Stack Name: $STACK_NAME"
echo "Services:"
docker service ls --filter label=com.docker.stack.namespace="$STACK_NAME"
echo ""
echo "=== Service URLs ==="
echo "Frontend: http://localhost:3000"
echo "API: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo "Grafana: http://localhost:3001"
echo "Prometheus: http://localhost:9090"
echo ""

# Send success notification
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -X POST -H 'Content-type: application/json' \
         --data "{\"text\":\"✅ NFL Sentiment Analyzer deployed successfully to $ENVIRONMENT environment at $(date)\"}" \
         "$SLACK_WEBHOOK_URL"
fi

log "Deployment process completed successfully!"