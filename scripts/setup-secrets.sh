#!/bin/bash

# Production Secrets Setup Script
# This script sets up Docker secrets for production deployment

set -e

echo "Setting up production secrets..."

# Create secrets directory if it doesn't exist
mkdir -p /var/lib/docker/secrets

# Function to create Docker secret
create_secret() {
    local secret_name=$1
    local secret_value=$2
    
    if [ -z "$secret_value" ]; then
        echo "Warning: Empty value for secret $secret_name"
        return 1
    fi
    
    echo "$secret_value" | docker secret create "$secret_name" - 2>/dev/null || {
        echo "Secret $secret_name already exists, updating..."
        docker secret rm "$secret_name" 2>/dev/null || true
        echo "$secret_value" | docker secret create "$secret_name" -
    }
    
    echo "✓ Created secret: $secret_name"
}

# Load environment variables
if [ -f .env.production ]; then
    source .env.production
else
    echo "Error: .env.production file not found"
    exit 1
fi

# Create secrets from environment variables
create_secret "mongo_root_password" "$MONGO_ROOT_PASSWORD"
create_secret "redis_password" "$REDIS_PASSWORD"
create_secret "jwt_secret_key" "$JWT_SECRET_KEY"
create_secret "twitter_api_key" "$TWITTER_API_KEY"
create_secret "twitter_api_secret" "$TWITTER_API_SECRET"
create_secret "twitter_bearer_token" "$TWITTER_BEARER_TOKEN"
create_secret "espn_api_key" "$ESPN_API_KEY"
create_secret "draftkings_api_key" "$DRAFTKINGS_API_KEY"
create_secret "sentry_dsn" "$SENTRY_DSN"
create_secret "grafana_password" "$GRAFANA_PASSWORD"

# SSL certificates (if provided)
if [ -f "ssl/cert.pem" ] && [ -f "ssl/key.pem" ]; then
    docker secret create ssl_cert ssl/cert.pem 2>/dev/null || {
        docker secret rm ssl_cert 2>/dev/null || true
        docker secret create ssl_cert ssl/cert.pem
    }
    docker secret create ssl_key ssl/key.pem 2>/dev/null || {
        docker secret rm ssl_key 2>/dev/null || true
        docker secret create ssl_key ssl/key.pem
    }
    echo "✓ Created SSL certificate secrets"
fi

echo "All secrets have been created successfully!"
echo "You can now deploy the application using: docker stack deploy -c docker-compose.yml -c docker-compose.prod.yml nfl-sentiment"