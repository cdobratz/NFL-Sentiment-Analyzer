#!/bin/bash

# Automated Backup Script for NFL Sentiment Analyzer
# This script creates backups of MongoDB and application data

set -e

# Configuration
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}
S3_BUCKET=${S3_BACKUP_BUCKET}

# MongoDB configuration
MONGO_HOST=${MONGO_HOST:-mongodb}
MONGO_PORT=${MONGO_PORT:-27017}
MONGO_DB=${DATABASE_NAME:-nfl_sentiment}
MONGO_USER=${MONGO_ROOT_USERNAME:-admin}
MONGO_PASS=${MONGO_ROOT_PASSWORD}

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Create backup directory
mkdir -p "$BACKUP_DIR"

log "Starting backup process..."

# MongoDB backup
log "Creating MongoDB backup..."
MONGO_BACKUP_FILE="$BACKUP_DIR/mongodb_backup_$DATE.gz"

if [ -n "$MONGO_PASS" ]; then
    mongodump --host "$MONGO_HOST:$MONGO_PORT" \
              --db "$MONGO_DB" \
              --username "$MONGO_USER" \
              --password "$MONGO_PASS" \
              --authenticationDatabase admin \
              --gzip \
              --archive="$MONGO_BACKUP_FILE"
else
    mongodump --host "$MONGO_HOST:$MONGO_PORT" \
              --db "$MONGO_DB" \
              --gzip \
              --archive="$MONGO_BACKUP_FILE"
fi

log "MongoDB backup completed: $MONGO_BACKUP_FILE"

# Application logs backup
log "Creating application logs backup..."
LOGS_BACKUP_FILE="$BACKUP_DIR/logs_backup_$DATE.tar.gz"
if [ -d "/app/logs" ]; then
    tar -czf "$LOGS_BACKUP_FILE" -C /app logs/
    log "Logs backup completed: $LOGS_BACKUP_FILE"
fi

# Configuration backup
log "Creating configuration backup..."
CONFIG_BACKUP_FILE="$BACKUP_DIR/config_backup_$DATE.tar.gz"
tar -czf "$CONFIG_BACKUP_FILE" \
    --exclude='*.env*' \
    --exclude='secrets' \
    -C / \
    app/nginx/ \
    app/monitoring/ \
    app/scripts/ \
    2>/dev/null || true

log "Configuration backup completed: $CONFIG_BACKUP_FILE"

# Upload to S3 if configured
if [ -n "$S3_BUCKET" ] && [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    log "Uploading backups to S3..."
    
    aws s3 cp "$MONGO_BACKUP_FILE" "s3://$S3_BUCKET/mongodb/" --storage-class STANDARD_IA
    aws s3 cp "$LOGS_BACKUP_FILE" "s3://$S3_BUCKET/logs/" --storage-class STANDARD_IA
    aws s3 cp "$CONFIG_BACKUP_FILE" "s3://$S3_BUCKET/config/" --storage-class STANDARD_IA
    
    log "Backups uploaded to S3 successfully"
fi

# Cleanup old local backups
log "Cleaning up old backups (older than $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "*.gz" -type f -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "*.tar.gz" -type f -mtime +$RETENTION_DAYS -delete

# Cleanup old S3 backups if configured
if [ -n "$S3_BUCKET" ] && [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
    CUTOFF_DATE=$(date -d "$RETENTION_DAYS days ago" +%Y-%m-%d)
    aws s3 ls "s3://$S3_BUCKET/" --recursive | while read -r line; do
        FILE_DATE=$(echo "$line" | awk '{print $1}')
        FILE_PATH=$(echo "$line" | awk '{print $4}')
        
        if [[ "$FILE_DATE" < "$CUTOFF_DATE" ]]; then
            aws s3 rm "s3://$S3_BUCKET/$FILE_PATH"
            log "Deleted old S3 backup: $FILE_PATH"
        fi
    done
fi

# Verify backup integrity
log "Verifying backup integrity..."
if [ -f "$MONGO_BACKUP_FILE" ]; then
    if gzip -t "$MONGO_BACKUP_FILE"; then
        log "MongoDB backup integrity verified"
    else
        log "ERROR: MongoDB backup integrity check failed"
        exit 1
    fi
fi

log "Backup process completed successfully"

# Send notification (if configured)
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -X POST -H 'Content-type: application/json' \
         --data "{\"text\":\"✅ NFL Sentiment Analyzer backup completed successfully at $(date)\"}" \
         "$SLACK_WEBHOOK_URL"
fi