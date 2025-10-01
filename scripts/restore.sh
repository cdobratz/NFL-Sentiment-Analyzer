#!/bin/bash

# Disaster Recovery Script for NFL Sentiment Analyzer
# This script restores backups of MongoDB and application data

set -e

# Configuration
BACKUP_DIR="/backups"
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

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -d DATE     Restore from specific date (YYYYMMDD_HHMMSS)"
    echo "  -l          List available backups"
    echo "  -s          Download from S3 before restore"
    echo "  -h          Show this help message"
    exit 1
}

# List available backups
list_backups() {
    log "Available local backups:"
    ls -la "$BACKUP_DIR"/mongodb_backup_*.gz 2>/dev/null || log "No local MongoDB backups found"
    
    if [ -n "$S3_BUCKET" ] && [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
        log "Available S3 backups:"
        aws s3 ls "s3://$S3_BUCKET/mongodb/" --recursive
    fi
}

# Download backup from S3
download_from_s3() {
    local backup_date=$1
    
    if [ -z "$S3_BUCKET" ] || [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        log "ERROR: S3 configuration not found"
        exit 1
    fi
    
    log "Downloading backup from S3..."
    aws s3 cp "s3://$S3_BUCKET/mongodb/mongodb_backup_${backup_date}.gz" "$BACKUP_DIR/"
    aws s3 cp "s3://$S3_BUCKET/logs/logs_backup_${backup_date}.tar.gz" "$BACKUP_DIR/" || true
    aws s3 cp "s3://$S3_BUCKET/config/config_backup_${backup_date}.tar.gz" "$BACKUP_DIR/" || true
}

# Restore MongoDB
restore_mongodb() {
    local backup_file=$1
    
    if [ ! -f "$backup_file" ]; then
        log "ERROR: Backup file not found: $backup_file"
        exit 1
    fi
    
    log "Restoring MongoDB from: $backup_file"
    
    # Verify backup integrity
    if ! gzip -t "$backup_file"; then
        log "ERROR: Backup file is corrupted"
        exit 1
    fi
    
    # Create a backup of current data before restore
    CURRENT_BACKUP="$BACKUP_DIR/pre_restore_backup_$(date +%Y%m%d_%H%M%S).gz"
    log "Creating backup of current data: $CURRENT_BACKUP"
    
    if [ -n "$MONGO_PASS" ]; then
        mongodump --host "$MONGO_HOST:$MONGO_PORT" \
                  --db "$MONGO_DB" \
                  --username "$MONGO_USER" \
                  --password "$MONGO_PASS" \
                  --authenticationDatabase admin \
                  --gzip \
                  --archive="$CURRENT_BACKUP" || log "Warning: Could not backup current data"
        
        # Restore from backup
        mongorestore --host "$MONGO_HOST:$MONGO_PORT" \
                     --username "$MONGO_USER" \
                     --password "$MONGO_PASS" \
                     --authenticationDatabase admin \
                     --drop \
                     --gzip \
                     --archive="$backup_file"
    else
        mongodump --host "$MONGO_HOST:$MONGO_PORT" \
                  --db "$MONGO_DB" \
                  --gzip \
                  --archive="$CURRENT_BACKUP" || log "Warning: Could not backup current data"
        
        # Restore from backup
        mongorestore --host "$MONGO_HOST:$MONGO_PORT" \
                     --drop \
                     --gzip \
                     --archive="$backup_file"
    fi
    
    log "MongoDB restore completed successfully"
}

# Restore application logs
restore_logs() {
    local backup_file=$1
    
    if [ ! -f "$backup_file" ]; then
        log "Warning: Logs backup file not found: $backup_file"
        return
    fi
    
    log "Restoring application logs from: $backup_file"
    
    # Create backup of current logs
    if [ -d "/app/logs" ]; then
        mv /app/logs "/app/logs.backup.$(date +%Y%m%d_%H%M%S)" || true
    fi
    
    # Extract logs backup
    tar -xzf "$backup_file" -C /app/
    
    log "Application logs restored successfully"
}

# Restore configuration
restore_config() {
    local backup_file=$1
    
    if [ ! -f "$backup_file" ]; then
        log "Warning: Configuration backup file not found: $backup_file"
        return
    fi
    
    log "Restoring configuration from: $backup_file"
    
    # Extract configuration backup
    tar -xzf "$backup_file" -C /
    
    log "Configuration restored successfully"
}

# Main script
BACKUP_DATE=""
LIST_ONLY=false
DOWNLOAD_S3=false

while getopts "d:lsh" opt; do
    case $opt in
        d)
            BACKUP_DATE="$OPTARG"
            ;;
        l)
            LIST_ONLY=true
            ;;
        s)
            DOWNLOAD_S3=true
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

# List backups and exit if requested
if [ "$LIST_ONLY" = true ]; then
    list_backups
    exit 0
fi

# Determine backup date
if [ -z "$BACKUP_DATE" ]; then
    # Find the most recent backup
    LATEST_BACKUP=$(ls -1 "$BACKUP_DIR"/mongodb_backup_*.gz 2>/dev/null | sort -r | head -n1)
    if [ -n "$LATEST_BACKUP" ]; then
        BACKUP_DATE=$(basename "$LATEST_BACKUP" | sed 's/mongodb_backup_\(.*\)\.gz/\1/')
    else
        log "ERROR: No backups found and no date specified"
        exit 1
    fi
fi

log "Using backup date: $BACKUP_DATE"

# Download from S3 if requested
if [ "$DOWNLOAD_S3" = true ]; then
    download_from_s3 "$BACKUP_DATE"
fi

# Confirm restore operation
echo "WARNING: This will replace current data with backup from $BACKUP_DATE"
echo "Current data will be backed up before restore."
read -p "Are you sure you want to continue? (yes/no): " -r
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    log "Restore operation cancelled"
    exit 0
fi

log "Starting restore process..."

# Restore MongoDB
MONGO_BACKUP_FILE="$BACKUP_DIR/mongodb_backup_${BACKUP_DATE}.gz"
restore_mongodb "$MONGO_BACKUP_FILE"

# Restore logs
LOGS_BACKUP_FILE="$BACKUP_DIR/logs_backup_${BACKUP_DATE}.tar.gz"
restore_logs "$LOGS_BACKUP_FILE"

# Restore configuration
CONFIG_BACKUP_FILE="$BACKUP_DIR/config_backup_${BACKUP_DATE}.tar.gz"
restore_config "$CONFIG_BACKUP_FILE"

log "Restore process completed successfully"

# Send notification (if configured)
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -X POST -H 'Content-type: application/json' \
         --data "{\"text\":\"🔄 NFL Sentiment Analyzer restore completed successfully from backup $BACKUP_DATE at $(date)\"}" \
         "$SLACK_WEBHOOK_URL"
fi