#!/bin/bash

# Setup Cron Jobs for NFL Sentiment Analyzer
# This script configures automated tasks like backups and security scans

set -e

# Configuration
CRON_USER=${CRON_USER:-root}
BACKUP_SCHEDULE=${BACKUP_SCHEDULE:-"0 2 * * *"}  # Daily at 2 AM
SECURITY_SCAN_SCHEDULE=${SECURITY_SCAN_SCHEDULE:-"0 4 * * 0"}  # Weekly on Sunday at 4 AM

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Setting up cron jobs for automated tasks..."

# Create cron jobs file
CRON_FILE="/tmp/nfl-sentiment-cron"

cat > "$CRON_FILE" << EOF
# NFL Sentiment Analyzer Automated Tasks
# Generated on $(date)

# Daily backup at 2 AM
$BACKUP_SCHEDULE cd /app && ./scripts/backup.sh >> /var/log/backup.log 2>&1

# Weekly security scan on Sunday at 4 AM
$SECURITY_SCAN_SCHEDULE cd /app && ./scripts/security-scan.sh >> /var/log/security-scan.log 2>&1

# Daily log rotation at 1 AM
0 1 * * * /usr/sbin/logrotate /etc/logrotate.d/nfl-sentiment

# Weekly cleanup of old Docker images on Saturday at 3 AM
0 3 * * 6 docker image prune -f >> /var/log/docker-cleanup.log 2>&1

# Daily health check at every hour
0 * * * * curl -f http://localhost:8000/health > /dev/null 2>&1 || echo "Health check failed at \$(date)" >> /var/log/health-check.log

EOF

# Install cron jobs
log "Installing cron jobs for user: $CRON_USER"
crontab -u "$CRON_USER" "$CRON_FILE"

# Create log rotation configuration
log "Setting up log rotation..."
cat > "/etc/logrotate.d/nfl-sentiment" << EOF
/app/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 app app
    postrotate
        # Restart services if needed
        docker service update --force nfl-sentiment_api || true
    endscript
}

/var/log/nginx/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 nginx nginx
    postrotate
        docker service update --force nfl-sentiment_nginx || true
    endscript
}

/var/log/backup.log
/var/log/security-scan.log
/var/log/docker-cleanup.log
/var/log/health-check.log {
    weekly
    missingok
    rotate 12
    compress
    delaycompress
    notifempty
    create 644 root root
}
EOF

# Create systemd service for monitoring cron jobs
log "Creating systemd service for cron monitoring..."
cat > "/etc/systemd/system/nfl-sentiment-monitor.service" << EOF
[Unit]
Description=NFL Sentiment Analyzer Monitoring Service
After=network.target

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'if ! pgrep -x "crond" > /dev/null; then echo "Cron service is not running" | logger -t nfl-sentiment-monitor; systemctl start crond; fi'

[Install]
WantedBy=multi-user.target
EOF

cat > "/etc/systemd/system/nfl-sentiment-monitor.timer" << EOF
[Unit]
Description=Run NFL Sentiment Analyzer monitoring every 5 minutes
Requires=nfl-sentiment-monitor.service

[Timer]
OnCalendar=*:0/5
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Enable and start the monitoring service
systemctl daemon-reload
systemctl enable nfl-sentiment-monitor.timer
systemctl start nfl-sentiment-monitor.timer

# Create monitoring script for cron job status
log "Creating cron job monitoring script..."
cat > "/usr/local/bin/check-cron-jobs.sh" << 'EOF'
#!/bin/bash

# Check if cron jobs are running properly
LOG_FILE="/var/log/cron-monitor.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

# Check if backup ran in the last 25 hours
LAST_BACKUP=$(find /backups -name "mongodb_backup_*.gz" -mtime -1 | wc -l)
if [ "$LAST_BACKUP" -eq 0 ]; then
    log "WARNING: No backup found in the last 24 hours"
    # Send alert if webhook is configured
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
             --data '{"text":"⚠️ NFL Sentiment Analyzer: No backup found in the last 24 hours"}' \
             "$SLACK_WEBHOOK_URL"
    fi
fi

# Check if security scan ran in the last 8 days
LAST_SECURITY_SCAN=$(find /tmp/security-scans -name "security-report-*.html" -mtime -8 | wc -l)
if [ "$LAST_SECURITY_SCAN" -eq 0 ]; then
    log "WARNING: No security scan found in the last 7 days"
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
             --data '{"text":"⚠️ NFL Sentiment Analyzer: No security scan found in the last 7 days"}' \
             "$SLACK_WEBHOOK_URL"
    fi
fi

# Check disk space
DISK_USAGE=$(df / | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 85 ]; then
    log "WARNING: Disk usage is at ${DISK_USAGE}%"
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -X POST -H 'Content-type: application/json' \
             --data "{\"text\":\"⚠️ NFL Sentiment Analyzer: Disk usage is at ${DISK_USAGE}%\"}" \
             "$SLACK_WEBHOOK_URL"
    fi
fi

log "Cron job monitoring completed"
EOF

chmod +x /usr/local/bin/check-cron-jobs.sh

# Add monitoring script to cron
echo "0 6 * * * /usr/local/bin/check-cron-jobs.sh" | crontab -u "$CRON_USER" -

# Cleanup temporary files
rm -f "$CRON_FILE"

log "Cron jobs setup completed successfully!"
log "Installed jobs:"
crontab -u "$CRON_USER" -l

log "Log rotation configured for:"
log "- Application logs: /app/logs/*.log"
log "- Nginx logs: /var/log/nginx/*.log"
log "- System logs: /var/log/backup.log, /var/log/security-scan.log, etc."

log "Monitoring service enabled: nfl-sentiment-monitor.timer"
log "Cron job monitoring script: /usr/local/bin/check-cron-jobs.sh"