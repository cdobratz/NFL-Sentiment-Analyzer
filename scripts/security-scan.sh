#!/bin/bash

# Security Scanning Script for NFL Sentiment Analyzer
# This script runs various security scans and vulnerability assessments

set -e

# Configuration
SCAN_RESULTS_DIR="/tmp/security-scans"
DATE=$(date +%Y%m%d_%H%M%S)

# Logging
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Create results directory
mkdir -p "$SCAN_RESULTS_DIR"

log "Starting security scanning process..."

# 1. Container vulnerability scanning with Trivy
log "Running Trivy container vulnerability scan..."
if command -v trivy &> /dev/null; then
    trivy image --format json --output "$SCAN_RESULTS_DIR/trivy-container-$DATE.json" nfl-sentiment-analyzer:latest || true
    trivy fs --format json --output "$SCAN_RESULTS_DIR/trivy-filesystem-$DATE.json" . || true
    log "Trivy scan completed"
else
    log "Warning: Trivy not installed, skipping container vulnerability scan"
fi

# 2. Python dependency scanning with Safety
log "Running Python dependency vulnerability scan..."
if command -v safety &> /dev/null; then
    safety check --json --output "$SCAN_RESULTS_DIR/safety-$DATE.json" || true
    log "Safety scan completed"
else
    log "Warning: Safety not installed, installing..."
    pip install safety
    safety check --json --output "$SCAN_RESULTS_DIR/safety-$DATE.json" || true
fi

# 3. Secrets scanning with TruffleHog
log "Running secrets scanning..."
if command -v trufflehog &> /dev/null; then
    trufflehog filesystem . --json > "$SCAN_RESULTS_DIR/trufflehog-$DATE.json" || true
    log "TruffleHog scan completed"
else
    log "Warning: TruffleHog not installed, skipping secrets scan"
fi

# 4. Static code analysis with Bandit
log "Running static code analysis..."
if command -v bandit &> /dev/null; then
    bandit -r app/ -f json -o "$SCAN_RESULTS_DIR/bandit-$DATE.json" || true
    log "Bandit scan completed"
else
    log "Warning: Bandit not installed, installing..."
    pip install bandit
    bandit -r app/ -f json -o "$SCAN_RESULTS_DIR/bandit-$DATE.json" || true
fi

# 5. Docker security scanning
log "Running Docker security best practices scan..."
if command -v docker-bench-security &> /dev/null; then
    docker run --rm --net host --pid host --userns host --cap-add audit_control \
        -e DOCKER_CONTENT_TRUST=$DOCKER_CONTENT_TRUST \
        -v /etc:/etc:ro \
        -v /usr/bin/containerd:/usr/bin/containerd:ro \
        -v /usr/bin/runc:/usr/bin/runc:ro \
        -v /usr/lib/systemd:/usr/lib/systemd:ro \
        -v /var/lib:/var/lib:ro \
        -v /var/run/docker.sock:/var/run/docker.sock:ro \
        --label docker_bench_security \
        docker/docker-bench-security > "$SCAN_RESULTS_DIR/docker-bench-$DATE.txt" || true
    log "Docker security scan completed"
else
    log "Warning: Docker Bench Security not available, skipping Docker security scan"
fi

# 6. Network security scanning (if nmap is available)
log "Running network security scan..."
if command -v nmap &> /dev/null; then
    nmap -sV -sC -O localhost > "$SCAN_RESULTS_DIR/nmap-$DATE.txt" || true
    log "Network scan completed"
else
    log "Warning: Nmap not installed, skipping network scan"
fi

# 7. SSL/TLS configuration testing (if testssl.sh is available)
log "Running SSL/TLS configuration test..."
if [ -f "/usr/local/bin/testssl.sh" ]; then
    /usr/local/bin/testssl.sh --jsonfile "$SCAN_RESULTS_DIR/testssl-$DATE.json" https://localhost || true
    log "SSL/TLS test completed"
else
    log "Warning: testssl.sh not available, skipping SSL/TLS test"
fi

# 8. Generate security report
log "Generating security report..."
REPORT_FILE="$SCAN_RESULTS_DIR/security-report-$DATE.html"

cat > "$REPORT_FILE" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Security Scan Report - $DATE</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .critical { background-color: #ffebee; border-color: #f44336; }
        .warning { background-color: #fff3e0; border-color: #ff9800; }
        .info { background-color: #e3f2fd; border-color: #2196f3; }
        .success { background-color: #e8f5e8; border-color: #4caf50; }
        pre { background-color: #f5f5f5; padding: 10px; border-radius: 3px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>NFL Sentiment Analyzer - Security Scan Report</h1>
        <p>Generated on: $(date)</p>
        <p>Scan ID: $DATE</p>
    </div>

    <div class="section info">
        <h2>Scan Summary</h2>
        <p>This report contains the results of automated security scans performed on the NFL Sentiment Analyzer application.</p>
        <ul>
            <li>Container Vulnerability Scan (Trivy)</li>
            <li>Python Dependency Scan (Safety)</li>
            <li>Secrets Detection (TruffleHog)</li>
            <li>Static Code Analysis (Bandit)</li>
            <li>Docker Security Best Practices</li>
            <li>Network Security Scan</li>
            <li>SSL/TLS Configuration Test</li>
        </ul>
    </div>

    <div class="section">
        <h2>Scan Results</h2>
        <p>Detailed scan results are available in the following files:</p>
        <ul>
EOF

# List all generated scan files
for file in "$SCAN_RESULTS_DIR"/*-"$DATE".*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "            <li><a href=\"$filename\">$filename</a></li>" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" << EOF
        </ul>
    </div>

    <div class="section warning">
        <h2>Recommendations</h2>
        <ul>
            <li>Review all critical and high-severity vulnerabilities immediately</li>
            <li>Update dependencies with known security issues</li>
            <li>Implement fixes for any secrets detected in the codebase</li>
            <li>Address Docker security best practice violations</li>
            <li>Configure SSL/TLS properly for production deployment</li>
            <li>Set up automated security scanning in CI/CD pipeline</li>
        </ul>
    </div>

    <div class="section info">
        <h2>Next Steps</h2>
        <ol>
            <li>Prioritize fixes based on severity and exploitability</li>
            <li>Create tickets for each security issue that needs attention</li>
            <li>Schedule regular security scans (weekly/monthly)</li>
            <li>Monitor security advisories for used dependencies</li>
            <li>Implement security testing in development workflow</li>
        </ol>
    </div>
</body>
</html>
EOF

log "Security report generated: $REPORT_FILE"

# 9. Check for critical issues and alert
CRITICAL_ISSUES=0

# Count critical issues from various scans
if [ -f "$SCAN_RESULTS_DIR/trivy-container-$DATE.json" ]; then
    TRIVY_CRITICAL=$(jq -r '.Results[]?.Vulnerabilities[]? | select(.Severity == "CRITICAL") | .VulnerabilityID' "$SCAN_RESULTS_DIR/trivy-container-$DATE.json" 2>/dev/null | wc -l || echo 0)
    CRITICAL_ISSUES=$((CRITICAL_ISSUES + TRIVY_CRITICAL))
fi

if [ -f "$SCAN_RESULTS_DIR/bandit-$DATE.json" ]; then
    BANDIT_HIGH=$(jq -r '.results[] | select(.issue_severity == "HIGH") | .test_id' "$SCAN_RESULTS_DIR/bandit-$DATE.json" 2>/dev/null | wc -l || echo 0)
    CRITICAL_ISSUES=$((CRITICAL_ISSUES + BANDIT_HIGH))
fi

log "Security scan completed. Found $CRITICAL_ISSUES critical/high-severity issues."

# Send alert if critical issues found
if [ "$CRITICAL_ISSUES" -gt 0 ] && [ -n "$SLACK_WEBHOOK_URL" ]; then
    curl -X POST -H 'Content-type: application/json' \
         --data "{\"text\":\"🚨 Security Alert: Found $CRITICAL_ISSUES critical/high-severity security issues in NFL Sentiment Analyzer. Please review the security report immediately.\"}" \
         "$SLACK_WEBHOOK_URL"
fi

# Cleanup old scan results (keep last 10)
log "Cleaning up old scan results..."
ls -t "$SCAN_RESULTS_DIR"/security-report-*.html | tail -n +11 | xargs rm -f || true
ls -t "$SCAN_RESULTS_DIR"/*-*.json | tail -n +51 | xargs rm -f || true
ls -t "$SCAN_RESULTS_DIR"/*-*.txt | tail -n +51 | xargs rm -f || true

log "Security scanning process completed successfully"
echo "Report available at: $REPORT_FILE"