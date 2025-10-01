# NFL Sentiment Analyzer - Production Deployment Guide

This guide provides comprehensive instructions for deploying the NFL Sentiment Analyzer to production environments.

## Prerequisites

- Docker Engine 20.10+ with Docker Compose
- Docker Swarm initialized (for production deployment)
- Minimum 4GB RAM, 2 CPU cores, 20GB disk space
- SSL certificates (for HTTPS)
- Domain name configured (optional but recommended)

## Quick Start

1. **Clone the repository and navigate to the project directory**
2. **Copy and configure environment variables:**
   ```bash
   cp .env.example .env.production
   # Edit .env.production with your configuration
   ```
3. **Deploy to production:**
   ```bash
   ./scripts/deploy.sh -e production
   ```

## Detailed Deployment Process

### 1. Environment Configuration

Create and configure your production environment file:

```bash
cp .env.example .env.production
```

Required environment variables:
- `MONGO_ROOT_PASSWORD`: Strong password for MongoDB
- `REDIS_PASSWORD`: Password for Redis cache
- `JWT_SECRET_KEY`: Secret key for JWT tokens (generate with `openssl rand -hex 32`)
- External API keys for data sources (Twitter, ESPN, etc.)

### 2. SSL Certificate Setup

For HTTPS support, place your SSL certificates in the `ssl/` directory:
```bash
mkdir -p ssl/
# Copy your certificates
cp /path/to/your/cert.pem ssl/
cp /path/to/your/key.pem ssl/
```

Or use Let's Encrypt:
```bash
# Install certbot
sudo apt-get install certbot

# Generate certificates
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/cert.pem
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/key.pem
```

### 3. Initialize Docker Swarm

```bash
docker swarm init
```

### 4. Setup Secrets

```bash
./scripts/setup-secrets.sh
```

### 5. Deploy the Application

```bash
./scripts/deploy.sh -e production -s nfl-sentiment
```

### 6. Verify Deployment

Check service status:
```bash
docker service ls
docker service logs nfl-sentiment_api
```

Test endpoints:
```bash
curl http://localhost:8000/health
curl http://localhost:3000/health
```

### 7. Setup Automated Tasks

Configure backups and monitoring:
```bash
./scripts/setup-cron.sh
```

## Service Architecture

The production deployment includes the following services:

### Core Services
- **Frontend**: React application (port 3000)
- **API**: FastAPI backend (port 8000)
- **MongoDB**: Database (port 27017)
- **Redis**: Cache (port 6379)
- **Nginx**: Reverse proxy and load balancer (ports 80, 443)

### Monitoring Stack
- **Prometheus**: Metrics collection (port 9090)
- **Grafana**: Dashboards and visualization (port 3001)
- **Loki**: Log aggregation (port 3100)
- **Promtail**: Log collection agent

## Monitoring and Observability

### Access Monitoring Dashboards

- **Grafana**: http://localhost:3001 (admin/admin - change on first login)
- **Prometheus**: http://localhost:9090
- **API Documentation**: http://localhost:8000/docs

### Log Management

Logs are centralized and available through:
- **Application logs**: `/app/logs/`
- **Nginx logs**: `/var/log/nginx/`
- **Grafana Loki**: Centralized log aggregation

### Alerts and Notifications

Configure Slack notifications by setting `SLACK_WEBHOOK_URL` in your environment file.

## Backup and Recovery

### Automated Backups

Backups run daily at 2 AM and include:
- MongoDB database dump
- Application logs
- Configuration files

### Manual Backup

```bash
./scripts/backup.sh
```

### Restore from Backup

```bash
# List available backups
./scripts/restore.sh -l

# Restore from specific backup
./scripts/restore.sh -d 20241201_020000

# Restore latest backup
./scripts/restore.sh
```

## Security

### Security Scanning

Automated security scans run weekly:
```bash
./scripts/security-scan.sh
```

### Security Features

- JWT-based authentication
- Rate limiting on API endpoints
- HTTPS encryption (when SSL configured)
- Container security scanning
- Dependency vulnerability scanning
- Secrets management with Docker secrets

## Scaling

### Horizontal Scaling

Scale individual services:
```bash
docker service scale nfl-sentiment_api=3
docker service scale nfl-sentiment_frontend=2
```

### Resource Limits

Services are configured with resource limits in `docker-compose.prod.yml`:
- API: 1GB RAM, 1 CPU
- Frontend: 512MB RAM, 0.5 CPU
- Database: 2GB RAM, 2 CPU

## Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   docker service logs nfl-sentiment_api
   docker service ps nfl-sentiment_api
   ```

2. **Database connection issues**
   ```bash
   docker exec -it $(docker ps -q -f name=nfl-sentiment_mongodb) mongosh
   ```

3. **SSL certificate issues**
   - Verify certificate files exist and have correct permissions
   - Check certificate expiration dates
   - Ensure domain name matches certificate

### Health Checks

All services include health checks:
```bash
# Check all service health
docker service ls

# Detailed health status
docker service ps nfl-sentiment_api --no-trunc
```

### Performance Monitoring

Monitor system resources:
```bash
# Container resource usage
docker stats

# System resources
htop
df -h
```

## Maintenance

### Regular Tasks

1. **Weekly**: Review security scan results
2. **Monthly**: Update dependencies and rebuild images
3. **Quarterly**: Review and rotate secrets
4. **As needed**: Scale services based on load

### Updates and Deployments

1. **Test in staging environment first**
2. **Create backup before deployment**
3. **Use blue-green deployment for zero downtime**
4. **Monitor health checks after deployment**

### Log Rotation

Logs are automatically rotated:
- Application logs: Daily, keep 30 days
- System logs: Weekly, keep 12 weeks
- Backup logs: Monthly, keep 12 months

## Support and Monitoring

### Key Metrics to Monitor

- API response times and error rates
- Database connection pool usage
- Memory and CPU utilization
- Disk space usage
- SSL certificate expiration

### Alerting Thresholds

- API error rate > 5%
- Response time > 2 seconds
- Memory usage > 85%
- Disk usage > 85%
- SSL certificate expires in < 30 days

## Environment-Specific Configurations

### Staging Environment

```bash
./scripts/deploy.sh -e staging
```

### Development Environment

```bash
docker-compose up -d
```

## Additional Resources

- [Docker Swarm Documentation](https://docs.docker.com/engine/swarm/)
- [Prometheus Monitoring](https://prometheus.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)
- [MongoDB Production Deployment](https://docs.mongodb.com/manual/administration/production-notes/)

For support, check the logs first, then consult the troubleshooting section above.