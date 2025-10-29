# Cloud Deployment Guide

This guide covers deploying the NFL Sentiment Analyzer to Render or Railway for simple cloud hosting.

## Prerequisites

1. **GitHub Repository**: Your code should be pushed to a GitHub repository
2. **Environment Variables**: Copy `.env.cloud.template` to `.env.production` and fill in values
3. **Database**: You'll need MongoDB and Redis instances (can be managed services)

## Option 1: Deploy to Render

### Step 1: Set up Render Account
1. Go to [render.com](https://render.com) and sign up
2. Connect your GitHub account

### Step 2: Create Database Services
1. **Create MongoDB Service**:
   - Click "New" → "Private Service"
   - Connect your repository
   - Set Name: `nfl-sentiment-mongodb`
   - Docker Image: `mongo:7`
   - Set disk storage (10GB recommended)
   - Add environment variable: `MONGO_INITDB_ROOT_PASSWORD`

2. **Create Redis Service**:
   - Click "New" → "Redis"
   - Set Name: `nfl-sentiment-redis`
   - Choose plan (Starter recommended)

### Step 3: Deploy Backend API
1. Click "New" → "Web Service"
2. Connect your repository
3. Use these settings:
   - **Name**: `nfl-sentiment-api`
   - **Runtime**: Docker
   - **Dockerfile Path**: `./Dockerfile.cloud`
   - **Region**: Oregon (or closest to your users)
   - **Plan**: Starter

4. **Environment Variables**:
   ```
   ENVIRONMENT=production
   LOG_LEVEL=info
   WORKERS=2
   PORT=8000
   JWT_SECRET_KEY=your_secure_jwt_secret
   MONGODB_URL=[Connection string from MongoDB service]
   REDIS_URL=[Connection string from Redis service]
   CORS_ORIGINS=https://nfl-sentiment-frontend.onrender.com
   ```

### Step 4: Deploy Frontend
1. Click "New" → "Static Site"
2. Connect your repository
3. Use these settings:
   - **Name**: `nfl-sentiment-frontend`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm ci && npm run build`
   - **Publish Directory**: `dist`

4. **Environment Variables**:
   ```
   NODE_ENV=production
   VITE_API_URL=https://nfl-sentiment-api.onrender.com
   ```

### Step 5: Configure Custom Domains (Optional)
- Set up custom domains in Render dashboard
- Update CORS_ORIGINS in backend service

## Option 2: Deploy to Railway

### Step 1: Set up Railway Account
1. Go to [railway.app](https://railway.app) and sign up
2. Connect your GitHub account

### Step 2: Create New Project
1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your repository

### Step 3: Add Database Services
1. **Add MongoDB**:
   - Click "New" → "Database" → "MongoDB"
   - Note the connection string from environment variables

2. **Add Redis**:
   - Click "New" → "Database" → "Redis"
   - Note the connection string from environment variables

### Step 4: Configure Backend Service
1. Railway should auto-detect your app
2. Set environment variables in Railway dashboard:
   ```
   ENVIRONMENT=production
   LOG_LEVEL=info
   WORKERS=2
   JWT_SECRET_KEY=your_secure_jwt_secret
   MONGODB_URL=${{MongoDB.DATABASE_URL}}
   REDIS_URL=${{Redis.REDIS_URL}}
   CORS_ORIGINS=https://your-frontend-domain.up.railway.app
   ```

3. **Override Build Settings** (if needed):
   - Build Command: Uses Dockerfile automatically
   - Start Command: Set in `railway.json`

### Step 5: Deploy Frontend
1. Add new service to same project
2. Set root directory to `frontend`
3. Configure build:
   - Build Command: `npm ci && npm run build`
   - Start Command: `npm run preview -- --host 0.0.0.0 --port $PORT`
4. Set environment variables:
   ```
   NODE_ENV=production
   VITE_API_URL=https://your-backend-domain.up.railway.app
   ```

## Database Setup

### MongoDB Collections
Your app will automatically create necessary collections, but you may want to create indexes:

```javascript
// Connect to your MongoDB instance and run:
use nfl_sentiment;

// Create indexes for better performance
db.sentiment_analyses.createIndex({ "created_at": -1 });
db.sentiment_analyses.createIndex({ "team_id": 1, "created_at": -1 });
db.sentiment_analyses.createIndex({ "source": 1, "created_at": -1 });
db.teams.createIndex({ "name": 1 });
db.teams.createIndex({ "abbreviation": 1 });
```

## File Upload Configuration

Your app now supports JSON file uploads for X.com data. The endpoint `/sentiment/upload` accepts:
- JSON files with X.com/Twitter data
- Maximum file size: 50MB
- Automatic parsing and sentiment analysis

## Testing Deployment

1. **Health Check**: Visit `https://your-api-domain/health`
2. **API Documentation**: Visit `https://your-api-domain/docs`
3. **Upload Test**: Try uploading a JSON file via the frontend
4. **Sentiment Analysis**: Test the `/sentiment/analyze` endpoint

## Monitoring

Both platforms provide:
- **Logs**: Real-time application logs
- **Metrics**: CPU, memory, and network usage
- **Alerts**: Set up alerts for service failures

## Troubleshooting

### Common Issues:

1. **Build Failures**:
   - Check Dockerfile syntax
   - Verify all dependencies are in pyproject.toml
   - Check build logs for specific errors

2. **Database Connection Issues**:
   - Verify connection strings are correct
   - Check network permissions
   - Ensure database services are running

3. **CORS Errors**:
   - Update CORS_ORIGINS environment variable
   - Include all frontend domains

4. **File Upload Issues**:
   - Check MAX_UPLOAD_SIZE environment variable
   - Verify file format matches expected JSON structure

### Platform-Specific Issues:

**Render**:
- Services may sleep after inactivity (upgrade to paid plan to prevent)
- Check service logs in Render dashboard

**Railway**:
- Monitor usage limits on free plan
- Check project logs and metrics

## Security Considerations

1. **Environment Variables**: Never commit secrets to Git
2. **HTTPS**: Both platforms provide SSL/TLS by default
3. **Database Access**: Use strong passwords and connection strings
4. **Rate Limiting**: API includes built-in rate limiting
5. **File Validation**: Uploaded files are validated before processing

## Scaling

**Render**:
- Upgrade to higher tier plans for more resources
- Use multiple instances for high availability

**Railway**:
- Scale vertically by upgrading resources
- Monitor usage and upgrade plan as needed

Both platforms support:
- Automatic deployments on Git push
- Zero-downtime deployments
- Easy rollbacks to previous versions