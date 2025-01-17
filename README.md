# NFL Sentiment Analysis MVP

This is a MVP (Minimum Viable Product) version of the NFL Sentiment Analysis project. It provides real-time sentiment analysis for NFL-related content using machine learning.

## Features

- FastAPI backend with sentiment analysis endpoint
- MongoDB integration for data persistence
- Docker and Docker Compose setup for easy deployment
- Basic sentiment analysis using pre-trained transformer models

## Getting Started

### Prerequisites

- Docker and Docker Compose installed
- Git (for version control)

### Running Locally

1. Clone the repository
2. Navigate to the project directory
3. Build and run the containers:

```bash
docker-compose up --build
```

The API will be available at http://localhost:8000

### API Endpoints

- `GET /`: Health check endpoint
- `POST /analyze`: Analyze sentiment of provided text
- `GET /recent`: Get recent sentiment analyses

### Example Usage

```bash
# Analyze sentiment
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "The Chiefs are playing amazing football today!"}'

# Get recent analyses
curl "http://localhost:8000/recent"
```

## Deployment

This MVP is designed to be deployed to Digital Ocean. Detailed deployment instructions will be provided separately.

## Next Steps

- Add authentication
- Implement real-time data collection from Twitter/X
- Add more sophisticated sentiment analysis
- Create frontend dashboard
- Add automated testing
- Set up CI/CD pipeline
