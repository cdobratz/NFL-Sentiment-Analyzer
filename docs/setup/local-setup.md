# Local Development Setup

This guide will help you set up the NFL Sentiment Analyzer project for local development.

## Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose
- Git
- MongoDB Atlas account

## Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/cdobratz/NFL-Sentiment-Analyzer.git
   cd NFL-Sentiment-Analyzer
   ```

2. **Set Up Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB Atlas credentials
   ```

4. **Run with Docker**
   ```bash
   docker-compose up --build
   ```

5. **Access the Application**
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## Development Workflow

1. Create a feature branch from dev:
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test locally

3. Push changes and create a PR to the dev branch

## Common Issues and Solutions

### MongoDB Connection Issues
- Ensure your IP address is whitelisted in MongoDB Atlas
- Verify connection string format in .env file
- Check network connectivity

### Docker Issues
- Ensure ports 8000 is not in use
- Try removing old containers and images:
  ```bash
  docker-compose down
  docker system prune
  ```
