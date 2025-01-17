# NFL Sentiment Analyzer

A real-time sentiment analysis tool for NFL games that helps fans and analysts understand the public sentiment around games, teams, and players. Built with modern MLOps practices and scalable architecture.

## 🏈 Features

- Real-time sentiment analysis of NFL-related content
- FastAPI backend with MongoDB Atlas integration
- Containerized with Docker for easy deployment
- Scalable architecture ready for high-traffic game days

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/cdobratz/NFL-Sentiment-Analyzer.git
cd NFL-Sentiment-Analyzer
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your MongoDB Atlas credentials
```

3. Run with Docker:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

## 🔧 Tech Stack

- **Backend**: Python, FastAPI
- **Database**: MongoDB Atlas
- **ML**: HuggingFace Transformers
- **Infrastructure**: Docker, GitHub Actions
- **Future**: React Frontend, Twitter API Integration

## 📚 API Documentation

Once running, visit `http://localhost:8000/docs` for interactive API documentation.

Example endpoints:
- `POST /analyze`: Analyze sentiment of NFL-related text
- `GET /recent`: Get recent sentiment analyses

## 🛣️ Roadmap

- [ ] User authentication
- [ ] Real-time Twitter/X data integration
- [ ] Advanced sentiment analysis models
- [ ] Interactive dashboard
- [ ] Automated model retraining
- [ ] Deployment to Digital Ocean

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
