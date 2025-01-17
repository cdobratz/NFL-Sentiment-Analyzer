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

## 📚 Documentation

Detailed documentation is available in the [docs](docs) directory:

- [Local Development Setup](docs/setup/local-setup.md)
- [Production Deployment Guide](docs/setup/deployment.md)
- [Architecture Overview](docs/architecture/tech-stack.md)
- [Contributing Guidelines](CONTRIBUTING.md)

Visit our [Documentation Index](docs/README.md) for a complete list of documentation.

## 🔧 Tech Stack

- **Backend**: Python, FastAPI
- **Database**: MongoDB Atlas
- **ML**: HuggingFace Transformers
- **Infrastructure**: Docker, GitHub Actions

## 🛣️ Roadmap

- [ ] User authentication
- [ ] Real-time Twitter/X data integration
- [ ] Advanced sentiment analysis models
- [ ] Interactive dashboard
- [ ] Automated model retraining
- [ ] Deployment to Digital Ocean

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
