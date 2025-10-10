# NFL Sentiment Analyzer

A real-time sentiment analysis tool for NFL games that helps fans and analysts understand the public sentiment around games, teams, and players. Built with modern MLOps practices and scalable architecture.

## 🏈 Features

- Real-time sentiment analysis of NFL-related content
- FastAPI backend with MongoDB Atlas integration
- Containerized with Docker for easy deployment
- Scalable architecture ready for high-traffic game days

## ✨ Recent Improvements

- **Enhanced Security**: Replaced dangerous `redis.flushdb()` with safe prefix-based key deletion to prevent accidental data loss in shared Redis instances
- **Performance Optimization**: Improved MongoDB query efficiency by using direct integer equality instead of `$in` operators for better index utilization
- **Bug Fixes**: Fixed CORS validator to properly read environment settings from `.env` files using Pydantic's ValidationInfo
- **CI/CD Improvements**: Resolved NumPy 2.0 compatibility issues in test suite and updated Docker builds to use UV package manager
- **Code Quality**: Applied Black formatting and improved test reliability with tolerance-based floating-point comparisons

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
- [Type Checking TODO](docs/type-checking-todo.md) - Plan for re-enabling mypy

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
