# Railway-optimized Dockerfile for backend API
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files and README (required by build backend)
COPY pyproject.toml ./
COPY uv.lock ./
COPY README.md ./

# Install dependencies using uv with optimizations for Railway
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cache/pip \
    uv sync --frozen --no-dev

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY start.sh ./start.sh

# Create logs directory
RUN mkdir -p /app/logs

# Make startup script executable and set permissions
RUN chmod +x /app/start.sh

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port 8000 (Railway will handle port mapping)
EXPOSE 8000

# Health check - Railway handles this via healthcheckPath in railway.json
# No HEALTHCHECK needed as Railway uses external health checks

# Use our custom startup script
CMD ["./start.sh"]