# Railway-optimized frontend Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Install serve globally for serving static files
RUN npm install -g serve

# Expose port (Railway sets this dynamically)
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:${PORT:-3000} || exit 1

# Start the application (run as root for Railway compatibility)
CMD ["sh", "-c", "serve -s dist -l ${PORT:-3000}"]