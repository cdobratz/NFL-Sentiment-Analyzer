# syntax = docker/dockerfile:1.7
# Railway-optimized frontend Dockerfile
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY . .

# Make startup script executable
RUN chmod +x start.sh

# Build the application
RUN npm run build

# Install serve globally for serving static files
RUN npm install -g serve

# Expose port 3000 (Railway will handle port mapping)
EXPOSE 3000

# Start the application using startup script
CMD ["/app/start.sh"]