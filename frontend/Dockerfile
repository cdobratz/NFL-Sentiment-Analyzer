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

# Make startup script executable
RUN chmod +x start.sh

# Expose port 3000 (Railway will handle port mapping)
EXPOSE 3000

# Health check - Railway handles this externally
# No HEALTHCHECK needed as Railway uses external health checks

# Start the application using startup script
CMD ["./start.sh"]