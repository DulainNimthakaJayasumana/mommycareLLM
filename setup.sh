#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "No .env file found. Creating from example..."
    cp .env.example .env
    echo "Please edit the .env file with your API keys before proceeding."
    exit 1
fi

# Create books directory if it doesn't exist
mkdir -p books

# Build and start the services
echo "Building and starting services..."
docker-compose up --build -d

# Check service status
echo "Checking service status..."
docker-compose ps

echo "Services started successfully!"
echo "FastAPI service is available at: http://localhost:8000/docs"
echo "Telegram bot should be running and responding to messages."
echo ""
echo "To view logs:"
echo "  - Telegram bot: docker-compose logs -f telegram-bot"
echo "  - API service: docker-compose logs -f fastapi-service"
echo ""
echo "To index PDFs, place them in the books/ directory and run:"
echo "  docker-compose exec telegram-bot python agenticChunking.py"