#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t mommycare-llm:latest .

# Run the container
echo "Starting container..."
docker run -d --name mommycare-api \
  -p 8080:8080 \
  --env-file .env \
  mommycare-llm:latest

echo "Container is running at http://localhost:8080"
echo "You can check logs with: docker logs mommycare-api"