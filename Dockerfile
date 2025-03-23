FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libportaudio2 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api.py .
COPY LLMmain.py .
COPY trans.py .

# Create a directory for temporary audio files
RUN mkdir -p /tmp/audio

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD ["python", "api.py"]