# Use Python 3.10-slim as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file first for dependency caching
COPY requirements.txt .

# Install system dependencies (for pdfminer, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . /app/

# Expose the port for your API (if needed)
EXPOSE 8080

# Install supervisor to manage multiple processes
RUN apt-get update && apt-get install -y supervisor && apt-get clean

# Copy supervisor configuration file
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Run supervisor when the container launches
CMD ["/usr/bin/supervisord"]