# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file (if you have one)
# If you don't have a requirements.txt file, you should create one with all your dependencies
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . /app/

# Make port available if your API needs it (adjust as needed)
EXPOSE 8080

# Install supervisor to manage multiple processes
RUN apt-get update && apt-get install -y supervisor && apt-get clean

# Configure supervisor
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Run supervisord when the container launches
CMD ["/usr/bin/supervisord"]