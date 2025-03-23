# Use an appropriate base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .


# Expose the port that will be used
EXPOSE 8080

# Run the FastAPI application directly
CMD ["python", "api.py"]
