# Use the official Python slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements file
COPY requirements.txt .

# Install dependencies with an increased timeout
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Run both FastAPI and Telegram bot using a script
CMD ["bash", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & python telegram_bot.py"]
