FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port 8000
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:api", "--host", "0.0.0.0", "--port", "8000"]
