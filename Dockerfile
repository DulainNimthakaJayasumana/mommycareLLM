# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first (to leverage caching)
COPY requirements.txt .

# Upgrade pip before installing dependencies
RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the FastAPI port
EXPOSE 8080

# Set the default command to run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
