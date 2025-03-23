# Use an appropriate base image
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy all necessary files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the start.sh script has execute permissions
RUN chmod +x /start.sh

# Expose the correct port for Cloud Run
EXPOSE 8080

# Run the startup script
CMD ["/bin/sh", "/start.sh"]
