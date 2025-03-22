# Use an appropriate base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy all files to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy and run the startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Run the script
CMD ["/bin/sh", "/start.sh"]
