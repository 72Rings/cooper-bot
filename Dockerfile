# Use an official lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy server files (including server.py and requirements.txt)
COPY server/ ./

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose port Flask runs on
EXPOSE 5000

# Start the Flask server
CMD ["python", "server.py"]
