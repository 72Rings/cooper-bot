# Use an official lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy your backend server files
COPY server/ ./

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port used by Railway (typically 8080)
EXPOSE 8080

# Start the Flask server on port 8080
ENV PORT 8080
CMD ["python", "server.py"]
