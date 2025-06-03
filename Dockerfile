# Use official Python 3.10 slim base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including tzdata for timezone support
RUN apt-get update && apt-get install -y \
    gcc \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set timezone environment and configure timezone data
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install supervisord
RUN pip install supervisor

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for database and logs
RUN mkdir -p /app/database /app/logs

# Copy supervisord configuration
COPY supervisord.conf /etc/supervisord.conf

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Run supervisord
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
