# Optimized Dockerfile for ASCENDS ML API
# Use Python slim image instead of full Ubuntu + Anaconda
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages with pip (much faster than conda)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 7777

# Run the FastAPI server
CMD ["uvicorn", "ascends_server:app", "--host", "0.0.0.0", "--port", "7777"]
