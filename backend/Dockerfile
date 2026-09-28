# Base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install system dependencies if required for C-extensions/raster processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies list and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source and model artifacts
COPY . .

# Expose default Cloud Run container port
EXPOSE 8080

# Run FastAPI via Uvicorn in shell-form to allow runtime PORT expansion on Cloud Run
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
