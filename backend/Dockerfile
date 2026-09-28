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
    libexpat1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies list and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source and model artifacts
COPY . .

# Download full model artifact from GitHub Release to replace Git LFS pointer
RUN curl -fL --retry 3 -o ml/models/lulc_rf_model.joblib \
    https://github.com/sarthak-panda-18/LandUseLandCover/releases/download/v1.2-model/lulc_rf_model_v2.joblib \
    && test "$(stat -c%s ml/models/lulc_rf_model.joblib)" -gt 10000000

# Expose default Cloud Run container port
EXPOSE 8080

# Run FastAPI via Uvicorn in shell-form to allow runtime PORT expansion on Cloud Run
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
