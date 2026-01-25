# Base image with Python 3.13 (matches local venv)
FROM python:3.13-slim

# Set workdir
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	curl \
	&& rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt ./

# Install Python deps (pin scikit-learn for pickle compatibility)
RUN pip install --no-cache-dir --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt \
	&& pip install --no-cache-dir scikit-learn==1.6.1 evidently

# Copy project
COPY . .

# Environment: point to service account, configurable at runtime
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/config/service-account-key.json \
	PYTHONUNBUFFERED=1

# Default command runs batch prediction (can override with `docker run ...`)
CMD ["python", "batch_prediction.py"]

