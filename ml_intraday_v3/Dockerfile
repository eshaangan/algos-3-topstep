# Dockerfile for ml_intraday_v3 live trading system
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-mlv3.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-mlv3.txt

# Copy the entire ml_intraday_v3 directory
COPY . .

# Explicitly copy core directory (required for projectx_client)
COPY core /app/core

# Explicitly copy the validated model from root to correct location
COPY model_bundle_retrained_oct2024_nov2025.pkl /app/ml_intraday_v3/models/saved/model_bundle_retrained_oct2024_nov2025.pkl

# Set Python path
ENV PYTHONPATH=/app:/app/ml_intraday_v3

# Create logs directory
RUN mkdir -p /app/logs

# Set entry point to live trading runner with validated model
CMD ["python", "live_trading/live_runner.py", \
     "--model-bundle", "/app/ml_intraday_v3/models/saved/model_bundle_retrained_oct2024_nov2025.pkl", \
     "--no-confirm"]
