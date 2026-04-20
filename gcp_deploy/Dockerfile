FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    LIVE_CONFIG_DIR=/app/ml_intraday_v3/configs/live_dual_meta_mes_real \
    LIVE_MODEL_BUNDLE=/app/ml_intraday_v3/models/live/dual_meta_mes_live_bundle.pkl

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY gcp_deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
# We copy specific directories to keep the image clean
COPY core/ /app/core/
COPY ml_intraday_v3/models/live/ /app/ml_intraday_v3/models/live/
COPY ml_intraday_v3/ /app/ml_intraday_v3/

# Create logs directory
RUN mkdir -p logs

# Set the entrypoint to the live runner
# User can override CMD to run other scripts
ENTRYPOINT ["python", "ml_intraday_v3/live_trading/live_runner.py"]

# Default arguments (can be overridden)
CMD ["--config-dir", "/app/ml_intraday_v3/configs/live_dual_meta_mes_real", "--model-bundle", "/app/ml_intraday_v3/models/live/dual_meta_mes_live_bundle.pkl", "--no-confirm", "--log-level", "INFO"]
