# ML Scalper Live Runner — microstructure LightGBM model on MNQ ticks via TopstepX SignalR
# Default CMD is dry-run. Pass --live --yes only for real orders.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/ml_intraday_v3 \
    # Practice account defaults — override via --env-file on docker run
    ML_N_CONTRACTS=15

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY gcp_deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/ /app/core/
COPY ml_intraday_v3/ /app/ml_intraday_v3/

RUN mkdir -p /app/logs

ENTRYPOINT ["python", "ml_intraday_v3/live/ml_live_runner.py"]
CMD ["--dry-run"]
