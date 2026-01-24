#!/bin/bash
#
# Deploy Multi-Market Trading System to GCP
# Supports MES (US) + NKD (Asian) sessions
#

set -e

# Configuration
ZONE="us-central1-a"
INSTANCE_NAME="algotrader"
PROJECT_DIR="ml_intraday_v3"

echo "=================================================="
echo "MULTI-MARKET GCP DEPLOYMENT"
echo "Markets: MES (US) + NKD (Asia)"
echo "=================================================="
echo ""

# Step 1: Download free historical data
echo "Step 1: Downloading free historical data..."
if [ ! -f "$PROJECT_DIR/data/download_free_futures.py" ]; then
    echo "❌ Error: download_free_futures.py not found"
    exit 1
fi

echo "  Running download script locally..."
cd $PROJECT_DIR
python data/download_free_futures.py \
    --start 2020-01-01 \
    --end 2025-12-31 \
    --output-dir data/raw_futures

if [ $? -ne 0 ]; then
    echo "❌ Data download failed"
    exit 1
fi
echo "✓ Historical data downloaded"
cd ..
echo ""

# Step 2: Train models locally (if not already trained)
echo "Step 2: Checking trained models..."

MES_MODEL="$PROJECT_DIR/runs/mes_production/bar_size=5m/model_bundle.pkl"
NKD_MODEL="$PROJECT_DIR/runs/nkd_production/bar_size=5m/model_bundle.pkl"

if [ ! -f "$MES_MODEL" ]; then
    echo "  Training MES model..."
    python -m ml_intraday_v3.cli build-train --config configs/training_mes.yaml
fi

if [ ! -f "$NKD_MODEL" ]; then
    echo "  Training NKD model..."
    python -m ml_intraday_v3.cli build-train --config configs/training_nkd.yaml
fi

echo "✓ Models ready"
echo ""

# Step 3: Package multi-market models
echo "Step 3: Packaging multi-market models..."
tar -czf multimarket_bundle.tar.gz \
    -C $PROJECT_DIR \
    runs/mes_production \
    runs/nkd_production \
    configs/live_trading_multimarket.yaml \
    data/raw_futures

echo "✓ Models packaged ($(du -h multimarket_bundle.tar.gz | cut -f1))"
echo ""

# Step 4: Upload to GCP
echo "Step 4: Uploading to GCP instance..."
gcloud compute scp multimarket_bundle.tar.gz $INSTANCE_NAME:~/ \
    --zone=$ZONE

if [ $? -ne 0 ]; then
    echo "❌ Upload failed. Is instance running?"
    echo "   Check with: gcloud compute instances list"
    exit 1
fi
echo "✓ Bundle uploaded"
echo ""

# Step 5: Remote extraction and setup
echo "Step 5: Setting up on remote instance..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "
    set -e
    echo '  Extracting bundle...'
    mkdir -p ~/algos/$PROJECT_DIR
    tar -xzf multimarket_bundle.tar.gz -C ~/algos/$PROJECT_DIR/

    echo '  Installing yfinance (for future data updates)...'
    cd ~/algos/$PROJECT_DIR
    source venv/bin/activate
    pip install yfinance >/dev/null 2>&1 || true

    echo '  Verifying models...'
    ls -lh ~/algos/$PROJECT_DIR/runs/*/bar_size=5m/model_bundle.pkl

    echo '  Creating state directory...'
    mkdir -p ~/algos/$PROJECT_DIR/state

    echo '✓ Remote setup complete'
"

if [ $? -ne 0 ]; then
    echo "❌ Remote setup failed"
    exit 1
fi
echo ""

# Step 6: Update systemd service for multi-market
echo "Step 6: Updating systemd service..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "
    set -e

    # Create multi-market service file
    sudo tee /etc/systemd/system/algotrader-multimarket.service > /dev/null <<'EOF'
[Unit]
Description=ML Intraday V3 Multi-Market Trading System
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=trader
WorkingDirectory=/home/trader/algos/$PROJECT_DIR
Environment=\"PATH=/home/trader/algos/$PROJECT_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin\"
ExecStart=/home/trader/algos/$PROJECT_DIR/venv/bin/python live_trading/live_runner_multimarket.py \
    --config configs/live_trading_multimarket.yaml
Restart=always
RestartSec=10
StandardOutput=append:/home/trader/algos/logs/multimarket.log
StandardError=append:/home/trader/algos/logs/multimarket_error.log

# Resource limits
MemoryMax=3G
CPUQuota=180%

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    echo '✓ Service file created'
"
echo ""

# Step 7: Verification checklist
echo "Step 7: Verification..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "
    cd ~/algos/$PROJECT_DIR
    source venv/bin/activate

    echo '  Checking Python dependencies...'
    python -c 'import pandas, numpy, sklearn, lightgbm, databento, yfinance; print(\"✓ Dependencies OK\")'

    echo '  Checking historical data...'
    python -c '
import pandas as pd
from pathlib import Path

mes_files = list(Path(\"data/raw_futures\").glob(\"mes_daily_*.parquet\"))
nkd_files = list(Path(\"data/raw_futures\").glob(\"nkd_daily_*.parquet\"))

if mes_files:
    mes = pd.read_parquet(mes_files[0])
    print(f\"  ✓ MES: {len(mes)} days\")
if nkd_files:
    nkd = pd.read_parquet(nkd_files[0])
    print(f\"  ✓ NKD: {len(nkd)} days\")
'
    echo '  Checking models...'
    test -f runs/mes_production/bar_size=5m/model_bundle.pkl && echo '  ✓ MES model exists'
    test -f runs/nkd_production/bar_size=5m/model_bundle.pkl && echo '  ✓ NKD model exists'

    echo '  Checking config...'
    test -f configs/live_trading_multimarket.yaml && echo '  ✓ Multi-market config exists'
"
echo ""

# Cleanup
rm multimarket_bundle.tar.gz

echo "=================================================="
echo "✓ MULTI-MARKET DEPLOYMENT COMPLETE!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. SSH into instance: gcloud compute ssh $INSTANCE_NAME --zone=$ZONE"
echo "2. Test multi-market runner in paper mode:"
echo "   cd ~/algos/$PROJECT_DIR"
echo "   source venv/bin/activate"
echo "   python live_trading/live_runner_multimarket.py --config configs/live_trading_multimarket.yaml"
echo ""
echo "3. If test passes, enable systemd service:"
echo "   sudo systemctl enable algotrader-multimarket"
echo "   sudo systemctl start algotrader-multimarket"
echo "   sudo systemctl status algotrader-multimarket"
echo ""
echo "4. Monitor logs:"
echo "   tail -f ~/algos/logs/multimarket.log"
echo ""
echo "Trading hours (CT):"
echo "  NKD (Asia):  18:00-03:00"
echo "  MES (US):    08:30-15:00"
echo "  Total:       ~16 hours/day"
echo ""
