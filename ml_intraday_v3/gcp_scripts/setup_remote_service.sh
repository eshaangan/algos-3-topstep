#!/bin/bash
set -e

# Configuration
ZONE="us-central1-a"
INSTANCE_NAME="algotrader"

echo "=== Setting up Systemd Service on $INSTANCE_NAME ==="

# Create service file locally
cat > algotrader.service <<EOF
[Unit]
Description=ML Intraday V3 Live Trading System
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=/home/$(whoami)/algos/ml_intraday_v3
Environment="PATH=/home/$(whoami)/algos/ml_intraday_v3/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/$(whoami)/algos/ml_intraday_v3/venv/bin/python live_trading/live_runner.py --config-name live_trading_topstep_50k_accel.yaml --model-bundle models/saved/model_bundle_topstep_candidate.pkl --no-confirm
Restart=always
RestartSec=10
StandardOutput=append:/home/$(whoami)/algos/logs/systemd.log
StandardError=append:/home/$(whoami)/algos/logs/systemd_error.log

# Resource limits
MemoryMax=3G
CPUQuota=180%

[Install]
WantedBy=multi-user.target
EOF

# Note: The User=$(whoami) above will run as the local user name, which might not match the remote user.
# Gcloud SSH usually uses the local username or google_sudoers.
# Better to fetch the remote username first.

echo "Detecting remote username..."
REMOTE_USER=$(gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "whoami" | tr -d '\r')
echo "Remote user: $REMOTE_USER"

# Regenerate service file with correct user
cat > algotrader.service <<EOF
[Unit]
Description=ML Intraday V3 Live Trading System
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$REMOTE_USER
WorkingDirectory=/home/$REMOTE_USER/algos/ml_intraday_v3
Environment="PATH=/home/$REMOTE_USER/algos/ml_intraday_v3/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/$REMOTE_USER/algos/ml_intraday_v3/venv/bin/python live_trading/live_runner.py --config-name live_trading_topstep_50k_accel.yaml --model-bundle models/saved/model_bundle_topstep_candidate.pkl --no-confirm
Restart=always
RestartSec=10
StandardOutput=append:/home/$REMOTE_USER/algos/logs/systemd.log
StandardError=append:/home/$REMOTE_USER/algos/logs/systemd_error.log

# Resource limits
MemoryMax=3G
CPUQuota=180%

[Install]
WantedBy=multi-user.target
EOF

# Upload
echo "Uploading service file..."
gcloud compute scp algotrader.service $INSTANCE_NAME:~/ --zone=$ZONE

# Install
echo "Installing service..."
gcloud compute ssh $INSTANCE_NAME --zone=$ZONE --command "
    mkdir -p ~/algos/logs
    sudo mv ~/algotrader.service /etc/systemd/system/algotrader.service
    sudo systemctl daemon-reload
    sudo systemctl enable algotrader
    sudo systemctl start algotrader
    sudo systemctl status algotrader --no-pager
"

# Cleanup
rm algotrader.service

echo "=== Service Setup Complete ==="
