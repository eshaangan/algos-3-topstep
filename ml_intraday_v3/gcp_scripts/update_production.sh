#!/bin/bash
set -e

# Run from project root
# ./ml_intraday_v3/gcp_scripts/update_production.sh

echo "=== STARTING PRODUCTION UPDATE ==="

# 1. Update Code
./ml_intraday_v3/gcp_scripts/deploy_code.sh

# 2. Update Models (Optional - ask user or just do it? Doing it ensures consistency)
# If models are large, might want to make this optional. 
# For now, I'll include it as "easy update" implies everything.
./ml_intraday_v3/gcp_scripts/deploy_models.sh

echo "=== PRODUCTION UPDATE COMPLETE ==="
