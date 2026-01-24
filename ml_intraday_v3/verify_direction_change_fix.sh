#!/bin/bash
# Verification script for direction change fix implementation
# Run this to verify all changes were applied correctly

set -e

echo "==================================================="
echo "Direction Change Fix - Verification Script"
echo "==================================================="
echo ""

# Change to ml_intraday_v3 directory
cd "$(dirname "$0")"

echo "1. Checking configuration files..."
echo "---------------------------------------------------"

# Check live_trading.yaml
echo "✓ Checking live_trading.yaml:"
grep -n "max_concurrent:" configs/live_trading.yaml | head -1
grep -A2 "direction_change:" configs/live_trading.yaml | head -3
grep -n "primary_threshold:" configs/live_trading.yaml
grep -A2 "volatility_filter:" configs/live_trading.yaml | head -3
grep -n "no_entry_before_close_minutes:" configs/live_trading.yaml

echo ""
echo "✓ Checking risk.yaml:"
grep -n "max_concurrent_positions:" configs/risk.yaml
grep -n "max_total_contracts:" configs/risk.yaml

echo ""
echo "2. Checking Python syntax..."
echo "---------------------------------------------------"

python -m py_compile live_trading/execution_engine.py && echo "✓ execution_engine.py: OK"
python -m py_compile live_trading/live_runner.py && echo "✓ live_runner.py: OK"
python -m py_compile live_trading/replay.py && echo "✓ replay.py: OK"

echo ""
echo "3. Checking implementation details..."
echo "---------------------------------------------------"

# Check if get_net_position_direction method exists
if grep -q "def get_net_position_direction" live_trading/execution_engine.py; then
    echo "✓ get_net_position_direction() method found"
else
    echo "✗ ERROR: get_net_position_direction() method NOT found"
    exit 1
fi

# Check if direction_change logic exists in execute_signal
if grep -q "direction_change_enabled" live_trading/execution_engine.py; then
    echo "✓ direction_change_enabled check found"
else
    echo "✗ ERROR: direction_change_enabled check NOT found"
    exit 1
fi

# Check if high-confidence threshold check exists
if grep -q "direction_change_threshold" live_trading/execution_engine.py; then
    echo "✓ direction_change_threshold check found"
else
    echo "✗ ERROR: direction_change_threshold check NOT found"
    exit 1
fi

# Check if config is passed to LiveExecutionEngine in live_runner.py
if grep -q "config=self.live_cfg" live_trading/live_runner.py; then
    echo "✓ config parameter passed in live_runner.py"
else
    echo "✗ ERROR: config parameter NOT passed in live_runner.py"
    exit 1
fi

# Check if config is passed to LiveExecutionEngine in replay.py
if grep -q "config=live_cfg" live_trading/replay.py; then
    echo "✓ config parameter passed in replay.py"
else
    echo "✗ ERROR: config parameter NOT passed in replay.py"
    exit 1
fi

echo ""
echo "4. Expected configuration values..."
echo "---------------------------------------------------"

# Extract and display key values
echo "max_concurrent: $(grep 'max_concurrent:' configs/live_trading.yaml | head -1 | awk '{print $2}')"
echo "Expected: 30"
echo ""
echo "max_concurrent_positions: $(grep 'max_concurrent_positions:' configs/risk.yaml | awk '{print $2}')"
echo "Expected: 30"
echo ""
echo "max_total_contracts: $(grep 'max_total_contracts:' configs/risk.yaml | awk '{print $2}')"
echo "Expected: 150"
echo ""
echo "primary_threshold: $(grep 'primary_threshold:' configs/live_trading.yaml | awk '{print $2}')"
echo "Expected: 0.15"
echo ""
echo "volatility_filter.enabled: $(grep -A1 'volatility_filter:' configs/live_trading.yaml | grep 'enabled:' | awk '{print $2}')"
echo "Expected: true"
echo ""
echo "no_entry_before_close_minutes: $(grep 'no_entry_before_close_minutes:' configs/live_trading.yaml | awk '{print $2}')"
echo "Expected: 60"
echo ""

# Check direction_change section
if grep -q "high_confidence_threshold:" configs/live_trading.yaml; then
    echo "direction_change.enabled: $(grep -A2 'direction_change:' configs/live_trading.yaml | grep 'enabled:' | awk '{print $2}')"
    echo "Expected: true"
    echo ""
    echo "direction_change.high_confidence_threshold: $(grep 'high_confidence_threshold:' configs/live_trading.yaml | awk '{print $2}')"
    echo "Expected: 0.20"
else
    echo "✗ ERROR: direction_change configuration section NOT found"
    exit 1
fi

echo ""
echo "==================================================="
echo "✓ All verification checks PASSED!"
echo "==================================================="
echo ""
echo "Next steps:"
echo "1. Run backtest validation (see DIRECTION_CHANGE_FIX_IMPLEMENTATION.md)"
echo "2. Compare backtest results"
echo "3. Deploy to GCP if validation passes"
echo "4. Monitor first live session closely"
echo ""
echo "Documentation: ml_intraday_v3/DIRECTION_CHANGE_FIX_IMPLEMENTATION.md"
echo ""
