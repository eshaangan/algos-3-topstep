#!/bin/bash
# Monitor optimized HMM progress

PID=89578
LOG_FILE="/tmp/hmm_optimized_build_weights.log"

echo "==================================================================="
echo "OPTIMIZED HMM PROGRESS MONITOR"
echo "==================================================================="
echo "PID: $PID"
echo "Started: $(date)"
echo ""

# Check if process is still running
if ps -p $PID > /dev/null 2>&1; then
    echo "✓ Process is RUNNING"
else
    echo "✗ Process is NOT running"
    exit 1
fi

echo ""
echo "-------------------------------------------------------------------"
echo "Latest log output:"
echo "-------------------------------------------------------------------"
tail -30 "$LOG_FILE"

echo ""
echo "-------------------------------------------------------------------"
echo "Progress indicators:"
echo "-------------------------------------------------------------------"

# Count progress markers
progress_count=$(grep -c "Progress:" "$LOG_FILE" 2>/dev/null || echo "0")
echo "Progress updates: $progress_count"

# Show estimated completion
if [ -f "$LOG_FILE" ]; then
    # Extract bar count from progress lines
    last_bar=$(grep "Progress:" "$LOG_FILE" | tail -1 | sed -n 's/.*bar \([0-9]*\).*/\1/p')
    if [ ! -z "$last_bar" ]; then
        total_bars=1037010
        pct=$((last_bar * 100 / total_bars))
        echo "Processed: $last_bar / $total_bars bars ($pct%)"

        # Estimate time remaining
        elapsed_sec=$(ps -o etime= -p $PID | tr '-' ':' | awk -F: '{ if (NF == 2) { print ($1 * 60) + $2 } else if (NF == 3) { print ($1 * 3600) + ($2 * 60) + $3 } else { print ($1 * 86400) + ($2 * 3600) + ($3 * 60) + $4 } }')
        if [ ! -z "$elapsed_sec" ] && [ $last_bar -gt 0 ]; then
            bars_per_sec=$(echo "scale=2; $last_bar / $elapsed_sec" | bc)
            remaining_bars=$((total_bars - last_bar))
            remaining_sec=$(echo "scale=0; $remaining_bars / $bars_per_sec" | bc)
            remaining_min=$((remaining_sec / 60))
            echo "Bars/sec: $bars_per_sec"
            echo "Est. remaining: ${remaining_min} minutes"
        fi
    fi
fi

echo ""
echo "==================================================================="
echo "Run this script again to check progress:"
echo "  bash check_optimized_hmm_progress.sh"
echo "==================================================================="
