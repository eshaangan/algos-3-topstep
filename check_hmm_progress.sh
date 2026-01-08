#!/bin/bash
# Quick progress check for TEST HMM build-weights (2022-2025)

echo "=== TEST HMM Build-Weights Progress (2022-2025) ==="
echo ""

PID=62610

# Check if process is running
if ps -p $PID > /dev/null 2>&1; then
    echo "Status: ✅ RUNNING"
    echo ""

    # Runtime
    ELAPSED=$(ps -p $PID -o etime | tail -1 | xargs)
    echo "Runtime: $ELAPSED"
    echo ""

    # CPU & Memory
    CPU=$(ps -p $PID -o %cpu | tail -1 | xargs)
    MEM=$(ps -p $PID -o %mem | tail -1 | xargs)
    echo "CPU: ${CPU}% | Memory: ${MEM}%"
    echo ""

    # Latest progress update
    echo "Latest progress:"
    grep "HMM Progress:" /tmp/hmm_test_build_weights.log | tail -1
    echo ""

    # Output size
    LINES=$(wc -l < /tmp/hmm_test_build_weights.log)
    echo "Log lines generated: $LINES"
    echo ""

    echo "Estimated time: 3-5 hours total (TEST: 179k bars, 1,418 refits)"
    echo "Check progress: tail -f /tmp/hmm_test_build_weights.log"
else
    echo "Status: ✅ COMPLETED or ❌ STOPPED!"
    echo ""

    # Check last progress
    echo "Last progress update:"
    grep "HMM Progress:" /tmp/hmm_test_build_weights.log | tail -1
    echo ""

    # Check for completion message
    if grep -q "HMM expanding window prediction complete" /tmp/hmm_test_build_weights.log; then
        echo "✅ HMM COMPLETED SUCCESSFULLY!"
    else
        echo "⚠️  Process stopped unexpectedly - check logs"
    fi
fi

echo ""
