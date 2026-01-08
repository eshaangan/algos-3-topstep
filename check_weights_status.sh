#!/bin/bash
# Quick status check for build-weights

echo "=== HMM Build-Weights Status ==="
echo ""

# Check if process is running
if ps -p 33313 > /dev/null 2>&1; then
    echo "Status: ✅ RUNNING"
    echo ""
    
    # Runtime
    ELAPSED=$(ps -p 33313 -o etime | tail -1 | xargs)
    echo "Runtime: $ELAPSED"
    echo ""
    
    # Latest progress
    echo "Latest activity:"
    tail -3 /tmp/claude/-Users-eshaanganguly-Documents-projects-algos-3-topstep/tasks/bf5e2d1.output | grep -o "Current: [0-9]*\.[0-9]*" | tail -1
    echo ""
    
    # Output size
    LINES=$(wc -l < /tmp/claude/-Users-eshaanganguly-Documents-projects-algos-3-topstep/tasks/bf5e2d1.output)
    echo "Warnings generated: $LINES"
    echo ""
    
    echo "Estimated remaining: 2-8 hours (uncertain)"
else
    echo "Status: ✅ COMPLETED!"
    echo ""
    
    # Check if training started
    if ps aux | grep "build-train" | grep python | grep -v grep > /dev/null; then
        echo "Training: ✅ STARTED"
    else
        echo "Training: ⏸️  PENDING (checking in 10 sec)"
    fi
fi
