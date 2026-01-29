#!/bin/bash
# Quick Retraining Script: One-command solution to retrain model on recent data
# 
# Usage:
#   bash quick_retrain.sh              # Full retraining with data fetch
#   bash quick_retrain.sh --cached     # Use cached data
#   bash quick_retrain.sh --help       # Show help

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored message
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

# Check if running from ml_intraday_v3 directory
if [ ! -f "retrain_q4_jan26.py" ]; then
    print_error "Must run from ml_intraday_v3 directory"
    echo "Usage: cd ml_intraday_v3 && bash quick_retrain.sh"
    exit 1
fi

# Parse arguments
USE_CACHED=false
if [ "$1" == "--cached" ]; then
    USE_CACHED=true
elif [ "$1" == "--help" ]; then
    echo "Quick Retraining Script"
    echo ""
    echo "Usage:"
    echo "  bash quick_retrain.sh              Full retraining (fetch data from Databento)"
    echo "  bash quick_retrain.sh --cached     Use cached data (skip API fetch)"
    echo "  bash quick_retrain.sh --help       Show this help"
    echo ""
    echo "What it does:"
    echo "  1. Fetch Q4 2024 + Jan 2026 data from Databento"
    echo "  2. Train model with same architecture as baseline"
    echo "  3. Create production model bundle"
    echo "  4. Generate deployment guide"
    echo ""
    echo "Expected time: ~10 minutes"
    exit 0
fi

# Header
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  MODEL RETRAINING: Fix Distribution Shift & Direction Bias"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check prerequisites
print_step "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "python3 not found"
    exit 1
fi
print_success "Python 3 found"

# Check .env file
if [ ! -f "../.env" ]; then
    print_warning ".env file not found in parent directory"
    print_warning "Databento API key may not be available"
    if [ "$USE_CACHED" = false ]; then
        print_error "Cannot fetch data without .env file"
        echo "Options:"
        echo "  1. Create ../.env with DATABENTO_API_KEY=your_key"
        echo "  2. Run with --cached flag to use cached data"
        exit 1
    fi
else
    print_success ".env file found"
fi

# Check required packages
print_step "Checking required packages..."

python3 -c "import pandas, numpy, lightgbm, joblib, yaml, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "Missing required packages"
    echo "Install with: pip install -r requirements-mlv3.txt"
    exit 1
fi
print_success "All packages installed"

# Run retraining
echo ""
print_step "Starting retraining process..."
echo ""

if [ "$USE_CACHED" = true ]; then
    print_warning "Using cached data (skipping Databento API)"
    python3 retrain_q4_jan26.py --use-cached-data
else
    print_step "Fetching fresh data from Databento (this may take 5 minutes)..."
    python3 retrain_q4_jan26.py
fi

RETRAIN_EXIT_CODE=$?

echo ""
echo "═══════════════════════════════════════════════════════════"

if [ $RETRAIN_EXIT_CODE -eq 0 ]; then
    print_success "RETRAINING COMPLETE!"
    echo ""
    echo "📦 New Model: models/saved/model_bundle_retrained_q4_jan26.pkl"
    echo "📋 Deployment Guide: RETRAINED_MODEL_DEPLOYMENT.md"
    echo ""
    print_warning "CRITICAL NEXT STEPS:"
    echo ""
    echo "1. Run validation backtest:"
    echo "   python backtest_databento_recent.py \\"
    echo "       --model-bundle models/saved/model_bundle_retrained_q4_jan26.pkl \\"
    echo "       --start-date 2026-01-04 \\"
    echo "       --end-date 2026-01-23"
    echo ""
    echo "2. Check success criteria:"
    echo "   - Win rate > 40% (baseline: 13.7%)"
    echo "   - Profit factor > 1.0 (baseline: 0.19)"
    echo "   - LONG % = 40-60% (baseline: 100%)"
    echo "   - SHORT % = 40-60% (baseline: 0%)"
    echo ""
    echo "3. Paper trade for 1 week (MANDATORY):"
    echo "   python live_trading/paper_trade.py --duration 7d"
    echo ""
    echo "4. Deploy to production ONLY if paper trading passes"
    echo ""
    print_success "See RETRAINED_MODEL_DEPLOYMENT.md for full deployment guide"
else
    print_error "RETRAINING FAILED"
    echo ""
    echo "Check logs above for error details"
    echo ""
    echo "Common issues:"
    echo "  1. Databento API key not set (use --cached if data already fetched)"
    echo "  2. Missing packages (pip install -r requirements-mlv3.txt)"
    echo "  3. Insufficient RAM (need 4GB+)"
    echo ""
    exit 1
fi

echo "═══════════════════════════════════════════════════════════"
echo ""
