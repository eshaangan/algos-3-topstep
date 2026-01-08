#!/bin/bash
# Monday Morning Startup Script
# Run this at 7:45 AM to verify everything is ready

echo "================================================================================"
echo "🚀 MONDAY MORNING STARTUP CHECKS"
echo "================================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
ALL_GOOD=true

# 1. Check directory
echo "1️⃣  Checking directory..."
if [ "$(basename "$(pwd)")" = "ml_intraday_v3" ]; then
    echo -e "   ${GREEN}✅ In correct directory${NC}"
else
    echo -e "   ${RED}❌ Wrong directory! Should be in ml_intraday_v3${NC}"
    echo "   Run: cd ml_intraday_v3"
    ALL_GOOD=false
fi
echo ""

# 2. Check model bundle
echo "2️⃣  Checking model bundle..."
if ls models/saved/*.pkl 1> /dev/null 2>&1; then
    MODEL_SIZE=$(ls -lh models/saved/*.pkl | awk '{print $5}')
    echo -e "   ${GREEN}✅ Model bundle found (${MODEL_SIZE})${NC}"
else
    echo -e "   ${RED}❌ Model bundle not found!${NC}"
    echo "   Expected: models/saved/model_bundle.pkl"
    ALL_GOOD=false
fi
echo ""

# 3. Check .env file
echo "3️⃣  Checking environment variables..."
if [ -f "../.env" ]; then
    if grep -q "DATABENTO_API_KEY" ../.env && grep -q "TOPSTEPX_ACCOUNT_ID" ../.env; then
        ACCOUNT_ID=$(grep "TOPSTEPX_ACCOUNT_ID" ../.env | cut -d'=' -f2)
        echo -e "   ${GREEN}✅ .env file configured${NC}"
        echo "   Account ID: $ACCOUNT_ID"
    else
        echo -e "   ${YELLOW}⚠️  .env missing some keys${NC}"
        ALL_GOOD=false
    fi
else
    echo -e "   ${RED}❌ .env file not found!${NC}"
    ALL_GOOD=false
fi
echo ""

# 4. Check infrastructure
echo "4️⃣  Running infrastructure tests..."
if python3 tests/test_infrastructure_fixes.py > /tmp/test_output.txt 2>&1; then
    echo -e "   ${GREEN}✅ Infrastructure tests PASSED${NC}"
else
    echo -e "   ${RED}❌ Infrastructure tests FAILED${NC}"
    echo "   Check: cat /tmp/test_output.txt"
    ALL_GOOD=false
fi
echo ""

# 5. Check monitoring
echo "5️⃣  Testing monitoring system..."
if [ -d "monitoring" ]; then
    echo -e "   ${GREEN}✅ Monitoring modules present${NC}"
else
    echo -e "   ${YELLOW}⚠️  Monitoring directory not found${NC}"
fi
echo ""

# 6. Check logs directory
echo "6️⃣  Checking logs directory..."
if [ -d "logs" ]; then
    echo -e "   ${GREEN}✅ Logs directory exists${NC}"
else
    echo -e "   ${YELLOW}⚠️  Creating logs directory...${NC}"
    mkdir -p logs
    echo -e "   ${GREEN}✅ Created logs directory${NC}"
fi
echo ""

# 7. Check market day
echo "7️⃣  Checking market day..."
DAY_OF_WEEK=$(date +%u)
if [ "$DAY_OF_WEEK" -ge 1 ] && [ "$DAY_OF_WEEK" -le 5 ]; then
    echo -e "   ${GREEN}✅ Today is $(date +%A) - Trading day!${NC}"
else
    echo -e "   ${YELLOW}⚠️  Today is $(date +%A) - Weekend!${NC}"
    echo "   Markets are closed"
fi
echo ""

# 8. Check time
echo "8️⃣  Checking time..."
HOUR=$(date +%H)
MINUTE=$(date +%M)
CURRENT_TIME=$(date +"%I:%M %p")

echo "   Current time: $CURRENT_TIME"

if [ "$HOUR" -lt 7 ]; then
    echo -e "   ${YELLOW}⚠️  Too early! Come back at 7:45 AM${NC}"
elif [ "$HOUR" -ge 7 ] && [ "$HOUR" -lt 8 ]; then
    echo -e "   ${GREEN}✅ Perfect time to run checks${NC}"
elif [ "$HOUR" -eq 8 ] && [ "$MINUTE" -lt 30 ]; then
    echo -e "   ${GREEN}✅ Good - ready to start system${NC}"
elif [ "$HOUR" -ge 8 ] && [ "$HOUR" -lt 15 ]; then
    echo -e "   ${GREEN}✅ Market is open - start trading!${NC}"
else
    echo -e "   ${YELLOW}⚠️  After market hours${NC}"
fi
echo ""

# Final summary
echo "================================================================================"
if [ "$ALL_GOOD" = true ]; then
    echo -e "${GREEN}✅ ALL CHECKS PASSED - READY TO TRADE!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review: cat MONDAY_MORNING_CHECKLIST.md"
    echo "  2. Start trading: PYTHONPATH=\"..\" python live_trading/live_runner.py"
else
    echo -e "${RED}❌ SOME CHECKS FAILED - FIX BEFORE TRADING${NC}"
    echo ""
    echo "Review errors above and fix before proceeding"
fi
echo "================================================================================"
echo ""
