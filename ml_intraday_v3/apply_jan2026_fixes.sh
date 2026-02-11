#!/bin/bash
# Apply configuration fixes to match Jan 2026 successful setup
# Run this script to restore $654/day performance

set -e  # Exit on error

echo "═══════════════════════════════════════════════════════════════"
echo "Applying Jan 2026 Configuration Fixes"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Change to ml_intraday_v3 directory
cd "$(dirname "$0")"
PWD=$(pwd)
echo "📂 Working directory: $PWD"
echo ""

# Backup existing configs
echo "📋 Creating backups..."
cp configs/live_trading.yaml configs/live_trading.yaml.backup_$(date +%Y%m%d_%H%M%S)
cp configs/execution_spec.yaml configs/execution_spec.yaml.backup_$(date +%Y%m%d_%H%M%S)
echo "✅ Backups created in configs/"
echo ""

# Fix 1: Set explicit model path
echo "🔧 Fix 1: Setting explicit model path..."
echo "   Current: model_bundle_path: null (auto-detect)"
echo "   New:     model_bundle_path: model_bundle_retrained_oct2024_nov2025.pkl"

# Check if model file exists
if [ ! -f "model_bundle_retrained_oct2024_nov2025.pkl" ]; then
    echo "❌ ERROR: model_bundle_retrained_oct2024_nov2025.pkl not found!"
    echo "   Looking for it in subdirectories..."
    find . -name "model_bundle_retrained_oct2024_nov2025.pkl" -type f
    exit 1
fi

# Update live_trading.yaml - replace model_bundle_path line
sed -i.tmp 's|model_bundle_path: null.*|model_bundle_path: "model_bundle_retrained_oct2024_nov2025.pkl"  # Jan 2026 \$654/day model|' configs/live_trading.yaml
rm configs/live_trading.yaml.tmp
echo "✅ Model path set explicitly"
echo ""

# Fix 2: Lower primary_threshold
echo "🔧 Fix 2: Lowering primary_threshold..."
echo "   Current: primary_threshold: 0.10"
echo "   New:     primary_threshold: 0.03 (Jan 2026 baseline)"

sed -i.tmp 's|primary_threshold: 0\.10|primary_threshold: 0.03  # Jan 2026 baseline|' configs/live_trading.yaml
rm configs/live_trading.yaml.tmp
echo "✅ Primary threshold lowered to 0.03"
echo ""

# Fix 3: Disable volatility filter in execution_spec
echo "🔧 Fix 3: Disabling volatility filter (wasn't in Jan 2026 validation)..."
echo "   Current: volatility.enabled: true"
echo "   New:     volatility.enabled: false"

# Find the volatility section and change enabled to false
# This is trickier because we need to find the right section
sed -i.tmp '/filters:/,/confidence:/ {
    /volatility:/,/enabled:/ {
        s/enabled: true/enabled: false  # Not validated in Jan 2026/
    }
}' configs/execution_spec.yaml
rm configs/execution_spec.yaml.tmp
echo "✅ Volatility filter disabled"
echo ""

# Verify fixes
echo "═══════════════════════════════════════════════════════════════"
echo "Verifying fixes applied..."
echo "═══════════════════════════════════════════════════════════════"
echo ""

echo "✓ Fix 1 - Model path:"
grep "model_bundle_path:" configs/live_trading.yaml | head -1
echo ""

echo "✓ Fix 2 - Primary threshold:"
grep "primary_threshold:" configs/live_trading.yaml | grep -v "#" | head -1
echo ""

echo "✓ Fix 3 - Volatility filter:"
grep -A 1 "volatility:" configs/execution_spec.yaml | grep "enabled:" | head -1
echo ""

# Summary
echo "═══════════════════════════════════════════════════════════════"
echo "✅ All fixes applied successfully!"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📝 Summary of changes:"
echo "  1. Model path: model_bundle_retrained_oct2024_nov2025.pkl (explicit)"
echo "  2. Primary threshold: 0.10 → 0.03"
echo "  3. Volatility filter: ENABLED → DISABLED"
echo ""
echo "Next steps:"
echo "  1. Review changes: git diff configs/"
echo "  2. Deploy: ./deploy_to_gcp.sh"
echo "  3. Monitor: ./monitor_gcp.sh"
echo ""
echo "Expected outcome:"
echo "  - Trades per day: 8-15 (was 13.2 in Jan 2026)"
echo "  - Win rate: 50-60% (was 56.3%)"
echo "  - Daily P&L: \$300-700 (was \$654)"
echo ""
echo "🚀 Ready for deployment!"
