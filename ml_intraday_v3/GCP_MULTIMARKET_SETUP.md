# GCP Multi-Market Deployment Guide

**Goal**: Deploy MES (US) + NKD (Asian) trading system on GCP for 24/7 operation.

---

## Quick Start (5 Steps)

### Step 1: Ensure GCP Infrastructure Exists
```bash
cd ml_intraday_v3/gcp_scripts

# Create GCP instance (if not already exists)
./setup_infrastructure.sh
```

**Verifies**: Instance `algotrader` running on `us-central1-a`

### Step 2: Download Free Historical Data (Local)
```bash
cd ml_intraday_v3

# Run the startup script we created earlier
./start_multimarket.sh
```

**Downloads**:
- MES daily data (2020-2025)
- NKD daily data (2020-2025)
- Calculates correlation to verify diversification

### Step 3: Train Models (Local)
```bash
# Create NKD training config (copy from MES)
cp configs/training_mes.yaml configs/training_nkd.yaml

# Edit configs/training_nkd.yaml:
# - Change symbol: "MES" → "NKD"
# - Change data paths to use NKD data
# - Adjust model_dir: models/mes_production → models/nkd_production

# Train MES model (if not already trained)
python -m ml_intraday_v3.cli build-train --config configs/training_mes.yaml

# Train NKD model
python -m ml_intraday_v3.cli build-train --config configs/training_nkd.yaml
```

**Creates**:
- `runs/mes_production/bar_size=5m/model_bundle.pkl`
- `runs/nkd_production/bar_size=5m/model_bundle.pkl`

### Step 4: Deploy to GCP
```bash
cd ml_intraday_v3/gcp_scripts

# One-command deployment
./deploy_multimarket.sh
```

**This script**:
1. Packages both models + config
2. Uploads to GCP instance
3. Extracts and sets up
4. Creates systemd service
5. Verifies deployment

**Time**: 5-10 minutes

### Step 5: Start Trading
```bash
# SSH into instance
gcloud compute ssh algotrader --zone=us-central1-a

# Test in paper mode first
cd ~/algos/ml_intraday_v3
source venv/bin/activate
python live_trading/live_runner_multimarket.py \
    --config configs/live_trading_multimarket.yaml

# If test passes (let run for 10 minutes), enable as service
sudo systemctl enable algotrader-multimarket
sudo systemctl start algotrader-multimarket
sudo systemctl status algotrader-multimarket

# Monitor logs
tail -f ~/algos/logs/multimarket.log
```

---

## Architecture Overview

### Market Coverage
| Market | Symbol | Session (CT) | Model | Status |
|--------|--------|--------------|-------|--------|
| **Asian** | NKD | 18:00-03:00 | `runs/nkd_production/` | Active |
| **US** | MES | 08:30-15:00 | `runs/mes_production/` | Active |

**Total**: ~16 hours of trading per day (vs. 6.5 hours for MES-only)

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                   Multi-Market Trader                    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │         Portfolio Risk Manager                 │    │
│  │  - Tracks aggregate P&L across all markets    │    │
│  │  - Enforces Topstep daily loss limit          │    │
│  │  - Max concurrent positions: 2                 │    │
│  │  - Correlation-based sizing adjustment         │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────┐        ┌──────────────┐             │
│  │  MES Session │        │  NKD Session │             │
│  │  08:30-15:00 │        │  18:00-03:00 │             │
│  │  Model: V3   │        │  Model: V3   │             │
│  └──────────────┘        └──────────────┘             │
│                                                          │
│                  ┌──────────────┐                       │
│                  │ Time Monitor │                       │
│                  │ Auto-switch  │                       │
│                  └──────────────┘                       │
└─────────────────────────────────────────────────────────┘
           │                                    │
           ▼                                    ▼
    Databento/Topstep                   Cloud Logging
    Live Data Feed                      (Optional)
```

### Risk Management (Critical!)

**Portfolio-Level Constraints**:
```yaml
portfolio_risk:
  max_daily_loss: 1000          # Topstep limit (TOTAL across both markets)
  max_trailing_drawdown: 2000   # Topstep limit (TOTAL)
  max_concurrent_positions: 2   # Can hold MES + NKD simultaneously
  max_total_contracts: 2        # Max 2 contracts total
```

**Why Portfolio-Level?**
- Topstep sees TOTAL P&L, not per-market
- If you lose $500 in NKD + $600 in MES = $1100 total loss = VIOLATION
- Must track aggregate risk in real-time

**Per-Market Allocation**:
```yaml
kelly_sizing:
  per_market_allocation:
    mes: 0.60  # 60% of risk budget
    nkd: 0.40  # 40% of risk budget
```

### Session Management

**Auto-Switch Logic**:
- System checks current time (CT) every minute
- Activates appropriate market based on trading hours
- 15-minute buffer before session end (stops new trades)
- Logs market transitions

**Example Timeline**:
```
02:45 CT - NKD session ending (within 15min buffer, stop new trades)
03:00 CT - NKD session closed
03:00-08:30 - No active markets (system idle)
08:30 CT - MES session starts
15:00 CT - MES session ends
18:00 CT - NKD session starts
... (repeats)
```

---

## Configuration Files

### 1. Live Trading Config
**Location**: `configs/live_trading_multimarket.yaml`

**Key Settings**:
```yaml
markets:
  mes:
    symbol: "MES"
    trading_hours:
      start: "08:30"
      end: "15:00"
    model_bundle_path: "runs/mes_production/bar_size=5m/model_bundle.pkl"

  nkd:
    symbol: "NKD"
    trading_hours:
      start: "18:00"
      end: "03:00"
    model_bundle_path: "runs/nkd_production/bar_size=5m/model_bundle.pkl"

portfolio_risk:
  max_daily_loss: 1000
  max_concurrent_positions: 2

trading:
  environment: "paper"  # Start with paper mode!
  auto_switch_markets: true
```

### 2. Systemd Service
**Location**: `/etc/systemd/system/algotrader-multimarket.service`

**Auto-created by**: `deploy_multimarket.sh`

**Features**:
- Auto-restart on failure
- Memory limit: 3GB
- CPU quota: 180% (1.8 cores)
- Logs to `~/algos/logs/multimarket.log`

---

## Monitoring & Maintenance

### Real-Time Monitoring
```bash
# SSH into instance
gcloud compute ssh algotrader --zone=us-central1-a

# View live logs
tail -f ~/algos/logs/multimarket.log

# Check service status
sudo systemctl status algotrader-multimarket

# View resource usage
htop

# Check disk space
df -h
```

### Key Metrics to Watch
1. **Daily P&L**: Should not exceed -$1000
2. **Memory usage**: Should stay < 2GB (have 4GB total)
3. **Market switches**: Should see transitions at 03:00 and 08:30
4. **Active market**: Check logs show correct market based on time

### GCP MCP Integration

With the GCP MCP server now configured, you can use natural language to:

```
"Check if my algotrader instance is running"
"Show me the logs from my trading instance"
"What's the CPU usage on algotrader?"
"List all compute instances in my project"
```

**MCP Config** (already added):
```json
{
  "mcpServers": {
    "gcp": {
      "command": "npx",
      "args": ["-y", "@google-cloud/gcloud-mcp"],
      "env": {
        "CLOUDSDK_CORE_PROJECT": "trading-algo-3"
      }
    }
  }
}
```

---

## Cost Estimate

### Monthly Cost (After $300 Credits)
| Resource | Cost |
|----------|------|
| e2-medium instance | $20.45/mo |
| 30GB disk | $1.20/mo |
| Network egress | $0.00 (< 1GB) |
| **Total** | **$21.65/mo** |

**With $300 credits**: ~14 months free

### Resource Usage
- **RAM**: ~1.2 GB (30% of 4GB) with both markets
- **CPU**: ~20% average (have 200% = 2 cores)
- **Disk**: ~15 GB used (have 30 GB)

**Plenty of headroom** for both markets without upgrade.

---

## Troubleshooting

### Issue: Models not found on remote
**Solution**:
```bash
# Re-run deployment script
cd ml_intraday_v3/gcp_scripts
./deploy_multimarket.sh
```

### Issue: Market not switching
**Check**:
1. Logs show current time (CT)
2. Config has correct trading hours
3. System timezone is set correctly

```bash
# On GCP instance
timedatectl
# Should show: Time zone: America/Chicago
```

### Issue: Daily loss limit violated
**Check**:
```bash
# View state file
cat ~/algos/ml_intraday_v3/state/trading_state.json

# Shows:
# {
#   "daily_pnl": -850.50,
#   "current_balance": 49149.50,
#   "peak_balance": 50000.00
# }
```

**If > $1000 loss**: System automatically stops trading for the day

### Issue: GCP MCP not working
**Solution**:
```bash
# Install Node.js if not present
brew install node  # macOS

# Verify npx is available
npx --version

# Restart Claude Code to reload MCP
```

---

## Testing Protocol

### Week 1: Paper Trading Validation
- [ ] Day 1: Deploy to GCP, run in paper mode
- [ ] Day 2: Verify NKD trades during Asian session (18:00-03:00 CT)
- [ ] Day 3: Verify MES trades during US session (08:30-15:00 CT)
- [ ] Day 4: Check market switches happen correctly
- [ ] Day 5: Test portfolio risk limits (simulate loss scenario)
- [ ] Day 6-7: 48-hour continuous operation test

**Success Criteria**:
- ✅ Zero crashes
- ✅ Correct market switching
- ✅ Risk limits enforced
- ✅ Logs show both markets trading

### Week 2-3: Live Testing (Small Size)
- [ ] Switch to live mode with 1 contract max
- [ ] Monitor for 2 weeks
- [ ] Compare performance to backtest
- [ ] Verify Topstep rule compliance

---

## Deployment Checklist

**Pre-Deployment**:
- [ ] Historical data downloaded (MES + NKD)
- [ ] Both models trained and validated
- [ ] Multi-market config created
- [ ] GCP instance running

**Deployment**:
- [ ] Run `./deploy_multimarket.sh`
- [ ] Verify upload successful
- [ ] SSH and test manually
- [ ] Enable systemd service

**Post-Deployment**:
- [ ] Monitor logs for 1 hour
- [ ] Verify market switch at next session boundary
- [ ] Check resource usage (RAM, CPU, disk)
- [ ] Set up daily monitoring routine

---

## File Summary

### New Files Created
1. **`configs/live_trading_multimarket.yaml`** - Multi-market configuration
2. **`live_trading/live_runner_multimarket.py`** - 24/7 orchestrator
3. **`gcp_scripts/deploy_multimarket.sh`** - One-command deployment
4. **`data/download_free_futures.py`** - Free data download script
5. **`GCP_MULTIMARKET_SETUP.md`** - This guide
6. **`.mcp.json`** - Updated with GCP MCP server

### Updated Files
1. **`requirements-mlv3.txt`** - Added `yfinance`
2. **`.claude/settings.local.json`** - Added GCP MCP permissions

---

## Next Steps

### TODAY (30 minutes):
1. Run `./start_multimarket.sh` to download data
2. Create `configs/training_nkd.yaml`
3. Train NKD model locally

### THIS WEEK:
1. Run `./deploy_multimarket.sh` to deploy to GCP
2. Test in paper mode for 3 days
3. Verify both markets trade correctly

### NEXT WEEK:
1. Switch to live mode (small size)
2. Monitor performance
3. Start Topstep Combine with multi-market system

---

## Support Resources

- **GCP Deployment Plan**: `GCP_DEPLOYMENT_PLAN.md` (single-market)
- **Multi-Market Architecture**: `MULTI_MARKET_ARCHITECTURE.md`
- **Quick Start Guide**: `QUICK_START_MULTI_MARKET.md`
- **GCP Console**: https://console.cloud.google.com
- **GCP MCP Docs**: https://docs.cloud.google.com/mcp/overview

---

**Status**: Ready for Deployment
**Created**: 2026-01-23
**Estimated Setup Time**: 2-3 hours (including training)
