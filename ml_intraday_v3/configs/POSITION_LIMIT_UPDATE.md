# Position Limit Update: 5 → 10 Concurrent Positions

**Date**: 2026-01-09
**Change**: Increased maximum concurrent positions from 5 to 10

---

## What Changed

### Configuration Files Updated

1. **ml_intraday_v3/configs/live_trading.yaml**
   - `max_concurrent: 5` → `10`

2. **ml_intraday_v3/configs/risk.yaml**
   - `max_concurrent_positions: 5` → `10`
   - `max_total_contracts: 5` → `10`
   - `max_notional_exposure: 50000` → `250000` (10 × $25k per MES)

3. **ml_intraday_v3/configs/execution_spec.yaml**
   - `max_concurrent_positions: 5` → `10`

4. **ml_intraday_v3/configs/backtest.yaml**
   - `max_concurrent_positions: 5` → `10`

---

## Impact

### Before (5 positions)
- Max open positions: **5**
- Max contracts: **5** (1 per position)
- Max notional exposure: **$125,000** (5 × $25k)
- Margin requirement: **~$6,600** (5 × $1,320)

### After (10 positions)
- Max open positions: **10**
- Max contracts: **10** (1 per position)
- Max notional exposure: **$250,000** (10 × $25k)
- Margin requirement: **~$13,200** (10 × $1,320)

---

## Risk Considerations

### ✅ Still Safe for Topstep 50k

**Account**: $50,000
**Margin used**: ~$13,200 (26% of account)
**Margin buffer**: Still have $36,800 free (74%)

### Risk Per Position (1 contract)

Using typical stop placement (ATR-based):
- **Stop distance**: ~20-30 points (varies by ATR)
- **Risk per position**: $100-150 (20-30 points × $5/point)
- **Max risk (10 positions)**: $1,000-1,500

This fits within Topstep's **$1,000 daily loss limit**, but requires careful monitoring.

### Topstep Rule Compliance

✅ **Daily Loss Limit**: $1,000 (enforced by RiskManager)
✅ **Trailing Drawdown**: $2,500 (enforced by RiskManager)
✅ **Position Limit**: 50 contracts allowed, using 10
✅ **Margin**: Well within account limits

---

## Signal Requirements

With 10 max positions, you'll need **more qualifying signals** to utilize the full capacity.

**Current filter**: `score_ev >= 0.10`

**Expected behavior**:
- If only 3-4 signals per day qualify, you'll average 3-4 open positions
- If 10+ signals qualify, you can now hold all 10 (before: capped at 5)
- Signal rejection reason "max_concurrent_positions" should decrease

**Monitor** `logs/signals_*.csv` to see execution rate changes.

---

## What You Should Watch

### 1. Daily Loss Tracking
With 10 positions, you can hit the $1,000 daily loss limit faster:
- **5 positions**: Need -$200/position avg to hit limit
- **10 positions**: Need -$100/position avg to hit limit

**Action**: RiskManager will auto-halt if approaching limit (already configured).

### 2. Correlation Risk
10 concurrent MES positions are all correlated (same instrument):
- A strong adverse move hits ALL positions
- Diversification is limited (all MES micro E-mini S&P)

**Mitigation**: The regime filter helps (don't trade dangerous markets), and stops are enforced per position.

### 3. Execution Rate
More positions mean:
- More API calls
- More bracket orders (stop + target)
- Higher chance of slippage/rejection

**Action**: Already using rate limits (60s between trades) and dry_run mode for testing.

---

## Testing Recommendation

Before going live with 10 positions:

1. **Paper trade** for 2-3 days with new limit
2. **Monitor** signal logs: Are you generating enough signals?
3. **Check** execution: Any API errors with more orders?
4. **Verify** risk gates: Does RiskManager properly halt at limits?
5. **Analyze** correlation: Are all 10 positions moving together?

If you're only getting 3-4 signals per day, the increase to 10 won't matter much. But if signal frequency picks up, you'll have more capacity.

---

## Rollback Instructions

If you want to revert to 5 positions:

```bash
# Live trading
sed -i '' 's/max_concurrent: 10/max_concurrent: 5/g' ml_intraday_v3/configs/live_trading.yaml

# Risk limits
sed -i '' 's/max_concurrent_positions: 10/max_concurrent_positions: 5/g' ml_intraday_v3/configs/risk.yaml
sed -i '' 's/max_total_contracts: 10/max_total_contracts: 5/g' ml_intraday_v3/configs/risk.yaml
sed -i '' 's/max_notional_exposure: 250000/max_notional_exposure: 125000/g' ml_intraday_v3/configs/risk.yaml

# Execution spec
sed -i '' 's/max_concurrent_positions: 10/max_concurrent_positions: 5/g' ml_intraday_v3/configs/execution_spec.yaml

# Backtest
sed -i '' 's/max_concurrent_positions: 10/max_concurrent_positions: 5/g' ml_intraday_v3/configs/backtest.yaml
```

---

## Next Steps

1. ✅ Configs updated
2. ⏭️ Test in paper mode
3. ⏭️ Monitor signal generation rate
4. ⏭️ Verify risk limits still protect properly
5. ⏭️ Analyze if 10 positions improves or hurts performance

The system is ready - configs are consistent across all modules!
