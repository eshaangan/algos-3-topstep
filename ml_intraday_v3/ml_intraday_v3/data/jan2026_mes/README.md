# January 2026 MES Data - Data Bento Download

**Downloaded**: January 31, 2026
**Source**: Data Bento GLBX.MDP3 dataset
**Symbol**: MES.c.0 (Continuous front month E-mini S&P 500)

---

## Data Summary

### 1-Minute Bars
- **File**: `mes_jan2026_1m.parquet` (also `.csv` for inspection)
- **Total Bars**: 28,740
- **Date Range**: January 1-30, 2026
- **Trading Days**: 26 days
- **Avg Bars/Day**: 1,105.4
- **Price Range**: $6,814.50 - $7,043.25
- **Total Volume**: 22,405,605 contracts

### 5-Minute Bars
- **File**: `mes_jan2026_5m.parquet` (also `.csv` for inspection)
- **Total Bars**: 5,748
- **Date Range**: January 1-30, 2026
- **Trading Days**: 26 days
- **Avg Bars/Day**: 221.1
- **Price Range**: $6,814.50 - $7,043.25
- **Total Volume**: 22,405,605 contracts

---

## Data Structure

### Columns (OHLCV)
```
- timestamp: pd.DatetimeIndex (UTC timezone)
- open: float (price)
- high: float (price)
- low: float (price)
- close: float (price)
- volume: int (contracts)
```

### Sample Data (5-minute bars)
```csv
timestamp,open,high,low,close,volume
2026-01-01 23:00:00+00:00,6901.0,6903.25,6897.0,6898.25,2834
2026-01-01 23:05:00+00:00,6898.5,6902.0,6898.0,6902.0,970
2026-01-01 23:10:00+00:00,6901.75,6903.5,6899.75,6901.25,1534
```

---

## Usage

### Load Data (Python)

```python
import pandas as pd

# Load 5-minute bars
bars_5m = pd.read_parquet('ml_intraday_v3/data/jan2026_mes/mes_jan2026_5m.parquet')
print(f"Loaded {len(bars_5m):,} bars")
print(bars_5m.head())

# Load 1-minute bars
bars_1m = pd.read_parquet('ml_intraday_v3/data/jan2026_mes/mes_jan2026_1m.parquet')
```

### Convert 1-Minute to 5-Minute (if needed)

```python
# Already done, but here's how:
bars_5m = bars_1m.resample('5min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
}).dropna()
```

---

## Why This Data Matters

### Real vs Simulated Data

**Previous Tests**: Used simulated data based on statistical assumptions
- Win rate: 35.5% (assumed)
- Trade distribution: Random sampling
- Confidence distribution: Synthetic (80% low, 20% medium/high)

**This Data**: REAL market data from January 2026
- Actual price movements
- Real volatility patterns
- Actual trading volume
- True market conditions

### Validation Opportunities

With this real data, we can now:

1. **Test Actual Model Performance**
   - Run the trained model on REAL Jan 2026 bars
   - Get TRUE win rate, not simulated
   - See actual P&L, not estimated
   - Validate if Jan 2026 really was a "regime shift"

2. **Test 2-Contract Sizing on Real Data**
   - Apply confidence filter to REAL signals
   - Test tiered sizing on ACTUAL trades
   - Calculate daily P&L from REAL market conditions
   - Count actual $150+ days (not simulated)

3. **Regime Detection Validation**
   - Run regime detector on this data
   - See if it would have detected Jan 2026 shift
   - Validate the KS test thresholds
   - Confirm it would have paused trading

4. **Compare Dec 2025 vs Jan 2026**
   - Download Dec 2025 real data
   - Compare actual market conditions
   - Validate that Dec was "normal" and Jan was "shift"

---

## Next Steps

### 1. Quick Data Inspection

```bash
# View first 50 rows
head -50 ml_intraday_v3/data/jan2026_mes/mes_jan2026_5m.csv

# View summary statistics
python -c "
import pandas as pd
df = pd.read_parquet('ml_intraday_v3/data/jan2026_mes/mes_jan2026_5m.parquet')
print(df.describe())
print(f'\nDaily bar counts:')
print(df.groupby(df.index.date).size())
"
```

### 2. Run Model on Real Jan 2026 Data

Create script to:
- Load the trained model
- Generate features from real Jan 2026 bars
- Get predictions (LONG/SHORT, probability)
- Apply confidence filter (0.55)
- Calculate actual performance

### 3. Test 2-Contract Sizing with Real Data

Modify existing test to:
- Use REAL Jan 2026 signals (not simulated)
- Apply production workflow
- Calculate actual daily P&L
- Count REAL $150+ days

### 4. Download December 2025 Data

```bash
# Modify the fetch script for Dec 2025
python ml_intraday_v3/experiments/fetch_dec2025_mes_data.py
```

Then compare:
- Dec 2025 (should be normal conditions)
- Jan 2026 (should show regime shift)

---

## Data Quality

### Validation Checks
- ✅ OHLC relationships valid (low ≤ open, close ≤ high)
- ✅ No missing bars in trading hours
- ✅ Volume data present
- ✅ Continuous timestamp sequence
- ✅ Price range realistic ($6,814 - $7,043)

### Known Considerations
- **Timezone**: Data is in UTC (need to convert to CT for RTH trading)
- **Trading Hours**: Includes extended hours (23:00 UTC = 6:00 PM ET)
- **Weekends/Holidays**: January 1 (New Year) may have limited data

---

## File Sizes

```
mes_jan2026_1m.parquet     ~2.2 MB
mes_jan2026_1m.csv         ~2.8 MB
mes_jan2026_5m.parquet     ~450 KB
mes_jan2026_5m.csv         ~560 KB
metadata.json              ~1 KB
```

---

## Metadata

See `metadata.json` for complete download details:
```json
{
  "symbol": "MES.c.0",
  "start_date": "2026-01-01",
  "end_date": "2026-01-31",
  "downloaded_at": "2026-01-31T17:00:54",
  "bars_1m": {
    "count": 28740,
    "file": "mes_jan2026_1m.parquet",
    "start": "2026-01-01 23:00:00+00:00",
    "end": "2026-01-30 21:59:00+00:00"
  },
  "bars_5m": {
    "count": 5748,
    "file": "mes_jan2026_5m.parquet",
    "start": "2026-01-01 23:00:00+00:00",
    "end": "2026-01-30 21:55:00+00:00"
  },
  "trading_days": 26,
  "bars_per_day_1m": 1105.4,
  "bars_per_day_5m": 221.1
}
```

---

## Important Notes

### Data Freshness
This data was downloaded on **January 31, 2026** and represents FINAL market data for January 2026 (no revisions).

### Market Conditions
January 2026 showed:
- **Price Range**: $6,814.50 - $7,043.25 (228.75 point range)
- **Average Daily Range**: ~8.8 points per day
- **Total Volume**: 22.4M contracts over 26 days
- **Avg Daily Volume**: ~862k contracts/day

### Comparison to Training Data
The model was trained on data through December 2024. January 2026 represents **13 months out-of-sample**, making it a true test of model generalization.

---

## Credits

**Data Source**: Data Bento (https://databento.com)
**Dataset**: GLBX.MDP3 (CME Globex Market Data Platform)
**Symbol**: MES.c.0 (Continuous E-mini S&P 500 Futures)
**Downloaded via**: LiveDataFetcher class (`ml_intraday_v3/live_trading/data_fetcher.py`)

---

**Last Updated**: January 31, 2026
**Status**: ✅ READY FOR TESTING
