# Trading Signals Skill - Quick Reference

The `generating-trading-signals` skill has been installed and customized for your MES/NKD futures trading project.

## Location

`.cursor/skills/generating-trading-signals/`

## What It Does

Generates trading signals using 7 technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Trend Analysis (SMA crossovers)
- Volume Analysis
- Stochastic Oscillator
- ADX (Average Directional Index)

Produces composite BUY/SELL signals with confidence scores and risk management levels (stop loss, take profit).

## Quick Start

### 1. Install Dependencies

```bash
pip install yfinance pandas numpy matplotlib
```

### 2. Scan Your Primary Instrument (MES)

```bash
python .cursor/skills/generating-trading-signals/scripts/scanner.py --symbols MES=F --detail
```

### 3. Scan All Futures Indices

```bash
python .cursor/skills/generating-trading-signals/scripts/scanner.py --watchlist futures_equity_indices --period 3m
```

### 4. Filter High-Confidence Signals

```bash
# Only show buy signals with 70%+ confidence
python .cursor/skills/generating-trading-signals/scripts/scanner.py \
  --watchlist futures_equity_indices \
  --filter buy \
  --min-confidence 70 \
  --rank confidence
```

## Available Watchlists (Customized for Your Project)

### Futures (Primary)
- `futures_equity_indices`: MES, ES, NQ, MNQ, YM, MYM, RTY, M2K
- `futures_international`: NKD, FDAX, FESX
- `futures_volatility`: VIX

### Reference (Original)
- `crypto_top10`: BTC-USD, ETH-USD, SOL-USD, etc.
- `crypto_defi`: UNI-USD, AAVE-USD, MKR-USD, etc.
- `stocks_tech`: AAPL, MSFT, GOOGL, etc.
- `etfs_major`: SPY, QQQ, IWM, DIA, VTI

## Common Commands

```bash
# Quick scan of MES
python .cursor/skills/generating-trading-signals/scripts/scanner.py --symbols MES=F

# Detailed analysis with all indicators
python .cursor/skills/generating-trading-signals/scripts/scanner.py --symbols MES=F --detail

# Scan multiple symbols
python .cursor/skills/generating-trading-signals/scripts/scanner.py --symbols MES=F,ES=F,NQ=F

# List all available watchlists
python .cursor/skills/generating-trading-signals/scripts/scanner.py --list-watchlists

# Save results to JSON
python .cursor/skills/generating-trading-signals/scripts/scanner.py \
  --watchlist futures_equity_indices \
  --output signals.json
```

## Signal Interpretation

### Signal Types
- **STRONG_BUY** (+2): Multiple strong buy indicators aligned
- **BUY** (+1): Moderate buy signals
- **NEUTRAL** (0): No clear direction
- **SELL** (-1): Moderate sell signals
- **STRONG_SELL** (-2): Multiple strong sell indicators aligned

### Confidence Levels
- **70-100%**: High conviction, strong signal
- **50-70%**: Moderate conviction
- **30-50%**: Weak signal, mixed indicators
- **0-30%**: No clear direction, avoid trading

## Configuration

Edit `.cursor/skills/generating-trading-signals/config/settings.yaml` to adjust:
- Indicator parameters (RSI periods, MACD settings, etc.)
- Signal weights (emphasize certain indicators)
- Risk management settings (ATR multiplier, R:R ratios)
- Watchlist customization

## Integration with Your ML System

Compare technical signals with ML predictions:

```bash
# Get technical signal
python .cursor/skills/generating-trading-signals/scripts/scanner.py \
  --symbols MES=F \
  --detail \
  --output technical_signals.json

# Run ML backtest
cd ml_intraday_v3
python live_trading/replay.py --config configs/live_trading.yaml

# Compare results
```

## AI Usage

When using Cursor AI, trigger this skill with phrases like:
- "Get trading signals for MES"
- "Check technical indicators for futures"
- "Analyze MES for entry opportunities"
- "Scan futures for buy signals"
- "Generate trading signals for my watchlist"

The skill will automatically activate and use the appropriate scripts.

## Files

- `SKILL.md`: Main skill instructions (read by AI)
- `scripts/scanner.py`: Main scanning tool
- `scripts/signals.py`: Signal generation logic
- `scripts/indicators.py`: Technical indicator calculations
- `config/settings.yaml`: Configuration
- `references/`: Additional documentation
  - `errors.md`: Troubleshooting guide
  - `examples.md`: Usage examples
  - `implementation.md`: Technical details

## Notes

- This skill uses yfinance for data, which provides delayed data (15-20 min for futures)
- For real-time analysis, integrate with your existing data sources (Databento/ProjectX)
- The skill complements your ML system by providing traditional technical analysis
- Risk management levels (stop loss/take profit) are based on ATR and should be adjusted for your risk parameters
