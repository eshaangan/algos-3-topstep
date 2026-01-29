#!/usr/bin/env python3
"""
Test BALANCED_V3 model across multiple long-term periods and market regimes.

Periods:
1. Q1 2024 (Bearish/Correction)
2. Q3 2024 (Volatile)
3. Full Year 2024 (Long-term)
4. Jan-Nov 2025 (In-Sample Sanity Check)
5. Dec 2025 (Out-of-Sample)
6. Jan 2026 (Out-of-Sample Bear/Chop)
"""

import logging
import sys
import shutil
import tempfile
import glob
import os
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.live_trading.replay import replay_session

#
# NOTE:
# `replay_session()` is extremely chatty (WARNING/INFO) and can make long multi-period
# tests painfully slow. We silence root logging and emit only this script's INFO lines.
#
logging.basicConfig(level=logging.ERROR, format="%(asctime)s [%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    _h = logging.StreamHandler(stream=sys.stdout)
    _h.setLevel(logging.INFO)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)

# Silence extremely chatty modules so multi-period runs finish in reasonable time.
for _noisy_logger in (
    "live_trading.execution_engine",
    "live_trading.feature_generator",
    "live_trading.replay",
    "backtesting_v3.risk",
):
    logging.getLogger(_noisy_logger).setLevel(logging.ERROR)

def run_backtest(model_name: str, model_path: Path, bars: pd.DataFrame, config_dir: Path) -> dict:
    """Run backtest and return summary metrics."""
    logger.info(f"\n   Running backtest for {len(bars):,} bars...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_dir = Path(temp_dir) / "backtest_run"
        bar_dir = run_dir / "bar_size=5m"
        bar_dir.mkdir(parents=True, exist_ok=True)
        
        # Save bars
        bars.to_parquet(bar_dir / "bars.parquet")
        
        # Create minimal label_schema.json
        label_schema = {
            "stop_multiple": 1.5,
            "target_multiple": 2.5,
            "atr_period": 14,
            "atr_bar_size": "5m",
        }
        with open(bar_dir / "label_schema.json", "w") as f:
            json.dump(label_schema, f)
        
        # Copy model to walkforward dir
        wf_dir = run_dir / "walkforward" / "bar_size=5m" / "window_0"
        wf_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(model_path, wf_dir / "model_bundle.pkl")
        
        # Run replay
        try:
            result = replay_session(
                run_dir=run_dir,
                config_dir=config_dir,
                bar_size="5m",
                model_bundle_path=None,
                start=None,
                end=None,
                max_bars=None,
                output_dir=None
            )
        except Exception as e:
            logger.error(f"   Backtest execution failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
        
        # Extract metrics
        trade_files = glob.glob("logs/trades_*.csv")
        if trade_files:
            latest_trade_file = max(trade_files, key=os.path.getctime)
            trades = pd.read_csv(latest_trade_file)
        else:
            trades = result.closed_positions if hasattr(result, 'closed_positions') else pd.DataFrame()
        
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'long_pct': 0,
                'short_pct': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_trade': 0,
                'sharpe': 0
            }
        
        # Direction
        if 'direction' in trades.columns:
            longs = (trades['direction'] == 'LONG').sum()
            shorts = (trades['direction'] == 'SHORT').sum()
        elif 'side' in trades.columns:
            longs = (trades['side'] == 1).sum()
            shorts = (trades['side'] == -1).sum()
        else:
            longs = 0
            shorts = 0
            
        total = len(trades)
        
        # P&L
        pnl_col = next((c for c in ['pnl', 'pnl_usd', 'realized_pnl'] if c in trades.columns), None)
        if not pnl_col:
            return {'error': 'No P&L column'}
            
        winners = (trades[pnl_col] > 0).sum()
        total_pnl = trades[pnl_col].sum()
        avg_trade = trades[pnl_col].mean()
        
        # Sharpe (approx)
        std_pnl = trades[pnl_col].std()
        sharpe = avg_trade / std_pnl if std_pnl > 0 else 0
        
        return {
            'total_trades': total,
            'long_pct': longs / total * 100,
            'short_pct': shorts / total * 100,
            'win_rate': winners / total * 100,
            'total_pnl': total_pnl,
            'avg_trade': avg_trade,
            'sharpe': sharpe
        }

def main():
    logger.info("="*80)
    logger.info("BALANCED_V3: LONG-TERM & MULTI-REGIME VALIDATION")
    logger.info("="*80)
    
    # 1. Load Data Sources
    logger.info("Loading data...")
    
    # Historical H5 (2020-2025)
    h5_path = Path("data/processed/mes_bars_databento_rth.h5")
    bars_hist = pd.read_hdf(h5_path, key='bars_5min')
    bars_hist['timestamp'] = pd.to_datetime(bars_hist['timestamp'])
    bars_hist = bars_hist.set_index('timestamp').sort_index()
    
    # Pre-calculate Regime (SMA 200 Days approx = 55,200 bars)
    # Using 50 Days (approx 13,800 bars) as a responsive proxy
    sma_period = 13800
    logger.info(f"Pre-calculating Regime (SMA {sma_period})...")
    
    # Ensure sufficient data
    if len(bars_hist) > sma_period:
        bars_hist['sma_long'] = bars_hist['close'].rolling(window=sma_period).mean()
        bars_hist['regime'] = 0
        bars_hist.loc[bars_hist['close'] > bars_hist['sma_long'], 'regime'] = 1  # Bull
        bars_hist.loc[bars_hist['close'] < bars_hist['sma_long'], 'regime'] = -1 # Bear
        
        # Fill NaN regimes with 0 (Neutral)
        bars_hist['regime'] = bars_hist['regime'].fillna(0).astype(int)
        
        logger.info(f"Regime distribution: {bars_hist['regime'].value_counts().to_dict()}")
    else:
        logger.warning("Not enough data for regime calculation!")

    def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DatetimeIndex is tz-aware UTC (required for reliable slicing)."""
        if df.empty:
            return df
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Expected DataFrame indexed by DatetimeIndex")
        if df.index.tz is None:
            df = df.copy()
            df.index = df.index.tz_localize("UTC")
        else:
            df = df.copy()
            df.index = df.index.tz_convert("UTC")
        return df

    bars_hist = _ensure_utc_index(bars_hist)

    # Recent Parquet (Jan 2026) - already prepared by prior workflow
    jan26_path = Path("ml_intraday_v3/backtest_results/jan_2026_test/bar_size=5m/bars.parquet")
    if jan26_path.exists():
        bars_jan26 = pd.read_parquet(jan26_path)
        bars_jan26 = _ensure_utc_index(bars_jan26)
    else:
        logger.warning(f"Jan 2026 data not found at {jan26_path}")
        bars_jan26 = pd.DataFrame()

    # 2. Define Periods (longer + multi-regime)
    periods = [
        {"name": "Q1 2024 (Bear/Correction)", "data": bars_hist, "start": "2024-02-01", "end": "2024-03-31"},
        {"name": "Q3 2024 (Volatile)", "data": bars_hist, "start": "2024-08-01", "end": "2024-09-30"},
        {"name": "Full Year 2024", "data": bars_hist, "start": "2024-01-01", "end": "2024-12-31"},
        {"name": "Jan-Nov 2025 (In-Sample Sanity)", "data": bars_hist, "start": "2025-01-01", "end": "2025-11-30"},
        # Data currently ends at 2025-12-18 in mes_bars_databento_rth.h5, so cap holdout accordingly.
        {"name": "Dec 2025 (Holdout)", "data": bars_hist, "start": "2025-12-01", "end": "2025-12-18"},
    ]

    if not bars_jan26.empty:
        periods.append({"name": "Jan 2026 (Out-of-Sample)", "data": bars_jan26, "start": None, "end": None})
        
    # 3. Model Setup
    model_path = Path("ml_intraday_v3/models/saved/model_bundle_balanced_v3.pkl")
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return
        
    config_dir = Path("ml_intraday_v3/configs")
    
    # 4. Run Tests
    results = []
    
    for p in periods:
        logger.info(f"\nTesting Period: {p['name']}")
        
        # Slice data
        df = p['data']
        if p['start']:
            start_dt = pd.Timestamp(p['start'], tz='UTC')
            end_dt = pd.Timestamp(p['end'], tz='UTC')
            # Ensure index is UTC
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
                
            df_slice = df[(df.index >= start_dt) & (df.index <= end_dt)]
        else:
            df_slice = df
            
        if df_slice.empty:
            logger.warning(f"   No data found for period {p['name']}")
            continue
            
        res = run_backtest(p['name'], model_path, df_slice, config_dir)
        if 'error' in res:
            logger.error(f"   Error: {res['error']}")
            continue
            
        res['period'] = p['name']
        results.append(res)
        
        logger.info(f"   Trades: {res['total_trades']}")
        logger.info(f"   Win Rate: {res['win_rate']:.1f}%")
        logger.info(f"   P&L: ${res['total_pnl']:,.2f}")
        
    # 5. Final Report
    logger.info("\n" + "="*80)
    logger.info("FINAL PERFORMANCE REPORT: BALANCED_V3")
    logger.info("="*80)
    
    res_df = pd.DataFrame(results)
    
    logger.info(f"\n{'Period':<30} {'Trades':>8} {'LONG%':>8} {'SHORT%':>8} {'Win%':>8} {'Avg Trade':>12} {'Total P&L':>12}")
    logger.info("-" * 110)
    
    for _, row in res_df.iterrows():
        logger.info(
            f"{row['period']:<30} "
            f"{row['total_trades']:>8.0f} "
            f"{row['long_pct']:>7.1f}% "
            f"{row['short_pct']:>7.1f}% "
            f"{row['win_rate']:>7.1f}% "
            f"${row['avg_trade']:>11.2f} "
            f"${row['total_pnl']:>11,.0f}"
        )

if __name__ == "__main__":
    main()
