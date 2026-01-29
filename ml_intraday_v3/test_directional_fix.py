#!/usr/bin/env python3
"""
Test directional bug fix by comparing:
1. OLD_BASELINE model with fixed replay.py
2. New retrained_clean model

Both should now show balanced LONG/SHORT distribution.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from ml_intraday_v3.live_trading.replay import replay_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def run_backtest(model_name: str, model_path: Path, bars: pd.DataFrame, config_dir: Path) -> dict:
    """Run backtest and return summary metrics."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: {model_name}")
    logger.info(f"{'='*80}")
    
    # Create temp run directory
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        run_dir = Path(temp_dir) / "backtest_run"
        bar_dir = run_dir / "bar_size=5m"
        bar_dir.mkdir(parents=True, exist_ok=True)
        
        # Save bars
        bars.to_parquet(bar_dir / "bars.parquet")
        
        # Create minimal label_schema.json
        import json
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
        result = replay_session(
            run_dir=run_dir,
            config_dir=config_dir,
            bar_size="5m",
            model_bundle_path=None,  # Will use walkforward bundle
            start=None,
            end=None,
            max_bars=None,
            output_dir=None
        )
        
        # Extract metrics from the saved trade log CSV (more reliable)
        import glob
        import os
        
        # Find most recent trades CSV
        trade_files = glob.glob("logs/trades_*.csv")
        if trade_files:
            latest_trade_file = max(trade_files, key=os.path.getctime)
            trades = pd.read_csv(latest_trade_file)
        else:
            trades = result.closed_positions
        
        if len(trades) == 0:
            logger.warning(f"   No trades generated!")
            return {
                'model': model_name,
                'total_trades': 0,
                'long_trades': 0,
                'short_trades': 0,
                'long_pct': 0,
                'short_pct': 0,
                'winners': 0,
                'losers': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0
            }
        
        logger.info(f"   Trade log columns: {list(trades.columns)}")
        
        # Get direction from 'direction' column (LONG/SHORT strings)
        if 'direction' in trades.columns:
            long_trades = (trades['direction'] == 'LONG').sum()
            short_trades = (trades['direction'] == 'SHORT').sum()
        elif 'side' in trades.columns:
            long_trades = (trades['side'] == 1).sum()
            short_trades = (trades['side'] == -1).sum()
        elif 'entry_side' in trades.columns:
            long_trades = (trades['entry_side'] == 1).sum()
            short_trades = (trades['entry_side'] == -1).sum()
        else:
            logger.warning(f"   No direction/side column found in trades!")
            long_trades = 0
            short_trades = 0
        
        total_trades = len(trades)
        
        # Get P&L from whichever column exists
        pnl_col = None
        for col in ['pnl', 'pnl_usd', 'realized_pnl']:
            if col in trades.columns:
                pnl_col = col
                break
        
        if pnl_col:
            winners = (trades[pnl_col] > 0).sum()
            losers = (trades[pnl_col] < 0).sum()
            win_rate = winners / total_trades if total_trades > 0 else 0
            total_pnl = trades[pnl_col].sum()
            avg_pnl = trades[pnl_col].mean()
        else:
            logger.warning("   No P&L column found!")
            winners = 0
            losers = 0
            win_rate = 0
            total_pnl = 0
            avg_pnl = 0
        
        summary = {
            'model': model_name,
            'total_trades': total_trades,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'long_pct': long_trades / total_trades * 100 if total_trades > 0 else 0,
            'short_pct': short_trades / total_trades * 100 if total_trades > 0 else 0,
            'winners': winners,
            'losers': losers,
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl
        }
        
        logger.info(f"\n   Total Trades: {total_trades:,}")
        logger.info(f"   LONG:  {long_trades:3d} ({summary['long_pct']:5.1f}%)")
        logger.info(f"   SHORT: {short_trades:3d} ({summary['short_pct']:5.1f}%)")
        logger.info(f"   Win Rate: {summary['win_rate']:.1f}% ({winners}W / {losers}L)")
        logger.info(f"   Total P&L: ${total_pnl:,.2f}")
        logger.info(f"   Avg P&L: ${avg_pnl:,.2f}")
        
        return summary


def main():
    logger.info("="*80)
    logger.info("MULTI-PERIOD BACKTEST VALIDATION")
    logger.info("="*80)
    logger.info("Testing model performance across multiple market regimes")
    logger.info("="*80)
    
    # Load data
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    logger.info(f"\nLoading data from: {data_path}")
    bars = pd.read_hdf(data_path, key='bars_5min')
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars = bars.set_index('timestamp').sort_index()
    
    # Define test periods across different market regimes
    test_periods = [
        {
            'name': 'Q1 2024 (Bearish)',
            'start': pd.Timestamp('2024-02-01', tz='UTC'),
            'end': pd.Timestamp('2024-03-31', tz='UTC'),
            'description': 'February-March 2024 market pullback'
        },
        {
            'name': 'Q3 2024 (Volatile)',
            'start': pd.Timestamp('2024-08-01', tz='UTC'),
            'end': pd.Timestamp('2024-09-30', tz='UTC'),
            'description': 'August-September 2024 volatility spike'
        },
        {
            'name': 'Dec 2025 (Bullish)',
            'start': pd.Timestamp('2025-12-01', tz='UTC'),
            'end': pd.Timestamp('2025-12-18', tz='UTC'),
            'description': 'December 2025 strong uptrend (held-out test)'
        },
    ]
    
    # Config directory
    config_dir = Path("ml_intraday_v3/configs")
    models_dir = Path("ml_intraday_v3/models/saved")
    
    # Models to test
    models_to_test = [
        {
            'name': 'RETRAINED_CLEAN (new)',
            'path': models_dir / "model_bundle_retrained_clean.pkl",
            'enabled': True
        },
        {
            'name': 'BALANCED_V3 (newest)',
            'path': Path("../runs/balanced_v3_q1_2024_q4_2025/walkforward/bar_size=5m/window_0/model_bundle.pkl"),
            'enabled': True
        },
    ]
    
    all_results = []
    
    # Test each model on each period
    for model in models_to_test:
        if not model['enabled']:
            continue
        
        if not model['path'].exists():
            logger.warning(f"\n⚠️  Model not found: {model['path']}")
            logger.warning(f"   Skipping {model['name']}")
            continue
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING MODEL: {model['name']}")
        logger.info(f"{'='*80}")
        
        for period in test_periods:
            # Filter bars to period
            bars_period = bars[(bars.index >= period['start']) & (bars.index <= period['end'])]
            
            if len(bars_period) == 0:
                logger.warning(f"   No data for period: {period['name']}")
                continue
            
            logger.info(f"\n{period['name']} ({period['description']})")
            logger.info(f"   {len(bars_period):,} bars ({bars_period.index[0].date()} to {bars_period.index[-1].date()})")
            
            try:
                result = run_backtest(
                    f"{model['name']} - {period['name']}",
                    model['path'],
                    bars_period,
                    config_dir
                )
                result['period'] = period['name']
                result['model_name'] = model['name']
                all_results.append(result)
            except Exception as e:
                logger.error(f"   ❌ Backtest failed: {e}")
                continue
    
    # Summary comparison
    logger.info("\n" + "="*80)
    logger.info("MULTI-PERIOD COMPARISON SUMMARY")
    logger.info("="*80)
    
    if not all_results:
        logger.error("No results to display!")
        return
    
    df = pd.DataFrame(all_results)
    
    logger.info("\nPerformance Across Market Regimes:")
    logger.info(f"{'Model':<30} {'Period':<20} {'Trades':>8} {'LONG%':>8} {'SHORT%':>8} {'Win%':>8} {'P&L':>12}")
    logger.info("-" * 110)
    
    for _, row in df.iterrows():
        logger.info(
            f"{row['model_name']:<30} "
            f"{row['period']:<20} "
            f"{row['total_trades']:>8.0f} "
            f"{row['long_pct']:>7.1f}% "
            f"{row['short_pct']:>7.1f}% "
            f"{row['win_rate']:>7.1f}% "
            f"${row['total_pnl']:>10,.0f}"
        )
    
    logger.info("\n" + "="*80)
    logger.info("KEY OBSERVATIONS BY PERIOD")
    logger.info("="*80)
    
    # Analyze by period
    for period_name in ['Q1 2024 (Bearish)', 'Q3 2024 (Volatile)', 'Dec 2025 (Bullish)']:
        period_results = df[df['period'] == period_name]
        if len(period_results) == 0:
            continue
        
        logger.info(f"\n{period_name}:")
        
        for _, row in period_results.iterrows():
            model = row['model_name']
            
            # Check directional balance
            if row['total_trades'] == 0:
                logger.warning(f"   ⚠️  {model}: No trades generated")
            elif row['long_pct'] > 90:
                logger.warning(f"   ⚠️  {model}: Extreme LONG bias ({row['long_pct']:.1f}%)")
            elif row['short_pct'] > 90:
                logger.warning(f"   ⚠️  {model}: Extreme SHORT bias ({row['short_pct']:.1f}%)")
            elif 30 <= row['long_pct'] <= 70:
                logger.info(f"   ✅ {model}: Balanced direction ({row['long_pct']:.1f}% LONG)")
            else:
                logger.info(f"   🔵 {model}: {row['long_pct']:.1f}% LONG, {row['short_pct']:.1f}% SHORT")
            
            # Check performance
            if row['win_rate'] > 50:
                logger.info(f"      ✅ Win rate: {row['win_rate']:.1f}% (profitable)")
            elif row['win_rate'] > 40:
                logger.info(f"      ⚠️  Win rate: {row['win_rate']:.1f}% (marginal)")
            else:
                logger.warning(f"      ❌ Win rate: {row['win_rate']:.1f}% (too low)")
    
    logger.info("\n" + "="*80)
    logger.info("AGGREGATE STATISTICS")
    logger.info("="*80)
    
    # Aggregate by model across all periods
    for model_name in df['model_name'].unique():
        model_results = df[df['model_name'] == model_name]
        
        avg_long_pct = model_results['long_pct'].mean()
        avg_win_rate = model_results['win_rate'].mean()
        total_pnl = model_results['total_pnl'].sum()
        total_trades = model_results['total_trades'].sum()
        
        logger.info(f"\n{model_name}:")
        logger.info(f"  Total trades: {total_trades:.0f} across {len(model_results)} periods")
        logger.info(f"  Avg LONG%: {avg_long_pct:.1f}%")
        logger.info(f"  Avg win rate: {avg_win_rate:.1f}%")
        logger.info(f"  Total P&L: ${total_pnl:,.2f}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ MULTI-PERIOD TESTING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
