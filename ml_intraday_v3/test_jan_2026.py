#!/usr/bin/env python3
"""
Fetch January 2026 MES data from Databento and test model performance.

Tests existing models on the most recent data to validate:
- Directional balance (LONG/SHORT distribution)
- Win rate and profitability
- Topstep rules compliance
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)

from ml_intraday_v3.live_trading.data_fetcher import LiveDataFetcher
from ml_intraday_v3.live_trading.replay import replay_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def fetch_jan_2026_data(output_dir: Path) -> Path:
    """
    Fetch January 2026 MES data from Databento.
    
    Returns:
        Path to run directory with fetched data
    """
    logger.info("="*80)
    logger.info("FETCHING JANUARY 2026 DATA FROM DATABENTO")
    logger.info("="*80)
    
    # Create fetcher for MES front month continuous contract
    fetcher = LiveDataFetcher(
        symbol="MES.c.0",  # MES front month, calendar roll
        bar_size="1m",
        lookback_bars=100,
    )
    
    # Fetch January 2026 data (up to today)
    start_date = "2026-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")  # Today
    
    logger.info(f"Fetching: {start_date} to {end_date}")
    
    try:
        bars_1m = fetcher.fetch_historical(start_date, end_date)
        logger.info(f"Fetched {len(bars_1m):,} 1-minute bars")
        
        if len(bars_1m) == 0:
            logger.error("No data fetched - check date range or Databento subscription")
            sys.exit(1)
        
        # Resample to 5-minute bars
        logger.info("Resampling to 5-minute bars...")
        bars_5m = bars_1m.resample('5T').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        logger.info(f"Resampled to {len(bars_5m):,} 5-minute bars")
        logger.info(f"Date range: {bars_5m.index[0]} to {bars_5m.index[-1]}")
        
        # Create run directory structure
        run_dir = output_dir / "jan_2026_test"
        bar_dir = run_dir / "bar_size=5m"
        bar_dir.mkdir(parents=True, exist_ok=True)
        
        # Save bars
        bars_5m.to_parquet(bar_dir / "bars.parquet")
        logger.info(f"Saved bars to: {bar_dir / 'bars.parquet'}")
        
        # Create minimal label_schema.json
        import json
        label_schema = {
            "stop_multiple": 1.5,
            "target_multiple": 2.5,
            "atr_period": 14,
            "atr_bar_size": "5m",
        }
        with open(bar_dir / "label_schema.json", "w") as f:
            json.dump(label_schema, f, indent=2)
        
        logger.info(f"✓ Data prepared in: {run_dir}")
        
        return run_dir
        
    except Exception as e:
        logger.error(f"Failed to fetch data: {e}")
        logger.error("Please check:")
        logger.error("  1. DATABENTO_API_KEY is set in .env")
        logger.error("  2. Your Databento subscription includes MES")
        logger.error("  3. Date range is valid (Jan 2026 may not exist yet)")
        raise


def run_model_test(
    model_name: str,
    model_path: Path,
    run_dir: Path,
    config_dir: Path
) -> dict:
    """Run backtest on Jan 2026 data and return results."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING: {model_name}")
    logger.info(f"{'='*80}")
    
    import shutil
    
    # Copy model to walkforward structure
    wf_dir = run_dir / "walkforward" / "bar_size=5m" / "window_0"
    wf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(model_path, wf_dir / "model_bundle.pkl")
    
    # Run replay session
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
        
        # Extract metrics from saved trade log
        import glob
        import os
        
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
                'error': 'No trades'
            }
        
        # Determine direction column
        if 'direction' in trades.columns:
            direction_col = 'direction'
            long_val = 'LONG'
            short_val = 'SHORT'
        elif 'side' in trades.columns:
            direction_col = 'side'
            long_val = 1
            short_val = -1
        else:
            logger.error("   No direction column found!")
            return {'model': model_name, 'error': 'No direction column'}
        
        # Determine P&L column
        pnl_col = None
        for col in ['pnl', 'pnl_usd', 'realized_pnl']:
            if col in trades.columns:
                pnl_col = col
                break
        
        if pnl_col is None:
            logger.error("   No P&L column found!")
            return {'model': model_name, 'error': 'No P&L column'}
        
        # Calculate metrics
        long_trades = trades[trades[direction_col] == long_val]
        short_trades = trades[trades[direction_col] == short_val]
        
        winners = trades[trades[pnl_col] > 0]
        losers = trades[trades[pnl_col] < 0]
        
        metrics = {
            'model': model_name,
            'total_trades': len(trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_pct': len(long_trades) / len(trades) * 100,
            'short_pct': len(short_trades) / len(trades) * 100,
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(trades) * 100,
            'total_pnl': trades[pnl_col].sum(),
            'avg_win': winners[pnl_col].mean() if len(winners) > 0 else 0,
            'avg_loss': losers[pnl_col].mean() if len(losers) > 0 else 0,
            'avg_trade': trades[pnl_col].mean(),
            'best_trade': trades[pnl_col].max(),
            'worst_trade': trades[pnl_col].min(),
        }
        
        # Log summary
        logger.info(f"\n   Total trades: {metrics['total_trades']}")
        logger.info(f"   LONG: {metrics['long_trades']} ({metrics['long_pct']:.1f}%)")
        logger.info(f"   SHORT: {metrics['short_trades']} ({metrics['short_pct']:.1f}%)")
        logger.info(f"   Win rate: {metrics['win_rate']:.1f}%")
        logger.info(f"   Total P&L: ${metrics['total_pnl']:,.2f}")
        logger.info(f"   Avg trade: ${metrics['avg_trade']:,.2f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"   Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return {'model': model_name, 'error': str(e)}


def main():
    logger.info("="*80)
    logger.info("JANUARY 2026 MODEL PERFORMANCE TEST")
    logger.info("="*80)
    logger.info("This script will:")
    logger.info("  1. Fetch Jan 2026 MES data from Databento")
    logger.info("  2. Test available models")
    logger.info("  3. Compare performance and directional balance")
    logger.info("="*80)
    
    # Check for API key
    import os
    if not os.getenv("DATABENTO_API_KEY"):
        logger.error("\n❌ DATABENTO_API_KEY not found in environment!")
        logger.error("   Please add it to your .env file")
        sys.exit(1)
    
    # Setup paths
    output_dir = Path("ml_intraday_v3/backtest_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config_dir = Path("ml_intraday_v3/configs")
    models_dir = Path("ml_intraday_v3/models/saved")
    
    # Step 1: Fetch data
    try:
        run_dir = fetch_jan_2026_data(output_dir)
    except Exception as e:
        logger.error(f"\n❌ Data fetch failed: {e}")
        logger.info("\nNote: If January 2026 hasn't occurred yet, this is expected.")
        logger.info("      The script will work once Jan 2026 data is available.")
        sys.exit(1)
    
    # Step 2: Define models to test
    models_to_test = [
        {
            'name': 'RETRAINED_CLEAN',
            'path': models_dir / "model_bundle_retrained_clean.pkl",
            'enabled': True
        },
        {
            'name': 'OLD_BASELINE',
            'path': models_dir / "model_bundle_OLD_BASELINE.pkl",
            'enabled': True
        },
        {
            'name': 'CURRENT_PRODUCTION',
            'path': models_dir / "model_bundle.pkl",
            'enabled': True
        },
    ]
    
    # Step 3: Test each model
    results = []
    
    for model in models_to_test:
        if not model['enabled']:
            continue
        
        if not model['path'].exists():
            logger.warning(f"\n⚠️  Model not found: {model['path']}")
            continue
        
        result = run_model_test(
            model['name'],
            model['path'],
            run_dir,
            config_dir
        )
        results.append(result)
    
    # Step 4: Summary comparison
    logger.info("\n" + "="*80)
    logger.info("JANUARY 2026 PERFORMANCE SUMMARY")
    logger.info("="*80)
    
    if not results:
        logger.error("No results to display!")
        return
    
    df = pd.DataFrame(results)
    
    logger.info(f"\n{'Model':<30} {'Trades':>8} {'LONG%':>8} {'SHORT%':>8} {'Win%':>8} {'P&L':>12}")
    logger.info("-" * 90)
    
    for _, row in df.iterrows():
        if 'error' in row:
            logger.error(f"{row['model']:<30} ERROR: {row['error']}")
            continue
        
        logger.info(
            f"{row['model']:<30} "
            f"{row['total_trades']:>8.0f} "
            f"{row['long_pct']:>7.1f}% "
            f"{row['short_pct']:>7.1f}% "
            f"{row['win_rate']:>7.1f}% "
            f"${row['total_pnl']:>10,.0f}"
        )
    
    # Analysis
    logger.info("\n" + "="*80)
    logger.info("KEY OBSERVATIONS")
    logger.info("="*80)
    
    for _, row in df.iterrows():
        if 'error' in row:
            continue
        
        model = row['model']
        
        # Check directional balance
        if row['total_trades'] == 0:
            logger.warning(f"\n⚠️  {model}: No trades generated")
        elif row['long_pct'] > 90:
            logger.warning(f"\n⚠️  {model}: Extreme LONG bias ({row['long_pct']:.1f}%)")
        elif row['short_pct'] > 90:
            logger.warning(f"\n⚠️  {model}: Extreme SHORT bias ({row['short_pct']:.1f}%)")
        elif 30 <= row['long_pct'] <= 70:
            logger.info(f"\n✅ {model}: Balanced direction ({row['long_pct']:.1f}% LONG)")
        else:
            logger.info(f"\n🔵 {model}: {row['long_pct']:.1f}% LONG, {row['short_pct']:.1f}% SHORT")
        
        # Check performance
        if row['win_rate'] > 50:
            logger.info(f"   ✅ Win rate: {row['win_rate']:.1f}% (profitable)")
        elif row['win_rate'] > 40:
            logger.info(f"   ⚠️  Win rate: {row['win_rate']:.1f}% (marginal)")
        else:
            logger.warning(f"   ❌ Win rate: {row['win_rate']:.1f}% (too low)")
        
        if row['total_pnl'] > 500:
            logger.info(f"   ✅ P&L: ${row['total_pnl']:,.2f} (strong)")
        elif row['total_pnl'] > 0:
            logger.info(f"   🔵 P&L: ${row['total_pnl']:,.2f} (positive)")
        else:
            logger.warning(f"   ❌ P&L: ${row['total_pnl']:,.2f} (losing)")
    
    logger.info("\n" + "="*80)
    logger.info("✅ JANUARY 2026 TESTING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
