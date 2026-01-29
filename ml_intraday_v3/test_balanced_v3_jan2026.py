#!/usr/bin/env python3
"""
Test Balanced V3 Model on January 2026 Data

Tests the newly trained balanced model and compares with old models.
"""

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.live_trading.replay import replay_session

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    # Keep this script readable; silence the live-stack spam.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(logging.ERROR)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )

    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    noisy = [
        "ml_intraday_v3.live_trading.execution_engine",
        "ml_intraday_v3.live_trading.feature_generator",
        "ml_intraday_v3.live_trading.replay",
        "ml_intraday_v3.monitoring.metrics_tracker",
    ]
    for name in noisy:
        logging.getLogger(name).setLevel(logging.ERROR)


def _write_replay_config_dir(base_config_dir: Path) -> Path:
    """
    Create a temp config directory for replay-only overrides.

    Supported env overrides:
      - REPLAY_PRIMARY_THRESHOLD (float): overrides signals.primary_threshold
      - REPLAY_PRIMARY_THRESHOLD_LONG (float): overrides signals.primary_threshold_long
      - REPLAY_PRIMARY_THRESHOLD_SHORT (float): overrides signals.primary_threshold_short
      - REPLAY_DISABLE_REGIME_FILTER (bool-ish): if true, sets regime_filter.enabled=false
    """
    import shutil
    from datetime import datetime

    out = (
        Path("ml_intraday_v3/backtest_results")
        / "_replay_cfg_tmp"
        / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)

    # Copy required config files.
    for name in ["features.yaml", "risk.yaml", "execution_spec.yaml", "live_trading.yaml"]:
        shutil.copy(base_config_dir / name, out / name)

    live_path = out / "live_trading.yaml"
    with open(live_path, "r") as f:
        live_cfg = yaml.safe_load(f) or {}

    # Apply overrides.
    if os.environ.get("REPLAY_PRIMARY_THRESHOLD"):
        thr = float(os.environ["REPLAY_PRIMARY_THRESHOLD"])
        live_cfg.setdefault("signals", {})
        live_cfg["signals"]["primary_threshold"] = thr

    if os.environ.get("REPLAY_PRIMARY_THRESHOLD_LONG"):
        thr = float(os.environ["REPLAY_PRIMARY_THRESHOLD_LONG"])
        live_cfg.setdefault("signals", {})
        live_cfg["signals"]["primary_threshold_long"] = thr

    if os.environ.get("REPLAY_PRIMARY_THRESHOLD_SHORT"):
        thr = float(os.environ["REPLAY_PRIMARY_THRESHOLD_SHORT"])
        live_cfg.setdefault("signals", {})
        live_cfg["signals"]["primary_threshold_short"] = thr

    if os.environ.get("REPLAY_ALLOWED_DIRECTIONS"):
        dirs = [
            d.strip().upper()
            for d in os.environ["REPLAY_ALLOWED_DIRECTIONS"].split(",")
            if d.strip()
        ]
        live_cfg.setdefault("signals", {})
        live_cfg["signals"]["allowed_directions"] = dirs

    disable_regime = os.environ.get("REPLAY_DISABLE_REGIME_FILTER", "").strip().lower()
    if disable_regime in {"1", "true", "yes", "y"}:
        live_cfg.setdefault("regime_filter", {})
        live_cfg["regime_filter"]["enabled"] = False

    with open(live_path, "w") as f:
        yaml.safe_dump(live_cfg, f, sort_keys=False)

    return out


def run_jan_2026_backtest(model_name: str, model_path: Path) -> dict:
    """Run backtest on Jan 2026 data."""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"TESTING: {model_name}")
    logger.info(f"{'='*80}")
    
    import shutil
    
    # Use existing Jan 2026 data
    run_dir = Path("ml_intraday_v3/backtest_results/jan_2026_test")
    if not run_dir.exists():
        logger.error(f"January 2026 data not found. Run test_jan_2026.py first.")
        return {'error': 'No Jan 2026 data'}
    
    # Copy model to walkforward structure
    wf_dir = run_dir / "walkforward" / "bar_size=5m" / "window_0"
    wf_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(model_path, wf_dir / "model_bundle.pkl")
    
    base_config_dir = Path("ml_intraday_v3/configs")
    config_dir = _write_replay_config_dir(base_config_dir)
    
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
        
        # Determine columns
        if 'direction' in trades.columns:
            direction_col = 'direction'
            long_val = 'LONG'
            short_val = 'SHORT'
        elif 'side' in trades.columns:
            direction_col = 'side'
            long_val = 1
            short_val = -1
        else:
            return {'model': model_name, 'error': 'No direction column'}
        
        pnl_col = None
        for col in ['pnl', 'pnl_usd', 'realized_pnl']:
            if col in trades.columns:
                pnl_col = col
                break
        
        if pnl_col is None:
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
    _configure_logging()
    logger.info("="*80)
    logger.info("JANUARY 2026 MODEL COMPARISON TEST")
    logger.info("="*80)
    
    models_dir = Path("ml_intraday_v3/models/saved")
    
    models_to_test = [
        {
            'name': 'CUSUM_BIDIR_20260127 (Ensemble)',
            'path': models_dir / "model_bundle_cusum_bidir_20260127.pkl",
        },
        {
            'name': 'CUSUM_BIDIR_RTH_20260127 (Ensemble)',
            'path': models_dir / "model_bundle_cusum_bidir_rth_20260127.pkl",
        },
        {
            'name': 'DUALSIDE_CUSUM_20260127 (Ensemble)',
            'path': models_dir / "model_bundle_dualside_cusum_20260127.pkl",
        },
        {
            'name': 'BALANCED_V3 (New)',
            'path': models_dir / "model_bundle_balanced_v3.pkl",
        },
        {
            'name': 'RETRAINED_CLEAN',
            'path': models_dir / "model_bundle_retrained_clean.pkl",
        },
        {
            'name': 'OLD_BASELINE',
            'path': models_dir / "model_bundle_OLD_BASELINE.pkl",
        },
    ]

    only_models_env = os.environ.get("ONLY_MODELS", "").strip()
    if only_models_env:
        allowed = {s.strip() for s in only_models_env.split(",") if s.strip()}
        models_to_test = [m for m in models_to_test if m["name"] in allowed]
    
    results = []
    
    for model in models_to_test:
        if not model['path'].exists():
            logger.warning(f"\nModel not found: {model['path']}")
            continue
        
        result = run_jan_2026_backtest(model['name'], model['path'])
        results.append(result)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("JANUARY 2026 COMPARISON RESULTS")
    logger.info("="*80)
    
    df = pd.DataFrame(results)
    
    logger.info(f"\n{'Model':<30} {'Trades':>8} {'LONG%':>8} {'SHORT%':>8} {'Win%':>8} {'P&L':>12}")
    logger.info("-" * 90)
    
    for _, row in df.iterrows():
        if 'error' in row and pd.notna(row.get('error')):
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
    logger.info("ANALYSIS")
    logger.info("="*80)
    
    for _, row in df.iterrows():
        if 'error' in row and pd.notna(row.get('error')):
            continue
        
        model = row['model']
        
        logger.info(f"\n{model}:")
        
        # Direction
        if row['total_trades'] == 0:
            logger.info(f"  No trades generated")
        elif row['long_pct'] > 90:
            logger.info(f"  Extreme LONG bias ({row['long_pct']:.1f}%) - NOT FIXED")
        elif row['short_pct'] > 90:
            logger.info(f"  Extreme SHORT bias ({row['short_pct']:.1f}%)")
        elif 30 <= row['long_pct'] <= 70:
            logger.info(f"  Balanced direction ({row['long_pct']:.1f}% LONG, {row['short_pct']:.1f}% SHORT)")
        else:
            logger.info(f"  {row['long_pct']:.1f}% LONG, {row['short_pct']:.1f}% SHORT")
        
        # Performance
        if row['win_rate'] > 50:
            logger.info(f"  Win rate: {row['win_rate']:.1f}% - Profitable")
        elif row['win_rate'] > 40:
            logger.info(f"  Win rate: {row['win_rate']:.1f}% - Marginal")
        else:
            logger.info(f"  Win rate: {row['win_rate']:.1f}% - Poor")
        
        if row['total_pnl'] > 500:
            logger.info(f"  P&L: ${row['total_pnl']:,.2f} - Strong performance")
        elif row['total_pnl'] > 0:
            logger.info(f"  P&L: ${row['total_pnl']:,.2f} - Positive")
        else:
            logger.info(f"  P&L: ${row['total_pnl']:,.2f} - Losing")
    
    logger.info("\n" + "="*80)
    logger.info("TESTING COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
