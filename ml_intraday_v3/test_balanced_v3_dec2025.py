#!/usr/bin/env python3
"""
Backtest BALANCED_V3 on Dec 2025 (holdout) via replay_session.

This creates a small replay run_dir (bars.parquet + label_schema.json + walkforward bundle)
and runs the same live-stack replay used by the Jan 2026 test.
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add project root
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.live_trading.replay import replay_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("Expected DatetimeIndex")
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def _write_replay_config_dir(base_config_dir: Path) -> Path:
    """
    Create a temp config directory for replay-only overrides.

    Supported env overrides:
      - REPLAY_PRIMARY_THRESHOLD (float): overrides signals.primary_threshold
      - REPLAY_PRIMARY_THRESHOLD_LONG (float): overrides signals.primary_threshold_long
      - REPLAY_PRIMARY_THRESHOLD_SHORT (float): overrides signals.primary_threshold_short
      - REPLAY_DISABLE_REGIME_FILTER (bool-ish): if true, sets regime_filter.enabled=false
    """
    from datetime import datetime

    out = (
        Path("ml_intraday_v3/backtest_results")
        / "_replay_cfg_tmp"
        / datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    )
    out.mkdir(parents=True, exist_ok=True)

    for name in ["features.yaml", "risk.yaml", "execution_spec.yaml", "live_trading.yaml"]:
        shutil.copy(base_config_dir / name, out / name)

    live_path = out / "live_trading.yaml"
    with open(live_path, "r") as f:
        live_cfg = yaml.safe_load(f) or {}

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


def main() -> int:
    logger.info("=" * 80)
    logger.info("DECEMBER 2025 BACKTEST (BALANCED_V3) - LIVE STACK REPLAY")
    logger.info("=" * 80)

    # Load Dec 2025 bars from the processed H5 (RTH)
    data_path = Path("data/processed/mes_bars_databento_rth.h5")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    bars = pd.read_hdf(data_path, key="bars_5min")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars = bars.set_index("timestamp").sort_index()
    bars = _ensure_utc_index(bars)

    start = pd.Timestamp("2025-12-01", tz="UTC")
    end = pd.Timestamp("2025-12-19", tz="UTC")  # dataset ends ~Dec 18 in this file
    bars_dec = bars[(bars.index >= start) & (bars.index < end)].copy()
    if bars_dec.empty:
        raise ValueError("No Dec 2025 bars found in H5 slice")

    logger.info(f"Bars: {len(bars_dec):,} ({bars_dec.index.min()} to {bars_dec.index.max()})")

    # Build a run_dir compatible with replay_session
    run_dir = Path("ml_intraday_v3/backtest_results/dec_2025_test")
    bar_dir = run_dir / "bar_size=5m"
    wf_dir = run_dir / "walkforward" / "bar_size=5m" / "window_0"
    bar_dir.mkdir(parents=True, exist_ok=True)
    wf_dir.mkdir(parents=True, exist_ok=True)

    bars_path = bar_dir / "bars.parquet"
    bars_dec.to_parquet(bars_path)

    # Keep schema consistent with current labeling/execution expectations.
    label_schema = {"stop_multiple": 1.5, "target_multiple": 2.5, "atr_period": 14, "atr_bar_size": "5m"}
    with open(bar_dir / "label_schema.json", "w") as f:
        json.dump(label_schema, f, indent=2)

    # Copy model bundle into walkforward location
    model_path = Path(os.environ.get("MODEL_BUNDLE_PATH", "ml_intraday_v3/models/saved/model_bundle_balanced_v3.pkl"))
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model bundle: {model_path}")
    shutil.copy(model_path, wf_dir / "model_bundle.pkl")

    # Run replay (with optional config overrides)
    base_config_dir = Path("ml_intraday_v3/configs")
    config_dir = _write_replay_config_dir(base_config_dir)
    replay_session(run_dir=run_dir, config_dir=config_dir, bar_size="5m")

    # Parse latest trade file written to logs/
    trade_files = sorted(Path("logs").glob("trades_*.csv"), key=os.path.getctime)
    if not trade_files:
        logger.warning("No trade logs found in logs/")
        return 1

    trades = pd.read_csv(trade_files[-1])
    if trades.empty:
        logger.info("No trades generated.")
        return 0

    # Columns vary slightly depending on tracker version
    pnl_col = next((c for c in ["pnl", "pnl_usd", "realized_pnl"] if c in trades.columns), None)
    if pnl_col is None:
        raise ValueError("Trade log missing P&L column")

    direction_col = "direction" if "direction" in trades.columns else ("side" if "side" in trades.columns else None)
    if direction_col is None:
        raise ValueError("Trade log missing direction/side column")

    long_val = "LONG" if direction_col == "direction" else 1
    short_val = "SHORT" if direction_col == "direction" else -1

    winners = trades[trades[pnl_col] > 0]
    total_trades = len(trades)
    long_trades = int((trades[direction_col] == long_val).sum())
    short_trades = int((trades[direction_col] == short_val).sum())
    win_rate = 100.0 * len(winners) / total_trades if total_trades else 0.0
    total_pnl = float(trades[pnl_col].sum())

    logger.info("")
    logger.info(f"Total trades: {total_trades}")
    logger.info(f"LONG: {long_trades} ({100.0 * long_trades / total_trades:.1f}%)")
    logger.info(f"SHORT: {short_trades} ({100.0 * short_trades / total_trades:.1f}%)")
    logger.info(f"Win rate: {win_rate:.1f}%")
    logger.info(f"Total P&L: ${total_pnl:,.2f}")
    logger.info(f"Avg trade: ${total_pnl / total_trades:,.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

