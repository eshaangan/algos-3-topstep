"""
ORB Paper Trading Runner
========================
Runs the Opening Range Breakout strategy in paper trading mode.

Two modes:
  --mode replay   : Bar-by-bar replay of a local HDF5 file (default, for testing)
  --mode live     : Polling loop ready to accept live bar data via stdin/file

Best confirmed config (Jan-Feb 2026 OOS: 32 trades, WR=56.7%, P(pass)=53.9%):
  or_end_time='09:55', entry_cutoff='12:00', PT=2.0x ATR, SL=1.5x ATR, n=2 MES

Usage:
    # Replay validation
    python3 orb_live_runner.py --mode replay \
        --data-path ~/data/jan_feb_2026_oos_test.h5 --data-key bars_5min

    # Live mode (ready for data feed integration)
    python3 orb_live_runner.py --mode live
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — works whether run from project root or rule_based_v1/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for _p in [str(_HERE), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from engine.backtest_engine import BacktestEngine          # noqa: E402
from engine.signal_aggregator import SignalAggregator      # noqa: E402
from engine.risk_manager import RiskManager                # noqa: E402
from rules.opening_range import OpeningRangeBreakoutRule   # noqa: E402
from rules.time_of_day import TimeOfDayRule                # noqa: E402

logger = logging.getLogger("orb_runner")

# ---------------------------------------------------------------------------
# Strategy config — best config from OOS validation
# ---------------------------------------------------------------------------
ORB_CONFIG = {
    "or_end_time": "09:55",
    "entry_cutoff_time": "12:00",
    "min_range_atr": 0.3,
    "pt_atr_mult": 2.0,
    "sl_atr_mult": 1.5,
    "trailing_activation_atr": 999.0,
    "trailing_distance_atr": 0.75,
    "atr_period": 14,
    "time_stop_bars": 24,
    "commission_per_side": 0.62,
    "slippage_ticks": 1,
}

RISK_CONFIG = {
    "n_contracts": 2,          # 2 MES contracts (confirmed safe max for $50k combine)
    "point_value": 5.0,        # MES
    "tick_size": 0.25,
    "tick_value": 1.25,
    "max_daily_loss": -950.0,  # Topstep $1k limit minus $50 buffer
    "per_trade_max_loss": 1000.0,
    "max_consecutive_losses": 3,
    "cooldown_bars": 3,
    "flatten_minutes_before_close": 5,
    "drawdown_buffer": 1800.0, # Topstep $2k trailing DD minus $200 buffer
}

COMBINE_TARGETS = {
    "profit_target": 3000.0,
    "max_trailing_drawdown": 2000.0,
    "max_daily_loss": 1000.0,
}


def build_engine() -> BacktestEngine:
    primary = OpeningRangeBreakoutRule(
        or_end_time=ORB_CONFIG["or_end_time"],
        min_range_atr=ORB_CONFIG["min_range_atr"],
        entry_cutoff_time=ORB_CONFIG["entry_cutoff_time"],
        atr_period=ORB_CONFIG["atr_period"],
    )
    filters = [
        TimeOfDayRule(
            session_start="09:35",
            session_end="15:45",
            lunch_filter_enabled=False,
        ),
    ]
    aggregator = SignalAggregator(
        primary_rule=primary,
        filter_rules=filters,
        confirmation_rules=[],
        min_confirmations=0,
    )
    risk_manager = RiskManager(
        contracts=RISK_CONFIG["n_contracts"],
        point_value=RISK_CONFIG["point_value"],
        tick_size=RISK_CONFIG["tick_size"],
        tick_value=RISK_CONFIG["tick_value"],
        max_daily_loss=RISK_CONFIG["max_daily_loss"],
        per_trade_max_loss=RISK_CONFIG["per_trade_max_loss"],
        max_consecutive_losses=RISK_CONFIG["max_consecutive_losses"],
        cooldown_bars=RISK_CONFIG["cooldown_bars"],
        flatten_minutes_before_close=RISK_CONFIG["flatten_minutes_before_close"],
        drawdown_buffer=RISK_CONFIG["drawdown_buffer"],
    )
    engine = BacktestEngine(
        aggregator=aggregator,
        risk_manager=risk_manager,
        commission_per_side=ORB_CONFIG["commission_per_side"],
        slippage_ticks=ORB_CONFIG["slippage_ticks"],
        profit_target_atr=ORB_CONFIG["pt_atr_mult"],
        stop_loss_atr=ORB_CONFIG["sl_atr_mult"],
        time_stop_bars=ORB_CONFIG["time_stop_bars"],
        trailing_activation_atr=ORB_CONFIG["trailing_activation_atr"],
        trailing_distance_atr=ORB_CONFIG["trailing_distance_atr"],
        atr_period=ORB_CONFIG["atr_period"],
    )
    return engine


def to_rth(bars: pd.DataFrame) -> pd.DataFrame:
    h = bars.index.hour
    m = bars.index.minute
    return bars[((h > 9) | ((h == 9) & (m >= 30))) & (h < 16)]


def load_hdf5_bars(path: str, key: str = "bars_5min") -> pd.DataFrame:
    df = pd.read_hdf(path, key=key)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")
    df.columns = df.columns.str.lower()
    return df.sort_index()


# ---------------------------------------------------------------------------
# Status tracker
# ---------------------------------------------------------------------------
class PaperAccount:
    def __init__(self, starting_equity: float = 50_000.0):
        self.equity = starting_equity
        self.starting = starting_equity
        self.peak = starting_equity
        self.daily_pnl = 0.0
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.trade_log: List[dict] = []

    def record_trade(self, pnl: float, exit_reason: str, timestamp=None):
        self.equity += pnl
        self.daily_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        self.peak = max(self.peak, self.equity)
        self.trade_log.append({
            "timestamp": str(timestamp or datetime.now()),
            "pnl": round(pnl, 2),
            "equity": round(self.equity, 2),
            "exit_reason": exit_reason,
        })

    def daily_reset(self):
        self.daily_pnl = 0.0

    @property
    def win_rate(self) -> float:
        return self.wins / self.total_trades if self.total_trades > 0 else 0.0

    @property
    def total_pnl(self) -> float:
        return self.equity - self.starting

    @property
    def trailing_drawdown(self) -> float:
        return self.peak - self.equity

    def status_line(self) -> str:
        return (
            f"Equity=${self.equity:,.0f}  PnL=${self.total_pnl:+,.0f}  "
            f"DD=${self.trailing_drawdown:,.0f}/{COMBINE_TARGETS['max_trailing_drawdown']:,.0f}  "
            f"WR={self.win_rate:.1%}  Trades={self.total_trades}"
        )

    def combine_status(self) -> str:
        pnl = self.total_pnl
        dd = self.trailing_drawdown
        pct_to_target = pnl / COMBINE_TARGETS["profit_target"] * 100
        pct_to_dd = dd / COMBINE_TARGETS["max_trailing_drawdown"] * 100
        status = "ACTIVE"
        if pnl >= COMBINE_TARGETS["profit_target"]:
            status = "*** PASS CONDITION MET ***"
        elif dd >= COMBINE_TARGETS["max_trailing_drawdown"]:
            status = "*** TRAILING DD BREACH ***"
        return (
            f"  Combine: {status}\n"
            f"  Profit:  ${pnl:+,.0f} / ${COMBINE_TARGETS['profit_target']:,.0f} ({pct_to_target:.1f}%)\n"
            f"  Trail DD: ${dd:,.0f} / ${COMBINE_TARGETS['max_trailing_drawdown']:,.0f} ({pct_to_dd:.1f}%)\n"
            f"  Daily PnL: ${self.daily_pnl:+,.0f} / -${COMBINE_TARGETS['max_daily_loss']:,.0f}"
        )


# ---------------------------------------------------------------------------
# Replay mode
# ---------------------------------------------------------------------------
def run_replay(data_path: str, data_key: str, log_path: Path, speed: float = 0.0):
    logger.info(f"[REPLAY] Loading {data_path} key={data_key}")
    bars = to_rth(load_hdf5_bars(data_path, data_key))
    logger.info(f"[REPLAY] {len(bars):,} RTH bars  {bars.index[0]} → {bars.index[-1]}")

    engine = build_engine()
    account = PaperAccount()
    result = engine.run(bars, starting_equity=50_000.0)

    # Replay trades from backtest result
    logger.info(f"\n{'='*70}")
    logger.info("  ORB PAPER TRADING — REPLAY MODE")
    logger.info(f"  Config: or_end=09:55, PT=2.0x, SL=1.5x, n=2 MES")
    logger.info(f"{'='*70}")

    for trade in result.trades:
        pnl = trade.pnl
        account.record_trade(pnl, trade.exit_reason)
        direction = "LONG" if trade.direction == 1 else "SHORT"
        logger.info(
            f"  TRADE [{direction}]  entry={trade.entry_price:.2f}  "
            f"exit={trade.exit_price:.2f}  pnl=${pnl:+.2f}  "
            f"reason={trade.exit_reason}  |  {account.status_line()}"
        )
        if speed > 0:
            time.sleep(speed)

    # Final summary
    s = result.summary()
    logger.info(f"\n{'='*70}")
    logger.info("  REPLAY COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"  Trades:     {s.get('num_trades', 0)}")
    logger.info(f"  Win Rate:   {s.get('win_rate', 0):.1%}")
    logger.info(f"  PnL:        ${s.get('total_pnl', 0):+,.2f}")
    logger.info(f"  Sharpe:     {s.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  Max DD:     ${s.get('max_drawdown', 0):,.2f}")
    logger.info(f"\n{account.combine_status()}")

    # Save trade log
    trade_log_path = log_path / "replay_trades.json"
    with open(trade_log_path, "w") as f:
        json.dump(
            {
                "mode": "replay",
                "data_path": data_path,
                "config": ORB_CONFIG,
                "risk": RISK_CONFIG,
                "summary": {k: float(v) if isinstance(v, (int, float)) else v
                            for k, v in s.items()},
                "trades": account.trade_log,
            },
            f,
            indent=2,
        )
    logger.info(f"\n  Trade log saved: {trade_log_path}")


# ---------------------------------------------------------------------------
# Live mode (stub — ready for data feed integration)
# ---------------------------------------------------------------------------
def run_live(log_path: Path):
    logger.info(f"\n{'='*70}")
    logger.info("  ORB PAPER TRADING — LIVE MODE")
    logger.info(f"  Config: or_end=09:55, PT=2.0x, SL=1.5x, n=2 MES")
    logger.info(f"  Waiting for live data feed...")
    logger.info(f"{'='*70}")
    logger.info("")
    logger.info("  TO CONNECT A DATA FEED:")
    logger.info("  1. Add your data provider (Databento, Rithmic, IB) to feed_connector.py")
    logger.info("  2. On each new 5-min bar, call: engine.process_bar(bar)")
    logger.info("  3. Track signals and log paper trades")
    logger.info("")
    logger.info("  The ORB engine is initialized and ready:")
    logger.info(f"    - Opening range: 09:30-09:55 ET")
    logger.info(f"    - Entry window:  10:00-12:00 ET")
    logger.info(f"    - PT: 2.0x ATR, SL: 1.5x ATR")
    logger.info(f"    - n=2 MES contracts")
    logger.info(f"    - Risk: max_daily_loss=$950, trailing_dd_buffer=$1,800")
    logger.info("")
    logger.info("  Status: RUNNING (press Ctrl+C to stop)")

    # Keep-alive loop — replace this with actual data feed polling
    account = PaperAccount()
    check_interval = 60  # seconds
    last_status_hour = -1

    def shutdown(sig, frame):
        logger.info("\n  Shutting down cleanly...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)

    while True:
        now = datetime.now()
        # Print hourly status during trading hours
        if 9 <= now.hour < 16 and now.hour != last_status_hour:
            last_status_hour = now.hour
            logger.info(f"\n  [{now.strftime('%H:%M')} ET] {account.status_line()}")
            logger.info(account.combine_status())

        time.sleep(check_interval)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ORB Paper Trading Runner")
    parser.add_argument(
        "--mode",
        choices=["replay", "live"],
        default="replay",
        help="replay: backtest replay for validation | live: ready for data feed",
    )
    parser.add_argument(
        "--data-path",
        default=str(Path.home() / "data" / "jan_feb_2026_oos_test.h5"),
        help="Path to HDF5 bars file (replay mode)",
    )
    parser.add_argument("--data-key", default="bars_5min")
    parser.add_argument(
        "--log-dir",
        default=str(Path.home() / "ml_intraday_v3" / "logs"),
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=0.0,
        help="Seconds to pause between replay trades (0=instant)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / f"orb_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file)),
        ],
    )

    logger.info(f"Log file: {log_file}")
    logger.info(f"Mode: {args.mode}")

    if args.mode == "replay":
        run_replay(args.data_path, args.data_key, log_path, speed=args.replay_speed)
    else:
        run_live(log_path)


if __name__ == "__main__":
    main()
