"""VWAP Mean Reversion Backtest — MNQ 2026 YTD.

Grid-sweeps entry_distance_atr and time_end to find optimal parameters.
Uses 5-min RTH bars resampled from 1-min Databento data.

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/backtest_vwap.py
    python rule_based_v1/diagnostics/backtest_vwap.py --backtest-only
    python rule_based_v1/diagnostics/backtest_vwap.py --entry-dist 0.75 --time-end 13:00
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from engine.signal_aggregator import SignalAggregator
from engine.risk_manager import RiskManager, TradeRecord
from rules.vwap_mean_reversion import VWAPMeanReversionRule

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data paths — prefer the freshest available file
# ---------------------------------------------------------------------------
_CANDIDATES = [
    ROOT / "data" / "processed" / "mnq_2026ytd_databento_5min_rth.h5",
    ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5",
    ROOT / "data" / "processed" / "mnq_aug25_apr26_1min_eth.h5",   # will be resampled
]
DATA_1MIN_ETH = ROOT / "data" / "processed" / "mnq_2026ytd_databento_1min_eth.h5"
RESULTS_PATH  = ROOT / "rule_based_v1" / "diagnostics" / "vwap_backtest_results.json"

# ---------------------------------------------------------------------------
# Fixed sim parameters (mirror ORB backtest)
# ---------------------------------------------------------------------------
POINT_VALUE   = 2.0
TICK_SIZE     = 0.25
TICK_VALUE    = 0.50
COMMISSION    = 0.62   # per side per contract
SLIPPAGE_TICKS = 1
N_CONTRACTS   = 2
MAX_DAILY_LOSS = -700.0
DRAWDOWN_BUFFER = 1_950.0
STARTING_EQUITY = 50_000.0


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def _load_5min_rth() -> pd.DataFrame:
    """Return 5-min RTH bars — from cached file or resampled from 1-min."""
    for path in _CANDIDATES:
        if path.exists():
            key = "bars_5min" if "5min" in path.name else "bars"
            try:
                df = pd.read_hdf(str(path), key=key)
                if df.index.tz is None:
                    df.index = df.index.tz_localize("US/Eastern")
                else:
                    df.index = df.index.tz_convert("US/Eastern")
                # If it's 1-min, resample to 5-min RTH
                if "1min" in path.name:
                    df = _to_5min_rth(df)
                logger.info(f"Loaded {len(df):,} 5-min bars from {path.name}")
                return df
            except Exception as e:
                logger.warning(f"Failed to load {path.name}: {e}")

    # Fallback: fetch from 1-min ETH file
    if DATA_1MIN_ETH.exists():
        try:
            df1 = pd.read_hdf(str(DATA_1MIN_ETH), key="bars_1min")
            if df1.index.tz is None:
                df1.index = df1.index.tz_localize("UTC").tz_convert("US/Eastern")
            return _to_5min_rth(df1)
        except Exception as e:
            logger.warning(f"Failed to load 1-min ETH: {e}")

    raise FileNotFoundError(
        "No MNQ 5-min data found. Run fetch_backtest_2026ytd.py first."
    )


def _to_5min_rth(df: pd.DataFrame) -> pd.DataFrame:
    df5 = df.resample("5min").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
    ).dropna(subset=["open"])
    if df5.index.tz is None:
        df5.index = df5.index.tz_localize("US/Eastern")
    else:
        df5.index = df5.index.tz_convert("US/Eastern")
    rth = (
        (df5.index.hour > 9) | ((df5.index.hour == 9) & (df5.index.minute >= 30))
    ) & (df5.index.hour < 16)
    return df5.loc[rth]


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------
@dataclass
class Position:
    direction: int
    entry_price: float
    entry_bar_idx: int
    stop_loss: float
    profit_target: float
    time_stop_bar: int
    atr_at_entry: float


def _slip(price: float, direction: int, is_entry: bool) -> float:
    s = SLIPPAGE_TICKS * TICK_SIZE
    return price + s * direction if is_entry else price - s * direction


def _pnl(entry: float, exit_: float, direction: int) -> float:
    raw = (exit_ - entry) * direction * N_CONTRACTS * POINT_VALUE
    return raw - 2 * COMMISSION * N_CONTRACTS


def _check_exit(pos: Position, bar: pd.Series, idx: int) -> tuple[bool, float, str]:
    h, l, c = bar["high"], bar["low"], bar["close"]
    if idx >= pos.time_stop_bar:
        return True, _slip(c, pos.direction, False), "time_stop"
    if pos.direction == 1:
        if l <= pos.stop_loss:
            return True, _slip(pos.stop_loss, 1, False), "stop_loss"
        if h >= pos.profit_target:
            return True, _slip(pos.profit_target, 1, False), "profit_target"
    else:
        if h >= pos.stop_loss:
            return True, _slip(pos.stop_loss, -1, False), "stop_loss"
        if l <= pos.profit_target:
            return True, _slip(pos.profit_target, -1, False), "profit_target"
    return False, 0.0, ""


# ---------------------------------------------------------------------------
# Single backtest run
# ---------------------------------------------------------------------------
def run_backtest(
    bars: pd.DataFrame,
    entry_distance_atr: float = 1.0,
    max_distance_atr: float = 3.0,
    time_start: str = "10:30",
    time_end: str = "13:30",
    long_only: bool = True,
    max_move_from_open_atr: float = 1.0,
    pt_mult: float = 2.0,
    sl_mult: float = 1.5,
    time_stop_bars: int = 12,
    max_vwap_trades_per_day: int = 2,
) -> tuple[list, list, list, dict]:
    rule = VWAPMeanReversionRule(
        entry_distance_atr=entry_distance_atr,
        max_distance_atr=max_distance_atr,
        time_start=time_start,
        time_end=time_end,
        long_only=long_only,
        max_move_from_open_atr=max_move_from_open_atr,
    )
    rm = RiskManager(
        contracts=N_CONTRACTS, point_value=POINT_VALUE,
        tick_size=TICK_SIZE, tick_value=TICK_VALUE,
        max_daily_loss=MAX_DAILY_LOSS,
        per_trade_max_loss=1000.0,
        max_consecutive_losses=10,
        cooldown_bars=3,
        drawdown_buffer=DRAWDOWN_BUFFER,
    )
    rm.reset_all(STARTING_EQUITY)

    min_bars = rule.required_bars()
    pos: Position | None = None
    trades: list[TradeRecord] = []
    eq_vals, eq_times = [STARTING_EQUITY], [bars.index[0]]
    equity = STARTING_EQUITY
    cur_date = None
    daily_pnl: dict = {}
    vwap_trades_today = 0

    for i in range(min_bars, len(bars)):
        bar = bars.iloc[i]
        bt = bars.index[i]
        bt_et = bt.tz_convert("US/Eastern") if bt.tzinfo else bt
        bdate = bt_et.date()

        if cur_date is not None and bdate != cur_date:
            daily_pnl[cur_date] = rm.daily_pnl
            rm.reset_daily()
            vwap_trades_today = 0
        cur_date = bdate
        rm.tick_bar()

        is_last = (i + 1 >= len(bars)) or (
            (bars.index[i + 1].tz_convert("US/Eastern")
             if bars.index[i + 1].tzinfo else bars.index[i + 1]).date() != bdate
        )
        sess_close = is_last or (bt_et.hour == 15 and bt_et.minute >= 55)

        if pos is not None:
            exited, exit_p, reason = _check_exit(pos, bar, i)
            if exited or sess_close:
                if sess_close and not exited:
                    exit_p = _slip(bar["close"], pos.direction, False)
                    reason = "session_close"
                p = _pnl(pos.entry_price, exit_p, pos.direction)
                tr = TradeRecord(
                    entry_bar=pos.entry_bar_idx, exit_bar=i,
                    direction=pos.direction, entry_price=pos.entry_price,
                    exit_price=exit_p, pnl=p, exit_reason=reason,
                )
                trades.append(tr)
                rm.record_trade(tr)
                equity += p
                eq_vals.append(equity)
                eq_times.append(bt)
                pos = None

        if pos is None and not sess_close and vwap_trades_today < max_vwap_trades_per_day:
            ok, _ = rm.can_trade()
            if ok:
                lookback = bars.iloc[max(0, i - min_bars + 1): i + 1]
                signal = rule.evaluate(lookback)
                if signal.has_signal:
                    cur_atr = signal.metadata.get("atr", 0.0)
                    if cur_atr > 0:
                        ep = _slip(bar["close"], signal.direction, True)
                        sl = (ep - sl_mult * cur_atr) if signal.direction == 1 else (ep + sl_mult * cur_atr)
                        pt = (ep + pt_mult * cur_atr) if signal.direction == 1 else (ep - pt_mult * cur_atr)
                        pos = Position(
                            direction=signal.direction, entry_price=ep,
                            entry_bar_idx=i, stop_loss=sl, profit_target=pt,
                            time_stop_bar=i + time_stop_bars, atr_at_entry=cur_atr,
                        )
                        vwap_trades_today += 1

    if cur_date and cur_date not in daily_pnl:
        daily_pnl[cur_date] = rm.daily_pnl

    return trades, eq_vals, eq_times, daily_pnl


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _metrics(trades, eq_vals, eq_times, daily_pnl, config: dict) -> dict:
    if not trades:
        return {"config": config, "num_trades": 0, "win_rate": 0.0, "sharpe": 0.0}

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    total = sum(t.pnl for t in trades)
    gp = sum(t.pnl for t in wins)
    gl = abs(sum(t.pnl for t in losses))

    eq = pd.Series(eq_vals, index=eq_times)
    max_dd = float((eq - eq.cummax()).min())

    daily = pd.Series(daily_pnl)
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 1 and daily.std() > 0 else 0.0

    n_days = len(daily_pnl)
    reasons = defaultdict(int)
    for t in trades:
        reasons[t.exit_reason] += 1
    reason_pct = {k: round(v / len(trades) * 100, 1) for k, v in reasons.items()}

    return {
        "config": config,
        "num_trades": len(trades),
        "trades_per_day": round(len(trades) / max(n_days, 1), 2),
        "win_rate": round(len(wins) / len(trades), 3),
        "total_pnl": round(total, 2),
        "avg_win": round(gp / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gl / len(losses), 2) if losses else 0.0,
        "profit_factor": round(gp / gl, 3) if gl > 0 else None,
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "exit_reasons": reason_pct,
        "n_days": n_days,
    }


# ---------------------------------------------------------------------------
# Grid sweep + print
# ---------------------------------------------------------------------------
def run_grid_sweep(bars: pd.DataFrame) -> list[dict]:
    grid = [
        (entry_dist, time_end)
        for entry_dist in [0.5, 0.75, 1.0, 1.25, 1.5]
        for time_end in ["12:00", "13:00", "13:30"]
    ]

    results = []
    for entry_dist, time_end in grid:
        cfg = {"entry_dist": entry_dist, "time_end": time_end, "pt": 2.0, "sl": 1.5}
        trades, eq_v, eq_t, dpnl = run_backtest(
            bars,
            entry_distance_atr=entry_dist,
            time_end=time_end,
            pt_mult=2.0,
            sl_mult=1.5,
        )
        m = _metrics(trades, eq_v, eq_t, dpnl, cfg)
        results.append(m)
        logger.info(
            f"entry_dist={entry_dist:.2f}  time_end={time_end}  "
            f"n={m['num_trades']}  WR={m['win_rate']:.1%}  "
            f"PnL=${m['total_pnl']:,.0f}  Sharpe={m['sharpe']:.2f}  "
            f"DD=${m['max_drawdown']:,.0f}"
        )

    return results


def print_results(m: dict) -> None:
    cfg = m["config"]
    print(f"\n{'='*60}")
    print(f"  VWAP Mean Reversion  |  entry_dist={cfg['entry_dist']}x  "
          f"time_end={cfg['time_end']}  PT={cfg['pt']}x  SL={cfg['sl']}x")
    print(f"{'='*60}")
    print(f"  Trades       : {m['num_trades']}  ({m['trades_per_day']:.2f}/day over {m['n_days']} days)")
    print(f"  Win Rate     : {m['win_rate']:.1%}")
    print(f"  Total PnL    : ${m['total_pnl']:,.2f}")
    print(f"  Avg Win/Loss : ${m['avg_win']:,.2f} / ${m['avg_loss']:,.2f}")
    print(f"  Profit Factor: {m['profit_factor']}")
    print(f"  Sharpe       : {m['sharpe']:.2f}")
    print(f"  Max Drawdown : ${m['max_drawdown']:,.2f}")
    print(f"  Exit reasons : {m['exit_reasons']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VWAP Mean Reversion Backtest")
    parser.add_argument("--backtest-only", action="store_true")
    parser.add_argument("--sweep", action="store_true", help="Run full grid sweep")
    parser.add_argument("--entry-dist", type=float, default=1.0)
    parser.add_argument("--time-start", default="10:30")
    parser.add_argument("--time-end", default="13:30")
    parser.add_argument("--bidirectional", action="store_true", help="Test long+short instead of live long-only mode")
    parser.add_argument("--max-move-from-open-atr", type=float, default=1.0)
    parser.add_argument("--pt", type=float, default=2.0)
    parser.add_argument("--sl", type=float, default=1.5)
    parser.add_argument("--time-stop", type=int, default=12)
    parser.add_argument("--max-trades", type=int, default=2)
    parser.add_argument("--start", default=None, help="Inclusive start date, e.g. 2026-04-01")
    parser.add_argument("--end", default=None, help="Exclusive end date, e.g. 2026-05-01")
    args = parser.parse_args()

    bars = _load_5min_rth()
    if args.start:
        bars = bars.loc[bars.index >= pd.Timestamp(args.start, tz="US/Eastern")]
    if args.end:
        bars = bars.loc[bars.index < pd.Timestamp(args.end, tz="US/Eastern")]
    logger.info(f"Data: {bars.index[0].date()} → {bars.index[-1].date()}  ({len(bars):,} bars)")

    if args.sweep:
        results = run_grid_sweep(bars)
        # Sort by Sharpe
        results.sort(key=lambda r: r["sharpe"], reverse=True)
        print(f"\n{'='*70}")
        print("  GRID SWEEP — sorted by Sharpe")
        print(f"{'='*70}")
        print(f"  {'entry_dist':>10}  {'time_end':>8}  {'n':>5}  {'WR':>6}  "
              f"{'PnL':>8}  {'Sharpe':>7}  {'DD':>8}")
        print(f"  {'-'*64}")
        for m in results:
            cfg = m["config"]
            wr = m["win_rate"]
            print(f"  {cfg['entry_dist']:>10.2f}  {cfg['time_end']:>8}  "
                  f"{m['num_trades']:>5}  {wr:>6.1%}  "
                  f"${m['total_pnl']:>7,.0f}  {m['sharpe']:>7.2f}  "
                  f"${m['max_drawdown']:>7,.0f}")

        best = results[0]
        print_results(best)
        with open(RESULTS_PATH, "w") as f:
            json.dump({"sweep": results, "best": best}, f, indent=2, default=str)
        logger.info(f"Saved → {RESULTS_PATH}")
        return

    # Single run
    trades, eq_vals, eq_times, daily_pnl = run_backtest(
        bars,
        entry_distance_atr=args.entry_dist,
        time_start=args.time_start,
        time_end=args.time_end,
        long_only=not args.bidirectional,
        max_move_from_open_atr=args.max_move_from_open_atr,
        pt_mult=args.pt,
        sl_mult=args.sl,
        time_stop_bars=args.time_stop,
        max_vwap_trades_per_day=args.max_trades,
    )
    cfg = {"entry_dist": args.entry_dist, "time_end": args.time_end,
           "pt": args.pt, "sl": args.sl}
    m = _metrics(trades, eq_vals, eq_times, daily_pnl, cfg)
    print_results(m)

    # Per-trade detail
    print(f"\n  {'#':<4} {'Dir':<6} {'Entry':>9} {'Exit':>9} {'PnL':>8}  Reason")
    print(f"  {'-'*54}")
    for j, t in enumerate(trades, 1):
        d = "LONG" if t.direction == 1 else "SHORT"
        print(f"  {j:<4} {d:<6} {t.entry_price:>9.2f} {t.exit_price:>9.2f} ${t.pnl:>7,.2f}  {t.exit_reason}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(m, f, indent=2, default=str)
    logger.info(f"Saved → {RESULTS_PATH}")


if __name__ == "__main__":
    main()
