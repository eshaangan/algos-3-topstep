#!/usr/bin/env python3
"""
ICT-style (rule-based) backtest: US RTH **opening range liquidity sweep + reclaim** on 5m MES/MNQ.

Rules (objective):
  - OR window 09:30–10:04 ET; post window (10:05–12:00 ET) take **first** signal only, **1 trade/day**.
  - LONG:  low < OR_low  and close > OR_low
  - SHORT: high > OR_high and close < OR_high
  - Stops/targets: SL = 1.5×ATR(14), TP = 3×ATR(14) at signal; time stop 24 bars; flat ≥15:55 ET.

See module docstring in git for usage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rule_based_v1"))
from utils.indicators import atr  # noqa: E402

sys.path.insert(0, str(ROOT))
from rule_based_v1.diagnostics.ifr_backtest import (  # noqa: E402
    _filter_ytd,
    _load_bars,
    fetch_and_save,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "ml_intraday_v3" / "experiments" / "results"
RESULTS_JSON = RESULTS_DIR / "ict_sweep_or_backtest.json"

SPECS = {
    "MES": {
        "symbol": "MES.c.0",
        "path": ROOT / "data" / "processed" / "mes_ict_sweep_5m.h5",
        "point_value": 5.0,
        "tick_size": 0.25,
        "commission": 0.62,
        "slippage_ticks": 1,
    },
    "MNQ": {
        "symbol": "MNQ.c.0",
        "path": ROOT / "data" / "processed" / "mnq_ict_sweep_5m.h5",
        "point_value": 2.0,
        "tick_size": 0.25,
        "commission": 0.62,
        "slippage_ticks": 1,
    },
}

OR_START = dt_time(9, 30)
OR_END = dt_time(10, 4)
ENTRY_END = dt_time(12, 0)
SESSION_FLAT = dt_time(15, 55)
ATR_PERIOD = 14
SL_ATR = 1.5
TP_ATR = 3.0
TIME_STOP_BARS = 24


def _t_et(ts: pd.Timestamp) -> dt_time:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("US/Eastern").time()


def _slip(price: float, direction: int, is_entry: bool, tick: float, slip_ticks: int) -> float:
    s = slip_ticks * tick
    return price + s * direction if is_entry else price - s * direction


@dataclass
class PlannedTrade:
    entry_i: int
    direction: int
    entry_price: float
    stop: float
    target: float


def _pnl(ep: float, x: float, direction: int, pv: float, n: int, comm: float) -> float:
    pts = (x - ep) if direction == 1 else (ep - x)
    return pts * pv * n - 2 * comm * n


def _build_planned(
    df: pd.DataFrame,
    atr_s: pd.Series,
    *,
    long_only: bool = False,
    min_sweep_atr: float = 0.0,
    sl_atr: float = SL_ATR,
    tp_atr: float = TP_ATR,
) -> list[PlannedTrade]:
    """One trade per calendar day (ET), first valid sweep in post window."""
    planned: list[PlannedTrade] = []
    ts_to_i = {pd.Timestamp(t): i for i, t in enumerate(df.index)}
    for _day, day_df in df.groupby(df.index.date):
        if len(day_df) < 8:
            continue
        tlist = day_df.index.map(lambda x: _t_et(pd.Timestamp(x)))
        or_mask = (tlist >= OR_START) & (tlist <= OR_END)
        or_bars = day_df.loc[or_mask]
        if len(or_bars) < 4:
            continue
        or_high = float(or_bars["high"].max())
        or_low = float(or_bars["low"].min())

        post_mask = (tlist > OR_END) & (tlist <= ENTRY_END)
        post = day_df.loc[post_mask]
        if post.empty:
            continue

        for ts, row in post.iterrows():
            ts = pd.Timestamp(ts)
            j = ts_to_i.get(ts)
            if j is None:
                continue
            a = float(atr_s.loc[ts])
            if np.isnan(a) or a <= 0:
                continue
            hi, lo, cl = float(row["high"]), float(row["low"]), float(row["close"])

            if lo < or_low and cl > or_low:
                sweep_depth = or_low - lo
                if min_sweep_atr > 0 and sweep_depth < min_sweep_atr * a:
                    continue
                ep = cl
                planned.append(
                    PlannedTrade(
                        entry_i=j,
                        direction=1,
                        entry_price=ep,
                        stop=ep - sl_atr * a,
                        target=ep + tp_atr * a,
                    )
                )
                break
            if not long_only and hi > or_high and cl < or_high:
                sweep_depth = hi - or_high
                if min_sweep_atr > 0 and sweep_depth < min_sweep_atr * a:
                    continue
                ep = cl
                planned.append(
                    PlannedTrade(
                        entry_i=j,
                        direction=-1,
                        entry_price=ep,
                        stop=ep + sl_atr * a,
                        target=ep - tp_atr * a,
                    )
                )
                break
    return planned


def _simulate_trade(
    df: pd.DataFrame,
    pt: PlannedTrade,
    spec: dict,
    n_contracts: int,
) -> tuple[float, str]:
    """Enter at close of entry_i; evaluate from next bar. Returns (pnl, reason)."""
    if pt.entry_i + 1 >= len(df):
        return 0.0, "no_followup_bars"
    tick = spec["tick_size"]
    pv = spec["point_value"]
    comm = spec["commission"]
    slip = spec["slippage_ticks"]
    idx = list(df.index)
    d = pt.direction
    ep = _slip(pt.entry_price, d, True, tick, slip)
    sl, tp = pt.stop, pt.target
    start = pt.entry_i + 1
    end_limit = min(len(df), pt.entry_i + 1 + TIME_STOP_BARS)
    day0 = idx[pt.entry_i].date()

    for j in range(start, end_limit):
        bar = df.iloc[j]
        ts = idx[j]
        h, l, c = float(bar["high"]), float(bar["low"]), float(bar["close"])
        t = _t_et(pd.Timestamp(ts))

        if ts.date() != day0 or t >= SESSION_FLAT:
            x = _slip(c, d, False, tick, slip)
            return _pnl(ep, x, d, pv, n_contracts, comm), "session_close"

        if d == 1:
            if l <= sl:
                x = _slip(sl, 1, False, tick, slip)
                return _pnl(ep, x, d, pv, n_contracts, comm), "stop_loss"
            if h >= tp:
                x = _slip(tp, 1, False, tick, slip)
                return _pnl(ep, x, d, pv, n_contracts, comm), "profit_target"
        else:
            if h >= sl:
                x = _slip(sl, -1, False, tick, slip)
                return _pnl(ep, x, d, pv, n_contracts, comm), "stop_loss"
            if l <= tp:
                x = _slip(tp, -1, False, tick, slip)
                return _pnl(ep, x, d, pv, n_contracts, comm), "profit_target"

    j = max(start, end_limit - 1)
    c = float(df.iloc[j]["close"])
    x = _slip(c, d, False, tick, slip)
    return _pnl(ep, x, d, pv, n_contracts, comm), "time_stop"


def run_backtest(
    df: pd.DataFrame,
    instrument: str,
    n_contracts: int = 1,
    *,
    long_only: bool = False,
    min_sweep_atr: float = 0.0,
    sl_atr: float | None = None,
    tp_atr: float | None = None,
) -> dict:
    df = df.sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.tz_convert("US/Eastern")

    spec = SPECS[instrument]
    atr_s = atr(df["high"], df["low"], df["close"], ATR_PERIOD)
    sla = sl_atr if sl_atr is not None else SL_ATR
    tpa = tp_atr if tp_atr is not None else TP_ATR
    planned = _build_planned(
        df, atr_s, long_only=long_only, min_sweep_atr=min_sweep_atr, sl_atr=sla, tp_atr=tpa
    )
    logger.info("Planned trades: %s", len(planned))

    trades = []
    for pt in planned:
        pnl, reason = _simulate_trade(df, pt, spec, n_contracts)
        trades.append({"pnl": round(pnl, 2), "reason": reason, "direction": pt.direction})

    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    total = sum(t["pnl"] for t in trades)
    gp = sum(wins) if wins else 0.0
    gl = abs(sum(losses)) if losses else 0.0

    eq = 50_000.0
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t["pnl"]
        peak = max(peak, eq)
        max_dd = min(max_dd, eq - peak)

    return {
        "instrument": instrument,
        "n_contracts": n_contracts,
        "num_trades": len(trades),
        "total_pnl": round(total, 2),
        "win_rate": round(len(wins) / len(trades), 3) if trades else None,
        "profit_factor": round(gp / gl, 3) if gl > 0 else None,
        "max_drawdown": round(max_dd, 2),
        "rules": {
            "or_et": "09:30-10:04",
            "signal_window_et": "10:05-12:00",
            "long_only": long_only,
            "min_sweep_atr": min_sweep_atr,
            "sl_atr": sla,
            "tp_atr": tpa,
            "time_stop_bars": TIME_STOP_BARS,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", choices=["MES", "MNQ"], default="MES")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--start", default="2024-06-01")
    ap.add_argument("--end", default="2026-04-15T23:59:59+00:00")
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--long-only", action="store_true", help="Only long setups (sweep OR low)")
    ap.add_argument("--min-sweep-atr", type=float, default=0.0, help="Min sweep depth in ATR units (0=off)")
    ap.add_argument("--sl-atr", type=float, default=None, help="Override stop distance (ATR mult)")
    ap.add_argument("--tp-atr", type=float, default=None, help="Override target distance (ATR mult)")
    ap.add_argument(
        "--grid",
        action="store_true",
        help="Run a small parameter grid; writes ict_sweep_or_grid_{mes|mnq}.csv",
    )
    args = ap.parse_args()

    spec = SPECS[args.instrument]
    path = spec["path"]

    df = None
    if args.fetch:
        try:
            df = fetch_and_save(spec["symbol"], path, args.start, args.end)
        except Exception as e:
            logger.error("Fetch failed: %s", e)
    if df is None and path.exists():
        df = _load_bars(path, "bars_5min")
    # Fallback: reuse IFR-style cached filenames if present
    if df is None and args.instrument == "MES":
        alt = ROOT / "data" / "processed" / "mes_2026ytd_5min.h5"
        if alt.exists():
            logger.info("Using cached %s", alt.name)
            df = _load_bars(alt, "bars_5min")
    if df is None and args.instrument == "MNQ":
        alt = ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5"
        if alt.exists():
            logger.info("Using cached %s", alt.name)
            df = _load_bars(alt, "bars_5min")

    if df is None or len(df) == 0:
        logger.error("No data. Run with --fetch or place HDF at %s", path)
        return 1

    df = _filter_ytd(df, args.start)
    logger.info("Bars: %s  %s → %s", len(df), df.index[0], df.index[-1])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.grid:
        rows = []
        for lo in (False, True):
            for ms in (0.0, 0.1, 0.2):
                st = run_backtest(
                    df,
                    args.instrument,
                    n_contracts=args.contracts,
                    long_only=lo,
                    min_sweep_atr=ms,
                    sl_atr=args.sl_atr,
                    tp_atr=args.tp_atr,
                )
                st["period_start"] = args.start
                st["period_end"] = args.end
                rows.append(
                    {
                        "long_only": lo,
                        "min_sweep_atr": ms,
                        "num_trades": st["num_trades"],
                        "total_pnl": st["total_pnl"],
                        "win_rate": st["win_rate"],
                        "profit_factor": st["profit_factor"],
                        "max_drawdown": st["max_drawdown"],
                    }
                )
        gpath = RESULTS_DIR / f"ict_sweep_or_grid_{args.instrument.lower()}.csv"
        pd.DataFrame(rows).to_csv(gpath, index=False)
        logger.info("Grid saved %s", gpath)
        print("\n" + "=" * 70)
        print(f"  GRID  {args.instrument}  {args.contracts}c")
        print("=" * 70)
        print(pd.DataFrame(rows).to_string(index=False))
        print("=" * 70 + "\n")
        grid_json = RESULTS_DIR / f"ict_sweep_or_grid_{args.instrument.lower()}.json"
        with open(grid_json, "w") as f:
            json.dump({"grid": rows, "instrument": args.instrument}, f, indent=2)
        logger.info("Grid JSON %s", grid_json)
        return 0

    stats = run_backtest(
        df,
        args.instrument,
        n_contracts=args.contracts,
        long_only=args.long_only,
        min_sweep_atr=args.min_sweep_atr,
        sl_atr=args.sl_atr,
        tp_atr=args.tp_atr,
    )
    stats["period_start"] = args.start
    stats["period_end"] = args.end

    with open(RESULTS_JSON, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved %s", RESULTS_JSON)

    print("\n" + "=" * 70)
    tag = "long-only " if args.long_only else ""
    print(f"  Sweep+reclaim OR (ICT-style proxy) {tag}|  {args.instrument}  {args.contracts}c  |  5m RTH")
    print("=" * 70)
    print(f"  Trades: {stats['num_trades']}  Total PnL: ${stats['total_pnl']:,.2f}")
    print(f"  Win rate: {stats['win_rate']}  PF: {stats['profit_factor']}  MaxDD: ${stats['max_drawdown']:,.2f}")
    if args.min_sweep_atr > 0:
        print(f"  min_sweep_atr={args.min_sweep_atr}")
    print("=" * 70 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
