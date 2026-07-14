"""Paper-specified market intraday momentum research.

Use the 09:30-10:00 ET index-futures return to choose the direction of a
15:30-15:55 ET trade. This is not an opening-range breakout: exposure is confined
to the final half-hour window.

Inspired by Gao, Han, Li, and Zhou (2018), and the hedging-demand mechanism in
Baltussen, Da, Lammers, and Martens (2021).
"""
from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"open", "high", "low", "close"}


def normalise_bars(bars: pd.DataFrame, *, timestamp_col: str | None = None, input_timezone: str | None = None) -> pd.DataFrame:
    out = bars.copy()
    if timestamp_col:
        if timestamp_col not in out.columns:
            raise ValueError(f"missing timestamp column {timestamp_col!r}")
        idx = pd.to_datetime(out.pop(timestamp_col), errors="raise")
    else:
        idx = pd.to_datetime(out.index, errors="raise")
    if idx.tz is None:
        if not input_timezone:
            raise ValueError("naive timestamps require input_timezone")
        idx = idx.tz_localize(input_timezone, ambiguous="NaT", nonexistent="NaT")
    out.index = idx.tz_convert("America/New_York")
    out = out[~out.index.isna()].sort_index()
    missing = REQUIRED_COLUMNS.difference(out.columns)
    if missing:
        raise ValueError(f"bars missing columns {sorted(missing)}")
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    return out


def _window_prices(day: pd.DataFrame, *, signal_start: time, signal_end: time, trade_start: time, trade_end: time):
    idx_time = pd.Series(day.index.time, index=day.index)
    signal = day[(idx_time >= signal_start) & (idx_time <= signal_end)]
    trade = day[(idx_time >= trade_start) & (idx_time <= trade_end)]
    if signal.empty or trade.empty:
        return None
    return (
        signal.index[0], float(signal.iloc[0]["open"]),
        signal.index[-1], float(signal.iloc[-1]["close"]),
        trade.index[0], float(trade.iloc[0]["open"]),
        trade.index[-1], float(trade.iloc[-1]["close"]),
    )


def build_close_momentum_events(
    bars: pd.DataFrame,
    *,
    signal_start: time = time(9, 30),
    signal_end: time = time(10, 0),
    trade_start: time = time(15, 30),
    trade_end: time = time(15, 55),
    min_abs_signal_bps: float = 0.0,
    contracts: int = 1,
    point_value: float = 2.0,
    tick_value: float = 0.50,
    commission_per_side: float = 0.62,
    slippage_ticks: float = 1.0,
) -> pd.DataFrame:
    """Generate one close-window trade per complete RTH day.

    ``min_abs_signal_bps=0`` is the primary paper specification. Non-zero values
    are sensitivity analysis and must be counted as additional trials.
    """
    if bars.index.tz is None:
        raise ValueError("bars must have a timezone-aware index")
    if not (signal_start < signal_end < trade_start < trade_end):
        raise ValueError("windows must be ordered and non-overlapping")
    if contracts <= 0 or point_value <= 0 or tick_value <= 0:
        raise ValueError("contract values must be positive")

    ny = bars.tz_convert("America/New_York") if str(bars.index.tz) != "America/New_York" else bars
    cost = 2.0 * contracts * (commission_per_side + slippage_ticks * tick_value)
    rows: list[dict] = []
    for session_date, day in ny.groupby(ny.index.date):
        prices = _window_prices(day, signal_start=signal_start, signal_end=signal_end, trade_start=trade_start, trade_end=trade_end)
        if prices is None:
            continue
        sig_start_ts, sig_open, sig_end_ts, sig_close, entry_ts, entry, exit_ts, exit_ = prices
        if sig_open <= 0:
            continue
        signal_bps = (sig_close / sig_open - 1.0) * 10_000.0
        if abs(signal_bps) < min_abs_signal_bps or signal_bps == 0:
            continue
        direction = 1 if signal_bps > 0 else -1
        points = direction * (exit_ - entry)
        pnl = points * point_value * contracts - cost
        rows.append({
            "day": session_date,
            "signal_start": sig_start_ts,
            "signal_end": sig_end_ts,
            "entry_time": entry_ts,
            "exit_time": exit_ts,
            "direction": direction,
            "signal_bps": signal_bps,
            "entry_price": entry,
            "exit_price": exit_,
            "points": points,
            "pnl": pnl,
        })
    return pd.DataFrame(rows)


def load_bars(path: Path, *, timestamp_col: str | None, input_timezone: str | None, hdf_key: str) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        raw = pd.read_parquet(path)
    elif suffix in {".h5", ".hdf", ".hdf5"}:
        raw = pd.read_hdf(path, key=hdf_key)
    else:
        raise ValueError(f"unsupported bars format: {path.suffix}")
    return normalise_bars(raw, timestamp_col=timestamp_col, input_timezone=input_timezone)


def main() -> None:
    ap = argparse.ArgumentParser(description="First-half-hour to last-half-hour momentum research")
    ap.add_argument("--bars", type=Path, required=True)
    ap.add_argument("--timestamp-col", default=None)
    ap.add_argument("--input-timezone", default=None)
    ap.add_argument("--hdf-key", default="bars_5min")
    ap.add_argument("--min-abs-signal-bps", type=float, default=0.0)
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--point-value", type=float, default=2.0)
    ap.add_argument("--tick-value", type=float, default=0.50)
    ap.add_argument("--commission-per-side", type=float, default=0.62)
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bars = load_bars(args.bars, timestamp_col=args.timestamp_col, input_timezone=args.input_timezone, hdf_key=args.hdf_key)
    events = build_close_momentum_events(
        bars,
        min_abs_signal_bps=args.min_abs_signal_bps,
        contracts=args.contracts,
        point_value=args.point_value,
        tick_value=args.tick_value,
        commission_per_side=args.commission_per_side,
        slippage_ticks=args.slippage_ticks,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out, index=False)
    if events.empty:
        print("no complete sessions")
        return
    pnl = events["pnl"]
    t = pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))) if len(pnl) > 1 and pnl.std(ddof=1) > 0 else np.nan
    print(f"events={len(events)} mean=${pnl.mean():+.2f} total=${pnl.sum():+.2f} WR={(pnl > 0).mean():.1%} t={t:.2f}")


if __name__ == "__main__":
    main()
