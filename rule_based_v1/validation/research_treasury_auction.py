"""Paper-specified Treasury auction concession and reversal research.

Implements the event windows from Fleming, Liu, and Nguyen (2026), FRBNY Staff
Report 1188. The primary specification is deliberately simple:

* short the matched Treasury future for the 180 minutes before auction close;
* remain flat across the result release;
* long from ``result_release + buffer`` through the next 180 minutes.

The module never infers auction times. Supply a TreasuryDirect-derived calendar
with explicit ``auction_close`` and ``result_release`` timestamps. This avoids
look-ahead and handles occasional non-13:00 closes correctly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_BAR_COLUMNS = {"open", "high", "low", "close"}
REQUIRED_AUCTION_COLUMNS = {"auction_close", "result_release", "security_term"}


def _as_utc_index(frame: pd.DataFrame, *, timestamp_col: str | None, input_timezone: str | None) -> pd.DataFrame:
    out = frame.copy()
    if timestamp_col:
        if timestamp_col not in out.columns:
            raise ValueError(f"missing timestamp column {timestamp_col!r}")
        idx = pd.to_datetime(out.pop(timestamp_col), errors="raise")
    else:
        idx = pd.to_datetime(out.index, errors="raise")
    if idx.tz is None:
        if not input_timezone:
            raise ValueError("naive bar timestamps require input_timezone")
        idx = idx.tz_localize(input_timezone, ambiguous="NaT", nonexistent="NaT")
    out.index = idx.tz_convert("UTC")
    out = out[~out.index.isna()].sort_index()
    missing = REQUIRED_BAR_COLUMNS.difference(out.columns)
    if missing:
        raise ValueError(f"bars missing columns {sorted(missing)}")
    if out.index.has_duplicates:
        out = out[~out.index.duplicated(keep="last")]
    return out


def normalise_auction_calendar(auctions: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_AUCTION_COLUMNS.difference(auctions.columns)
    if missing:
        raise ValueError(f"auction calendar missing columns {sorted(missing)}")
    out = auctions.copy()
    for col in ("auction_close", "result_release"):
        ts = pd.to_datetime(out[col], errors="raise", utc=True)
        out[col] = ts
    if (out["result_release"] < out["auction_close"]).any():
        raise ValueError("result_release cannot precede auction_close")
    out["security_term"] = out["security_term"].astype(str).str.strip()
    return out.sort_values("auction_close").reset_index(drop=True)


def _last_close_at_or_before(bars: pd.DataFrame, ts: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    pos = bars.index.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    return bars.index[pos], float(bars.iloc[pos]["close"])


def _first_open_at_or_after(bars: pd.DataFrame, ts: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    pos = bars.index.searchsorted(ts, side="left")
    if pos >= len(bars):
        return None
    return bars.index[pos], float(bars.iloc[pos]["open"])


def _leg_cost(*, contracts: int, commission_per_side: float, slippage_ticks: float, tick_value: float) -> float:
    return 2.0 * contracts * (commission_per_side + slippage_ticks * tick_value)


def build_auction_events(
    bars: pd.DataFrame,
    auctions: pd.DataFrame,
    *,
    security_terms: Iterable[str] | None = None,
    pre_minutes: int = 180,
    post_minutes: int = 180,
    result_buffer_minutes: int = 1,
    contracts: int = 1,
    point_value: float = 1_000.0,
    tick_value: float = 7.8125,
    commission_per_side: float = 0.62,
    slippage_ticks: float = 1.0,
) -> pd.DataFrame:
    """Construct pre-auction short and post-result long PnL rows.

    Bars must have a timezone-aware UTC index. Prices are assumed to use standard
    Treasury-futures decimal points, so one full point is worth ``point_value``.
    """
    if bars.index.tz is None:
        raise ValueError("bars must have a timezone-aware index")
    if pre_minutes <= 0 or post_minutes <= 0 or result_buffer_minutes < 0:
        raise ValueError("invalid event-window length")
    if contracts <= 0 or point_value <= 0 or tick_value <= 0:
        raise ValueError("contracts and contract values must be positive")

    calendar = normalise_auction_calendar(auctions)
    if security_terms is not None:
        allowed = {str(x).strip().lower() for x in security_terms}
        calendar = calendar[calendar["security_term"].str.lower().isin(allowed)]

    rows: list[dict] = []
    cost = _leg_cost(
        contracts=contracts,
        commission_per_side=commission_per_side,
        slippage_ticks=slippage_ticks,
        tick_value=tick_value,
    )
    for auction in calendar.itertuples(index=False):
        close_ts = pd.Timestamp(auction.auction_close)
        release_ts = pd.Timestamp(auction.result_release)
        pre_start_target = close_ts - pd.Timedelta(minutes=pre_minutes)
        post_start_target = release_ts + pd.Timedelta(minutes=result_buffer_minutes)
        post_end_target = release_ts + pd.Timedelta(minutes=post_minutes)

        pre_start = _first_open_at_or_after(bars, pre_start_target)
        pre_end = _last_close_at_or_before(bars, close_ts)
        post_start = _first_open_at_or_after(bars, post_start_target)
        post_end = _last_close_at_or_before(bars, post_end_target)
        if None in (pre_start, pre_end, post_start, post_end):
            continue
        pre_start_ts, pre_entry = pre_start
        pre_end_ts, pre_exit = pre_end
        post_start_ts, post_entry = post_start
        post_end_ts, post_exit = post_end
        if not (pre_start_ts < pre_end_ts <= close_ts):
            continue
        if not (post_start_ts < post_end_ts and post_start_ts >= post_start_target):
            continue

        pre_points = pre_entry - pre_exit
        post_points = post_exit - post_entry
        pre_pnl = pre_points * point_value * contracts - cost
        post_pnl = post_points * point_value * contracts - cost
        rows.append(
            {
                "day": close_ts.tz_convert("America/New_York").date(),
                "security_term": auction.security_term,
                "auction_close": close_ts,
                "result_release": release_ts,
                "pre_entry_time": pre_start_ts,
                "pre_exit_time": pre_end_ts,
                "post_entry_time": post_start_ts,
                "post_exit_time": post_end_ts,
                "pre_points": pre_points,
                "post_points": post_points,
                "pre_pnl": pre_pnl,
                "post_pnl": post_pnl,
                "pnl": pre_pnl + post_pnl,
            }
        )
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
    return _as_utc_index(raw, timestamp_col=timestamp_col, input_timezone=input_timezone)


def main() -> None:
    ap = argparse.ArgumentParser(description="FRBNY Treasury-auction concession/reversal research")
    ap.add_argument("--bars", type=Path, required=True)
    ap.add_argument("--auctions", type=Path, required=True)
    ap.add_argument("--security-term", action="append", default=["5-Year"])
    ap.add_argument("--timestamp-col", default=None)
    ap.add_argument("--input-timezone", default=None)
    ap.add_argument("--hdf-key", default="bars_1min")
    ap.add_argument("--contracts", type=int, default=1)
    ap.add_argument("--point-value", type=float, default=1_000.0)
    ap.add_argument("--tick-value", type=float, default=7.8125)
    ap.add_argument("--commission-per-side", type=float, default=0.62)
    ap.add_argument("--slippage-ticks", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    bars = load_bars(
        args.bars,
        timestamp_col=args.timestamp_col,
        input_timezone=args.input_timezone,
        hdf_key=args.hdf_key,
    )
    auctions = pd.read_csv(args.auctions)
    events = build_auction_events(
        bars,
        auctions,
        security_terms=args.security_term,
        contracts=args.contracts,
        point_value=args.point_value,
        tick_value=args.tick_value,
        commission_per_side=args.commission_per_side,
        slippage_ticks=args.slippage_ticks,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out, index=False)
    if events.empty:
        print("no complete auction windows")
        return
    pnl = events["pnl"]
    t = pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl))) if len(pnl) > 1 and pnl.std(ddof=1) > 0 else np.nan
    print(
        f"events={len(events)} mean=${pnl.mean():+.2f} total=${pnl.sum():+.2f} "
        f"WR={(pnl > 0).mean():.1%} t={t:.2f}"
    )


if __name__ == "__main__":
    main()
