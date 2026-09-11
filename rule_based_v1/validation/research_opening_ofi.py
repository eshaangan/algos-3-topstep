"""Run the preregistered opening-OFI continuation test.

See ``PREREG_opening_ofi_v1.md``.  This module deliberately exposes no signal
threshold or alternate holding-period arguments: those are not part of v1.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd

from ml_intraday_v3.features.depth_scaled_flow import (
    build_bars,
    replay_bbo_from_mbp1,
)

NY = "America/New_York"
ROUND_TRIP_COST = 2.05
POINT_VALUE = 2.0


def normalise_databento_mbp1(raw: pd.DataFrame) -> pd.DataFrame:
    """Map Databento's native MBP-1 schema to the common BBO schema."""
    d = raw.copy()
    if "recv_ns" not in d.columns:
        if not isinstance(d.index, pd.DatetimeIndex):
            raise ValueError("MBP-1 data needs recv_ns or a DatetimeIndex")
        idx = pd.DatetimeIndex(d.index)
        if idx.tz is None:
            raise ValueError("MBP-1 timestamps must be timezone aware")
        d["recv_ns"] = idx.tz_convert("UTC").asi8
    aliases = {
        "bid_px_00": "bid_px",
        "ask_px_00": "ask_px",
        "bid_sz_00": "bid_sz",
        "ask_sz_00": "ask_sz",
    }
    d = d.rename(columns={k: v for k, v in aliases.items() if k in d.columns})
    required = {"recv_ns", "bid_px", "ask_px", "bid_sz", "ask_sz"}
    missing = required.difference(d.columns)
    if missing:
        raise ValueError(f"MBP-1 data missing columns {sorted(missing)}")
    return d[list(sorted(required))]


def opening_signal(features: pd.DataFrame) -> float:
    """Return depth-scaled OFI in the frozen 09:30:00--09:30:14 window."""
    ts = pd.DatetimeIndex(pd.to_datetime(features["ts"], utc=True)).tz_convert(NY)
    clock = np.asarray(ts.time)
    keep = (clock >= time(9, 30)) & (clock < time(9, 30, 15))
    window = features.loc[keep]
    if window.empty:
        return np.nan
    depth = float(window["avg_depth"].replace([np.inf, -np.inf], np.nan).mean())
    if not np.isfinite(depth) or depth <= 0:
        return np.nan
    return float(window["ofi"].sum() / depth)


def session_prices(minute_bars: pd.DataFrame, session_day: date) -> tuple[float, float] | None:
    """Get frozen 09:31 and 09:36 closes for one New York session."""
    if not isinstance(minute_bars.index, pd.DatetimeIndex) or minute_bars.index.tz is None:
        raise ValueError("minute bars need a timezone-aware DatetimeIndex")
    bars = minute_bars.tz_convert(NY)
    entry_ts = pd.Timestamp.combine(session_day, time(9, 31)).tz_localize(NY)
    exit_ts = pd.Timestamp.combine(session_day, time(9, 36)).tz_localize(NY)
    if entry_ts not in bars.index or exit_ts not in bars.index:
        return None
    return float(bars.at[entry_ts, "close"]), float(bars.at[exit_ts, "close"])


def event_from_signal(
    session_day: date,
    signal: float,
    entry: float,
    exit_: float,
    partition: str,
    source: str,
) -> dict | None:
    if not np.isfinite(signal) or signal == 0 or entry <= 0 or exit_ <= 0:
        return None
    direction = 1 if signal > 0 else -1
    points = direction * (exit_ - entry)
    gross = points * POINT_VALUE
    return {
        "day": session_day.isoformat(),
        "partition": partition,
        "source": source,
        "signal": signal,
        "direction": direction,
        "entry": entry,
        "exit": exit_,
        "points": points,
        "gross_pnl": gross,
        "net_pnl": gross - ROUND_TRIP_COST,
    }


def historical_events(mbp1_dir: Path, minute_bars: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(mbp1_dir.glob("*.parquet")):
        raw = pd.read_parquet(path)
        common = normalise_databento_mbp1(raw)
        features = build_bars(replay_bbo_from_mbp1(common), grid_ms=1000)
        signal = opening_signal(features)
        session_day = pd.Timestamp(raw.index[0]).tz_convert(NY).date()
        prices = session_prices(minute_bars, session_day)
        if prices is None:
            continue
        partition = "development" if session_day <= date(2026, 4, 30) else "validation"
        event = event_from_signal(session_day, signal, *prices, partition, "NQ_MBP1")
        if event is not None:
            rows.append(event)
    return rows


def recent_events(feature_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(feature_dir.glob("features_*.parquet")):
        features = pd.read_parquet(path)
        ts = pd.DatetimeIndex(pd.to_datetime(features["ts"], utc=True)).tz_convert(NY)
        session_day = ts[0].date()
        signal = opening_signal(features)
        indexed = features.assign(_ts=ts).set_index("_ts")
        entry_ts = pd.Timestamp.combine(session_day, time(9, 31)).tz_localize(NY)
        exit_ts = pd.Timestamp.combine(session_day, time(9, 36)).tz_localize(NY)
        if entry_ts not in indexed.index or exit_ts not in indexed.index:
            continue
        entry = float(indexed.at[entry_ts, "mid"])
        exit_ = float(indexed.at[exit_ts, "mid"])
        event = event_from_signal(session_day, signal, entry, exit_, "final_holdout", "MNQ_MBO")
        if event is not None:
            rows.append(event)
    return rows


def _mean_t(values: pd.Series) -> tuple[float, float]:
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    t_stat = mean / (sd / np.sqrt(len(values))) if len(values) > 1 and sd > 0 else np.nan
    return mean, float(t_stat)


def bootstrap_mean_ci(values: np.ndarray, *, seed: int = 20260910, draws: int = 20_000) -> tuple[float, float]:
    """Percentile CI over independent session outcomes (one observation/day)."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    means = np.empty(draws)
    # Chunk to avoid allocating draws*n when this is reused on longer histories.
    for start in range(0, draws, 2_000):
        stop = min(start + 2_000, draws)
        sample = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[sample].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def summarise(events: pd.DataFrame) -> dict:
    partitions: dict[str, dict] = {}
    for name, part in events.groupby("partition", sort=False):
        pnl = part["net_pnl"]
        mean, t_stat = _mean_t(pnl)
        partitions[name] = {
            "n": int(len(part)),
            "mean_net_pnl": mean,
            "total_net_pnl": float(pnl.sum()),
            "win_rate": float((pnl > 0).mean()),
            "t_stat": t_stat,
        }
    pooled = events["net_pnl"].to_numpy(dtype=float)
    mean, t_stat = _mean_t(events["net_pnl"])
    ci = bootstrap_mean_ci(pooled)
    partition_gate = all(
        partitions.get(name, {}).get("mean_net_pnl", -np.inf) > 0
        for name in ("development", "validation", "final_holdout")
    )
    deployable_gate = bool(partition_gate and t_stat >= 2.0 and ci[0] > 0)
    return {
        "partitions": partitions,
        "pooled": {
            "n": int(len(events)),
            "mean_net_pnl": mean,
            "total_net_pnl": float(events["net_pnl"].sum()),
            "win_rate": float((events["net_pnl"] > 0).mean()),
            "t_stat": t_stat,
            "bootstrap_95pct_mean_ci": list(ci),
        },
        "partition_gate": bool(partition_gate),
        "deployable_gate": deployable_gate,
        "verdict": "PAPER_TRIAL_ONLY" if deployable_gate else "REJECT",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mbp1-dir", type=Path, required=True)
    ap.add_argument("--minute-bars", type=Path, required=True)
    ap.add_argument("--recent-features", type=Path, required=True)
    ap.add_argument("--out-events", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    args = ap.parse_args()

    minute_bars = pd.read_parquet(args.minute_bars)
    events = pd.DataFrame(
        historical_events(args.mbp1_dir, minute_bars) + recent_events(args.recent_features)
    ).sort_values("day")
    summary = summarise(events)
    args.out_events.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out_events, index=False)
    args.out_summary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
