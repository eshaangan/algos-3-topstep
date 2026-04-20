"""IVB-style ORB approximation backtest.

This recreates the public description of Fabio Valentini's IVB/ORB concept
with data available in this repo: 5-minute OHLCV bars. It is not a clone of
the proprietary DeepCharts model because we do not have MBO/order-flow data.

Model:
  1. Build the opening range from the first RTH bars.
  2. Approximate a fixed-range volume profile across that range.
  3. Trade only after range expansion with volume/body "aggression".
  4. Support two entry modes:
       - breakout: immediate close beyond OR high/low.
       - reload: pullback into OR value-area edge after a breakout.

Run from repo root:
  python rule_based_v1/diagnostics/ivb_orb_backtest.py
  python rule_based_v1/diagnostics/ivb_orb_backtest.py --instrument MNQ --mode reload
  python rule_based_v1/diagnostics/ivb_orb_backtest.py --instrument MES --sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in (str(ROOT), str(RBV1)):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS_PATH = RBV1 / "diagnostics" / "ivb_orb_results.json"

# MES_FULL merges the long history (2019-2025) with 2026 YTD; load_bars handles this via a list.
DATASETS = {
    "MES": ROOT / "data" / "processed" / "mes_2026_ytd_5m.h5",
    "MES_FULL": [
        ROOT / "data" / "processed" / "mes_bars.h5",
        ROOT / "data" / "processed" / "mes_2026_ytd_5m.h5",
    ],
    "MNQ": ROOT / "data" / "processed" / "mnq_2026ytd_5min.h5",
    "MNQ_FULL": ROOT / "data" / "processed" / "mnq_5min_aug25_mar26.h5",
    "MNK": ROOT / "data" / "processed" / "mnk_2026ytd_5min.h5",
    "M2K": ROOT / "data" / "processed" / "m2k_bars_5min.h5",
    "MGC": ROOT / "data" / "processed" / "mgc_bars_5min.h5",
}

SPECS = {
    "MES":      {"point_value": 5.0,  "tick_size": 0.25, "commission": 0.62, "slippage_ticks": 1},
    "MES_FULL": {"point_value": 5.0,  "tick_size": 0.25, "commission": 0.62, "slippage_ticks": 1},
    "MNQ":      {"point_value": 2.0,  "tick_size": 0.25, "commission": 0.62, "slippage_ticks": 1},
    "MNQ_FULL": {"point_value": 2.0,  "tick_size": 0.25, "commission": 0.62, "slippage_ticks": 1},
    "M2K":      {"point_value": 5.0,  "tick_size": 0.10, "commission": 0.62, "slippage_ticks": 1},
    "MGC":      {"point_value": 10.0, "tick_size": 0.10, "commission": 0.62, "slippage_ticks": 1},
    "MNK": {"point_value": 0.5, "tick_size": 5.0, "commission": 0.62, "slippage_ticks": 1},
}


@dataclass
class Position:
    direction: int
    entry_price: float
    entry_idx: int
    stop: float
    target: float
    mode: str
    breakeven_triggered: bool = field(default=False)


def _load_single(path: Path) -> pd.DataFrame:
    """Load one HDF or parquet file into a standardized OHLCV DataFrame."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(str(path))
    else:
        with pd.HDFStore(str(path), "r") as store:
            keys = [k.strip("/") for k in store.keys()]
        preferred = next((k for k in ("bars_5min", "bars") if k in keys), keys[0])
        df = pd.read_hdf(str(path), key=preferred)

    df = df.rename(columns=str.lower)
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["open", "high", "low", "close", "volume"]]


def load_bars(path) -> pd.DataFrame:
    """Load one or more HDF/parquet files and return a deduplicated sorted OHLCV DataFrame."""
    if isinstance(path, list):
        parts = [_load_single(p) for p in path]
        df = pd.concat(parts)
        df = df[~df.index.duplicated(keep="last")]
    else:
        df = _load_single(path)
    return df.sort_index()


def compute_atr(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    h = bars["high"]
    l = bars["low"]
    prev_c = bars["close"].shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def volume_profile(
    bars: pd.DataFrame,
    *,
    tick_size: float,
    bins: int = 24,
    value_area_pct: float = 0.70,
) -> dict[str, float]:
    """Approximate FRVP by spreading each bar's volume over touched bins."""
    lo = float(bars["low"].min())
    hi = float(bars["high"].max())
    if hi <= lo:
        return {"poc": hi, "vah": hi, "val": lo}

    step = max(tick_size, (hi - lo) / bins)
    edges = np.arange(lo, hi + step, step)
    if len(edges) < 2:
        edges = np.array([lo, lo + tick_size])
    hist = np.zeros(len(edges) - 1)

    for _, bar in bars.iterrows():
        bar_lo = float(bar["low"])
        bar_hi = float(bar["high"])
        vol = max(float(bar["volume"]), 0.0)
        touched = np.where((edges[:-1] <= bar_hi) & (edges[1:] >= bar_lo))[0]
        if len(touched) == 0:
            continue
        hist[touched] += vol / len(touched)

    centers = (edges[:-1] + edges[1:]) / 2
    if hist.sum() <= 0:
        return {"poc": float(centers[len(centers) // 2]), "vah": hi, "val": lo}

    poc_idx = int(hist.argmax())
    selected = {poc_idx}
    total = hist[poc_idx]
    target = hist.sum() * value_area_pct
    left = poc_idx - 1
    right = poc_idx + 1
    while total < target and (left >= 0 or right < len(hist)):
        left_vol = hist[left] if left >= 0 else -1
        right_vol = hist[right] if right < len(hist) else -1
        if right_vol >= left_vol:
            selected.add(right)
            total += max(right_vol, 0)
            right += 1
        else:
            selected.add(left)
            total += max(left_vol, 0)
            left -= 1

    return {
        "poc": float(centers[poc_idx]),
        "vah": float(edges[max(selected) + 1]),
        "val": float(edges[min(selected)]),
    }


def slip(price: float, direction: int, is_entry: bool, tick_size: float, ticks: int) -> float:
    amount = tick_size * ticks
    return price + amount * direction if is_entry else price - amount * direction


def trade_pnl(
    entry: float,
    exit_: float,
    direction: int,
    contracts: int,
    point_value: float,
    commission: float,
) -> float:
    return (exit_ - entry) * direction * contracts * point_value - 2 * commission * contracts


def aggression(bar: pd.Series, avg_volume: float, direction: int, min_volume_ratio: float) -> bool:
    if avg_volume <= 0:
        return False
    body = float(bar["close"] - bar["open"]) * direction
    rng = max(float(bar["high"] - bar["low"]), 1e-9)
    body_ratio = body / rng
    return body_ratio >= 0.35 and float(bar["volume"]) >= avg_volume * min_volume_ratio


def summarize(trades: list[dict], daily_pnl: dict, starting_equity: float, max_drawdown: float) -> dict:
    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    daily = pd.Series(daily_pnl, dtype=float).sort_index()
    sharpe = 0.0
    if len(daily) > 1 and daily.std(ddof=1) > 0:
        sharpe = float((daily.mean() / daily.std(ddof=1)) * np.sqrt(252))

    exit_counts: dict[str, int] = {}
    for t in trades:
        exit_counts[t["reason"]] = exit_counts.get(t["reason"], 0) + 1

    return {
        "num_trades": int(len(trades)),
        "win_rate": float((pnls > 0).mean()) if len(pnls) else 0.0,
        "total_pnl": float(pnls.sum()) if len(pnls) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else 0.0,
        "avg_trade": float(pnls.mean()) if len(pnls) else 0.0,
        "avg_winner": float(wins.mean()) if len(wins) else 0.0,
        "avg_loser": float(losses.mean()) if len(losses) else 0.0,
        "sharpe_daily": sharpe,
        "max_drawdown": float(max_drawdown),
        "ending_equity": float(starting_equity + (pnls.sum() if len(pnls) else 0.0)),
        "active_days": int((daily != 0).sum()) if len(daily) else 0,
        "worst_day": float(daily.min()) if len(daily) else 0.0,
        "best_day": float(daily.max()) if len(daily) else 0.0,
        "exit_breakdown": exit_counts,
    }


def run_ivb(
    bars: pd.DataFrame,
    spec: dict,
    *,
    mode: str = "both",
    contracts: int = 1,
    session_start: str = "09:30",
    or_minutes: int = 35,
    entry_cutoff: str = "14:00",
    max_trades_per_day: int = 2,
    min_volume_ratio: float = 1.35,
    reload_tolerance_ticks: int = 4,
    target_range_mult: float = 1.5,
    time_stop_bars: int = 18,
    atr_stop_mult: float = 1.0,
    breakeven_after_mult: float = 0.5,
    starting_equity: float = 50_000.0,
) -> dict:
    tick_size = spec["tick_size"]
    point_value = spec["point_value"]
    commission = spec["commission"]
    slippage_ticks = spec["slippage_ticks"]

    start_t = pd.Timestamp(session_start).time()
    cutoff_t = pd.Timestamp(entry_cutoff).time()
    equity = starting_equity
    peak = starting_equity
    max_dd = 0.0
    trades: list[dict] = []
    daily_pnl = defaultdict(float)

    atr_series = compute_atr(bars)

    for date, day in bars.groupby(bars.index.date):
        day = day.copy()
        rth = day[(day.index.time >= start_t) & (day.index.time < pd.Timestamp("16:00").time())]
        if len(rth) < 20:
            continue

        start_ts = rth.index[0]
        or_end_ts = start_ts + pd.Timedelta(minutes=or_minutes)
        or_bars = rth[(rth.index >= start_ts) & (rth.index < or_end_ts)]
        post = rth[(rth.index >= or_end_ts) & (rth.index.time <= cutoff_t)]
        if len(or_bars) < 4 or post.empty:
            continue

        or_high = float(or_bars["high"].max())
        or_low = float(or_bars["low"].min())
        or_range = or_high - or_low
        if or_range <= tick_size * 4:
            continue

        vp = volume_profile(or_bars, tick_size=tick_size)
        avg_volume = float(or_bars["volume"].mean())
        reload_tol = reload_tolerance_ticks * tick_size
        position: Position | None = None
        trades_today = 0
        broke_up = False
        broke_down = False

        for idx, (ts, bar) in enumerate(rth.iterrows()):
            if position is not None:
                close = float(bar["close"])
                high = float(bar["high"])
                low = float(bar["low"])
                exit_price = None
                reason = ""

                # breakeven: slide stop to entry once profit >= threshold
                if not position.breakeven_triggered and breakeven_after_mult > 0:
                    unrealized = (close - position.entry_price) * position.direction
                    if unrealized >= breakeven_after_mult * or_range:
                        if position.direction == 1 and position.stop < position.entry_price:
                            position.stop = position.entry_price
                            position.breakeven_triggered = True
                        elif position.direction == -1 and position.stop > position.entry_price:
                            position.stop = position.entry_price
                            position.breakeven_triggered = True

                if position.direction == 1:
                    if low <= position.stop:
                        exit_price = slip(position.stop, 1, False, tick_size, slippage_ticks)
                        reason = "stop"
                    elif high >= position.target:
                        exit_price = slip(position.target, 1, False, tick_size, slippage_ticks)
                        reason = "target"
                else:
                    if high >= position.stop:
                        exit_price = slip(position.stop, -1, False, tick_size, slippage_ticks)
                        reason = "stop"
                    elif low <= position.target:
                        exit_price = slip(position.target, -1, False, tick_size, slippage_ticks)
                        reason = "target"

                if exit_price is None and idx - position.entry_idx >= time_stop_bars:
                    exit_price = slip(float(bar["close"]), position.direction, False, tick_size, slippage_ticks)
                    reason = "time_stop"
                if exit_price is None and ts == rth.index[-1]:
                    exit_price = slip(float(bar["close"]), position.direction, False, tick_size, slippage_ticks)
                    reason = "session_close"

                if exit_price is not None:
                    pnl = trade_pnl(
                        position.entry_price,
                        exit_price,
                        position.direction,
                        contracts,
                        point_value,
                        commission,
                    )
                    equity += pnl
                    peak = max(peak, equity)
                    max_dd = min(max_dd, equity - peak)
                    daily_pnl[str(date)] += pnl
                    trades.append(
                        {
                            "date": str(date),
                            "entry_time": str(position.entry_idx),
                            "direction": "long" if position.direction == 1 else "short",
                            "mode": position.mode,
                            "entry": round(position.entry_price, 4),
                            "exit": round(exit_price, 4),
                            "pnl": round(pnl, 2),
                            "reason": reason,
                        }
                    )
                    position = None

            if ts < or_end_ts or ts.time() > cutoff_t or trades_today >= max_trades_per_day:
                continue
            if position is not None:
                continue

            close = float(bar["close"])
            high = float(bar["high"])
            low = float(bar["low"])

            if close > or_high:
                broke_up = True
            if close < or_low:
                broke_down = True

            entry_direction = 0
            entry_mode = ""
            if mode in {"breakout", "both"}:
                if close > or_high and aggression(bar, avg_volume, 1, min_volume_ratio):
                    entry_direction = 1
                    entry_mode = "breakout"
                elif close < or_low and aggression(bar, avg_volume, -1, min_volume_ratio):
                    entry_direction = -1
                    entry_mode = "breakout"

            if entry_direction == 0 and mode in {"reload", "both"}:
                if broke_up and low <= vp["vah"] + reload_tol and close > vp["vah"]:
                    if aggression(bar, avg_volume, 1, min_volume_ratio * 0.9):
                        entry_direction = 1
                        entry_mode = "reload"
                elif broke_down and high >= vp["val"] - reload_tol and close < vp["val"]:
                    if aggression(bar, avg_volume, -1, min_volume_ratio * 0.9):
                        entry_direction = -1
                        entry_mode = "reload"

            if entry_direction == 0:
                continue

            entry = slip(close, entry_direction, True, tick_size, slippage_ticks)

            # ATR-based stop when atr_stop_mult > 0, else legacy OR-floor stop
            if atr_stop_mult > 0:
                atr_val = float(atr_series.loc[:ts].iloc[-1])
                if entry_direction == 1:
                    stop = entry - atr_stop_mult * atr_val
                    target = entry + target_range_mult * or_range
                    if entry - stop <= tick_size:
                        continue
                else:
                    stop = entry + atr_stop_mult * atr_val
                    target = entry - target_range_mult * or_range
                    if stop - entry <= tick_size:
                        continue
            else:
                if entry_direction == 1:
                    stop = min(vp["val"], or_low) - tick_size
                    target = entry + target_range_mult * or_range
                    if entry - stop <= tick_size:
                        continue
                else:
                    stop = max(vp["vah"], or_high) + tick_size
                    target = entry - target_range_mult * or_range
                    if stop - entry <= tick_size:
                        continue

            position = Position(entry_direction, entry, idx, stop, target, entry_mode)
            trades_today += 1

    summary = summarize(trades, daily_pnl, starting_equity, abs(max_dd))
    return {"summary": summary, "trades": trades}


def run_sweep(
    bars: pd.DataFrame,
    spec: dict,
    output_path: Path,
    instrument: str,
) -> list[dict]:
    grid = list(product(
        [25, 30, 35, 40],           # or_minutes
        [1.0, 1.5, 2.0],            # target_range_mult
        [0.0, 0.75, 1.0, 1.5],     # atr_stop_mult (0 = legacy OR-floor stop)
        [1.0, 1.1, 1.25],           # min_volume_ratio
    ))

    candidates = []
    for or_min, tgt, atr_stop, vol in grid:
        r = run_ivb(
            bars,
            spec,
            or_minutes=or_min,
            target_range_mult=tgt,
            atr_stop_mult=atr_stop,
            min_volume_ratio=vol,
            entry_cutoff="14:00",
            max_trades_per_day=3,
            breakeven_after_mult=0.0,
        )
        s = r["summary"]
        if s["num_trades"] >= 30 and s["max_drawdown"] <= 500 and s["profit_factor"] >= 1.5:
            candidates.append({
                "params": {
                    "or_minutes": or_min,
                    "target_range_mult": tgt,
                    "atr_stop_mult": atr_stop,
                    "min_volume_ratio": vol,
                },
                **s,
            })

    candidates.sort(key=lambda x: x["sharpe_daily"], reverse=True)

    print(f"\nSweep results for {instrument} — {len(candidates)} configs passed filters")
    print(f"{'or_min':>6} {'tgt':>5} {'atr':>5} {'vol':>5} | {'n':>4} {'WR':>6} {'PnL':>8} {'PF':>5} {'Sharpe':>7} {'MaxDD':>7}")
    print("-" * 75)
    for c in candidates[:15]:
        p = c["params"]
        print(
            f"{p['or_minutes']:>6} {p['target_range_mult']:>5.1f} {p['atr_stop_mult']:>5.2f} {p['min_volume_ratio']:>5.2f} | "
            f"{c['num_trades']:>4} {c['win_rate']:>6.1%} {c['total_pnl']:>8.2f} {c['profit_factor']:>5.2f} "
            f"{c['sharpe_daily']:>7.2f} {c['max_drawdown']:>7.2f}"
        )

    out = {
        "model": "ivb_orb_sweep",
        "instrument": instrument,
        "filters": {"min_trades": 30, "max_drawdown": 500, "min_profit_factor": 1.5},
        "total_configs_tested": len(grid),
        "configs_passed": len(candidates),
        "top_configs": candidates[:20],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {output_path}")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description="IVB-style ORB approximation backtest")
    parser.add_argument("--instrument", choices=sorted(DATASETS), default=None)
    parser.add_argument("--mode", choices=["breakout", "reload", "both"], default="both")
    parser.add_argument("--contracts", type=int, default=1,
                        help="Number of contracts per trade.")
    parser.add_argument("--start", default=None,
                        help="Start date (YYYY-MM-DD). Defaults to all available data.")
    parser.add_argument("--end", default=None)
    parser.add_argument("--or-minutes", type=int, default=35)
    parser.add_argument("--min-volume-ratio", type=float, default=1.35)
    parser.add_argument("--target-range-mult", type=float, default=1.5)
    parser.add_argument("--time-stop-bars", type=int, default=18)
    parser.add_argument("--atr-stop-mult", type=float, default=1.0,
                        help="ATR multiplier for stop distance. Set 0 to use legacy OR-floor stop.")
    parser.add_argument("--breakeven-after-mult", type=float, default=0.5,
                        help="Slide stop to entry after this multiple of OR range profit. Set 0 to disable.")
    parser.add_argument("--entry-cutoff", default="14:00",
                        help="Last bar timestamp allowed for new entries (HH:MM, ET).")
    parser.add_argument("--max-trades", type=int, default=2,
                        help="Maximum new entries per day.")
    parser.add_argument("--sweep", action="store_true",
                        help="Run parameter grid sweep instead of single backtest.")
    parser.add_argument("--output", default=str(RESULTS_PATH))
    args = parser.parse_args()

    instruments = [args.instrument] if args.instrument else ["MES", "MNQ", "MNK"]

    if args.sweep:
        instrument = args.instrument or "MES"
        bars = load_bars(DATASETS[instrument])
        if args.start:
            bars = bars[bars.index >= pd.Timestamp(args.start, tz="US/Eastern")]
        if args.end:
            bars = bars[bars.index < pd.Timestamp(args.end, tz="US/Eastern")]
        sweep_path = Path(args.output) if args.output != str(RESULTS_PATH) else \
            RBV1 / "diagnostics" / "ivb_orb_sweep.json"
        run_sweep(bars, SPECS[instrument], sweep_path, instrument)
        return

    results = {}
    for instrument in instruments:
        bars = load_bars(DATASETS[instrument])
        if args.start:
            bars = bars[bars.index >= pd.Timestamp(args.start, tz="US/Eastern")]
        if args.end:
            bars = bars[bars.index < pd.Timestamp(args.end, tz="US/Eastern")]
        result = run_ivb(
            bars,
            SPECS[instrument],
            mode=args.mode,
            contracts=args.contracts,
            or_minutes=args.or_minutes,
            min_volume_ratio=args.min_volume_ratio,
            target_range_mult=args.target_range_mult,
            time_stop_bars=args.time_stop_bars,
            atr_stop_mult=args.atr_stop_mult,
            breakeven_after_mult=args.breakeven_after_mult,
            entry_cutoff=args.entry_cutoff,
            max_trades_per_day=args.max_trades,
        )
        results[instrument] = {
            "path": str(DATASETS[instrument]),
            "mode": args.mode,
            "contracts": args.contracts,
            "params": {
                "or_minutes": args.or_minutes,
                "min_volume_ratio": args.min_volume_ratio,
                "target_range_mult": args.target_range_mult,
                "time_stop_bars": args.time_stop_bars,
                "atr_stop_mult": args.atr_stop_mult,
                "breakeven_after_mult": args.breakeven_after_mult,
                "entry_cutoff": args.entry_cutoff,
                "max_trades": args.max_trades,
            },
            **result,
        }

    out = {
        "model": "ivb_orb_ohlcv_approximation",
        "notes": [
            "Approximates public IVB descriptions with OHLCV only.",
            "No proprietary MBO, iceberg, absorption, or speed-of-tape data is available.",
        ],
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\nIVB ORB approximation")
    print("=" * 80)
    for instrument, result in results.items():
        s = result["summary"]
        exits = s.get("exit_breakdown", {})
        exit_str = " ".join(f"{k}={v}" for k, v in sorted(exits.items()))
        print(
            f"{instrument:>3} {args.mode:>8} {s['num_trades']:>3} trades  "
            f"PnL=${s['total_pnl']:>8.2f}  WR={s['win_rate']:>5.1%}  "
            f"PF={s['profit_factor']:>4.2f}  Sharpe={s['sharpe_daily']:>5.2f}  "
            f"MaxDD=${s['max_drawdown']:>7.2f}  WorstDay=${s['worst_day']:>7.2f}"
        )
        print(f"    avgW=${s['avg_winner']:>7.2f}  avgL=${s['avg_loser']:>7.2f}  exits: {exit_str}")
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
