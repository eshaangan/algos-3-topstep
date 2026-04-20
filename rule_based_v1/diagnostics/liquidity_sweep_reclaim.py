"""MNQ liquidity sweep + reclaim backtest.

Chart-only model:
  - Build daily obvious levels: prior RTH high/low and overnight high/low.
  - During liquid RTH, wait for a wick through one of those levels.
  - Enter when price closes back inside the swept level within a few bars.
  - Stop just beyond the sweep extreme.
  - Target the opposite side of the same range.

This intentionally uses OHLCV only. No DOM, footprint, MBO, or order-flow data.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import time as dt_time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT / "data" / "processed" / "mnq_2026ytd_databento_1min_eth.h5"
RESULTS_PATH = ROOT / "rule_based_v1" / "diagnostics" / "liquidity_sweep_reclaim_results.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

POINT_VALUE = 2.0
TICK_SIZE = 0.25
COMMISSION = 0.62
SLIPPAGE_TICKS = 1
STARTING_EQUITY = 50_000.0

SESSION_START = dt_time(9, 30)
SESSION_END = dt_time(16, 0)
ENTRY_START = dt_time(9, 35)
ENTRY_END = dt_time(15, 0)
FLAT_TIME = dt_time(15, 55)

MAX_DAILY_LOSS = -950.0
PER_TRADE_MAX_LOSS = 1_000.0
_CONTEXT_CACHE: dict[int, tuple[list, dict, dict, dict]] = {}


@dataclass(frozen=True)
class Level:
    name: str
    price: float
    side: str
    opposite: float


@dataclass
class PendingSweep:
    level: Level
    direction: int
    start_i: int
    expire_i: int
    extreme: float


@dataclass
class Position:
    date: object
    direction: int
    entry_i: int
    entry_time: str
    entry: float
    stop: float
    target: float
    level_name: str
    swept_level: float
    sweep_extreme: float
    time_stop_i: int


def _load_bars(path: Path = DATA_PATH, key: str = "bars_1min_eth") -> pd.DataFrame:
    df = pd.read_hdf(str(path), key=key)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC").tz_convert("US/Eastern")
    else:
        df.index = df.index.tz_convert("US/Eastern")
    return df.sort_index()


def _parse_time(value: str) -> dt_time:
    hour, minute = value.split(":", 1)
    return dt_time(int(hour), int(minute))


def _is_rth_index(index: pd.DatetimeIndex) -> np.ndarray:
    return (
        ((index.hour > 9) | ((index.hour == 9) & (index.minute >= 30)))
        & (index.hour < 16)
    )


def _daily_rth_stats(bars: pd.DataFrame) -> dict:
    rth = bars.loc[_is_rth_index(bars.index)]
    stats: dict = {}
    for d, day in rth.groupby(rth.index.date):
        if day.empty:
            continue
        stats[d] = {
            "high": float(day["high"].max()),
            "low": float(day["low"].min()),
            "open": float(day["open"].iloc[0]),
            "close": float(day["close"].iloc[-1]),
        }
    return stats


def _overnight_stats(bars: pd.DataFrame, dates: list) -> dict:
    stats: dict = {}
    for d in dates:
        open_ts = pd.Timestamp.combine(d, SESSION_START).tz_localize("US/Eastern")
        start_ts = open_ts - pd.Timedelta(hours=15, minutes=30)
        end_ts = open_ts - pd.Timedelta(minutes=1)
        on = bars.loc[(bars.index >= start_ts) & (bars.index <= end_ts)]
        if on.empty:
            continue
        stats[d] = {
            "high": float(on["high"].max()),
            "low": float(on["low"].min()),
            "open": float(on["open"].iloc[0]),
            "close": float(on["close"].iloc[-1]),
        }
    return stats


def _make_levels(d, prev_d, rth_stats: dict, on_stats: dict) -> list[Level]:
    levels: list[Level] = []
    prev = rth_stats.get(prev_d)
    if prev:
        levels.append(Level("prior_rth_low", prev["low"], "low", prev["high"]))
        levels.append(Level("prior_rth_high", prev["high"], "high", prev["low"]))
    on = on_stats.get(d)
    if on and on["high"] > on["low"]:
        levels.append(Level("overnight_low", on["low"], "low", on["high"]))
        levels.append(Level("overnight_high", on["high"], "high", on["low"]))
    return levels


def _slip(price: float, direction: int, is_entry: bool) -> float:
    slip = SLIPPAGE_TICKS * TICK_SIZE
    return price + direction * slip if is_entry else price - direction * slip


def _pnl(entry: float, exit_price: float, direction: int, contracts: int) -> float:
    gross = (exit_price - entry) * direction * POINT_VALUE * contracts
    return gross - 2 * COMMISSION * contracts


def _result_summary(trades: list[dict], daily_pnl: dict, config: dict, period: str) -> dict:
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] < 0]
    total = float(sum(t["pnl"] for t in trades))
    gross_profit = float(sum(wins))
    gross_loss = abs(float(sum(losses)))

    eq = STARTING_EQUITY
    peak = eq
    max_dd = 0.0
    for t in trades:
        eq += t["pnl"]
        peak = max(peak, eq)
        max_dd = min(max_dd, eq - peak)

    daily = pd.Series(daily_pnl, dtype=float)
    nonzero_daily = daily[daily != 0]
    sharpe = (
        float(nonzero_daily.mean() / nonzero_daily.std() * np.sqrt(252))
        if len(nonzero_daily) > 1 and nonzero_daily.std() > 0
        else 0.0
    )
    reason_counts = Counter(t["exit_reason"] for t in trades)
    level_counts = Counter(t["level"] for t in trades)
    side_counts = Counter("long" if t["direction"] == 1 else "short" for t in trades)

    return {
        "strategy": "MNQ liquidity sweep reclaim",
        "period": period,
        "config": config,
        "num_trades": len(trades),
        "win_rate": round(len(wins) / len(trades), 3) if trades else 0.0,
        "total_pnl": round(total, 2),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else None,
        "sharpe": round(sharpe, 3),
        "max_drawdown": round(max_dd, 2),
        "avg_trade": round(total / len(trades), 2) if trades else 0.0,
        "best_day": round(float(daily.max()), 2) if len(daily) else 0.0,
        "worst_day": round(float(daily.min()), 2) if len(daily) else 0.0,
        "exit_reasons": dict(reason_counts),
        "level_counts": dict(level_counts),
        "side_counts": dict(side_counts),
        "daily_pnl": {str(k): round(float(v), 2) for k, v in sorted(daily_pnl.items())},
        "trades": trades,
    }


def run_backtest(
    bars: pd.DataFrame,
    *,
    contracts: int = 1,
    confirm_bars: int = 2,
    min_sweep_ticks: int = 2,
    stop_buffer_ticks: int = 2,
    min_rr: float = 0.8,
    max_trades_per_day: int = 1,
    time_stop_bars: int = 60,
    use_prior_rth: bool = True,
    use_overnight: bool = True,
    side: str = "both",
) -> dict:
    bars = bars.sort_index()
    cache_key = id(bars)
    if cache_key in _CONTEXT_CACHE:
        dates, rth_stats, on_stats, prev_by_date = _CONTEXT_CACHE[cache_key]
    else:
        dates = sorted(set(bars.loc[_is_rth_index(bars.index)].index.date))
        rth_stats = _daily_rth_stats(bars)
        on_stats = _overnight_stats(bars, dates)
        prev_by_date = {dates[i]: dates[i - 1] if i > 0 else None for i in range(len(dates))}
        _CONTEXT_CACHE[cache_key] = (dates, rth_stats, on_stats, prev_by_date)

    tick = TICK_SIZE
    trades: list[dict] = []
    daily_pnl: dict = {}

    for d in dates:
        day = bars.loc[(bars.index.date == d) & _is_rth_index(bars.index)].copy()
        if day.empty:
            continue
        levels = _make_levels(d, prev_by_date[d], rth_stats, on_stats)
        if not use_prior_rth:
            levels = [x for x in levels if not x.name.startswith("prior_rth")]
        if not use_overnight:
            levels = [x for x in levels if not x.name.startswith("overnight")]
        if side == "long":
            levels = [x for x in levels if x.side == "low"]
        elif side == "short":
            levels = [x for x in levels if x.side == "high"]
        if not levels:
            daily_pnl[d] = 0.0
            continue

        pending: dict[str, PendingSweep] = {}
        pos: Position | None = None
        trades_today = 0
        pnl_today = 0.0
        day_index = list(day.index)

        for i, (ts, bar) in enumerate(day.iterrows()):
            t = ts.time()
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])

            if pos is not None:
                exit_price = None
                exit_reason = ""
                if t >= FLAT_TIME:
                    exit_price = _slip(close, pos.direction, False)
                    exit_reason = "session_close"
                elif i >= pos.time_stop_i:
                    exit_price = _slip(close, pos.direction, False)
                    exit_reason = "time_stop"
                elif pos.direction == 1:
                    if low <= pos.stop:
                        exit_price = _slip(pos.stop, pos.direction, False)
                        exit_reason = "stop_loss"
                    elif high >= pos.target:
                        exit_price = _slip(pos.target, pos.direction, False)
                        exit_reason = "profit_target"
                else:
                    if high >= pos.stop:
                        exit_price = _slip(pos.stop, pos.direction, False)
                        exit_reason = "stop_loss"
                    elif low <= pos.target:
                        exit_price = _slip(pos.target, pos.direction, False)
                        exit_reason = "profit_target"

                if exit_price is not None:
                    pnl = _pnl(pos.entry, exit_price, pos.direction, contracts)
                    pnl_today += pnl
                    trades.append(
                        {
                            "date": str(d),
                            "entry_time": pos.entry_time,
                            "exit_time": ts.strftime("%H:%M"),
                            "direction": pos.direction,
                            "level": pos.level_name,
                            "swept_level": round(pos.swept_level, 2),
                            "sweep_extreme": round(pos.sweep_extreme, 2),
                            "entry": round(pos.entry, 2),
                            "stop": round(pos.stop, 2),
                            "target": round(pos.target, 2),
                            "exit": round(exit_price, 2),
                            "pnl": round(pnl, 2),
                            "exit_reason": exit_reason,
                        }
                    )
                    pos = None

            if (
                pos is not None
                or trades_today >= max_trades_per_day
                or pnl_today <= MAX_DAILY_LOSS
                or t < ENTRY_START
                or t > ENTRY_END
            ):
                continue

            expired = [key for key, p in pending.items() if i > p.expire_i]
            for key in expired:
                del pending[key]

            for key, p in list(pending.items()):
                level = p.level.price
                reclaimed = close > level if p.direction == 1 else close < level
                if not reclaimed:
                    if p.direction == 1:
                        p.extreme = min(p.extreme, low)
                    else:
                        p.extreme = max(p.extreme, high)
                    continue

                entry = _slip(close, p.direction, True)
                if p.direction == 1:
                    stop = p.extreme - stop_buffer_ticks * tick
                    target = p.level.opposite
                    risk_pts = entry - stop
                    reward_pts = target - entry
                else:
                    stop = p.extreme + stop_buffer_ticks * tick
                    target = p.level.opposite
                    risk_pts = stop - entry
                    reward_pts = entry - target

                risk_usd = risk_pts * POINT_VALUE * contracts
                if risk_pts <= 0 or reward_pts <= 0 or risk_usd > PER_TRADE_MAX_LOSS:
                    del pending[key]
                    continue
                if reward_pts / risk_pts < min_rr:
                    del pending[key]
                    continue

                pos = Position(
                    date=d,
                    direction=p.direction,
                    entry_i=i,
                    entry_time=ts.strftime("%H:%M"),
                    entry=entry,
                    stop=stop,
                    target=target,
                    level_name=p.level.name,
                    swept_level=p.level.price,
                    sweep_extreme=p.extreme,
                    time_stop_i=min(i + time_stop_bars, len(day_index) - 1),
                )
                trades_today += 1
                pending.clear()
                break

            if pos is not None:
                continue

            for level in levels:
                key = f"{level.name}:{level.price}"
                if key in pending:
                    continue
                if level.side == "low" and low <= level.price - min_sweep_ticks * tick:
                    pending[key] = PendingSweep(
                        level=level,
                        direction=1,
                        start_i=i,
                        expire_i=i + confirm_bars,
                        extreme=low,
                    )
                elif level.side == "high" and high >= level.price + min_sweep_ticks * tick:
                    pending[key] = PendingSweep(
                        level=level,
                        direction=-1,
                        start_i=i,
                        expire_i=i + confirm_bars,
                        extreme=high,
                    )

        if pos is not None:
            last_ts = day.index[-1]
            close = float(day["close"].iloc[-1])
            exit_price = _slip(close, pos.direction, False)
            pnl = _pnl(pos.entry, exit_price, pos.direction, contracts)
            pnl_today += pnl
            trades.append(
                {
                    "date": str(d),
                    "entry_time": pos.entry_time,
                    "exit_time": last_ts.strftime("%H:%M"),
                    "direction": pos.direction,
                    "level": pos.level_name,
                    "swept_level": round(pos.swept_level, 2),
                    "sweep_extreme": round(pos.sweep_extreme, 2),
                    "entry": round(pos.entry, 2),
                    "stop": round(pos.stop, 2),
                    "target": round(pos.target, 2),
                    "exit": round(exit_price, 2),
                    "pnl": round(pnl, 2),
                    "exit_reason": "session_close",
                }
            )

        daily_pnl[d] = pnl_today

    config = {
        "contracts": contracts,
        "confirm_bars": confirm_bars,
        "min_sweep_ticks": min_sweep_ticks,
        "stop_buffer_ticks": stop_buffer_ticks,
        "min_rr": min_rr,
        "max_trades_per_day": max_trades_per_day,
        "time_stop_bars": time_stop_bars,
        "use_prior_rth": use_prior_rth,
        "use_overnight": use_overnight,
        "side": side,
        "entry_window_et": f"{ENTRY_START:%H:%M}-{ENTRY_END:%H:%M}",
    }
    period = f"{dates[0]} to {dates[-1]}" if dates else ""
    return _result_summary(trades, daily_pnl, config, period)


def _grid(bars: pd.DataFrame, contracts: int) -> list[dict]:
    rows: list[dict] = []
    for confirm_bars in (0, 1, 2):
        for min_sweep_ticks in (1, 2):
            for stop_buffer_ticks in (1, 2):
                for min_rr in (0.5, 0.8):
                    result = run_backtest(
                        bars,
                        contracts=contracts,
                        confirm_bars=confirm_bars,
                        min_sweep_ticks=min_sweep_ticks,
                        stop_buffer_ticks=stop_buffer_ticks,
                        min_rr=min_rr,
                    )
                    rows.append(
                        {
                            "confirm_bars": confirm_bars,
                            "min_sweep_ticks": min_sweep_ticks,
                            "stop_buffer_ticks": stop_buffer_ticks,
                            "min_rr": min_rr,
                            "num_trades": result["num_trades"],
                            "total_pnl": result["total_pnl"],
                            "win_rate": result["win_rate"],
                            "profit_factor": result["profit_factor"],
                            "sharpe": result["sharpe"],
                            "max_drawdown": result["max_drawdown"],
                            "avg_trade": result["avg_trade"],
                        }
                    )
    return rows


def main() -> int:
    global ENTRY_START, ENTRY_END
    parser = argparse.ArgumentParser()
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--confirm-bars", type=int, default=2)
    parser.add_argument("--min-sweep-ticks", type=int, default=2)
    parser.add_argument("--stop-buffer-ticks", type=int, default=2)
    parser.add_argument("--min-rr", type=float, default=0.8)
    parser.add_argument("--max-trades-per-day", type=int, default=1)
    parser.add_argument("--time-stop-bars", type=int, default=60)
    parser.add_argument("--entry-start", default=f"{ENTRY_START:%H:%M}")
    parser.add_argument("--entry-end", default=f"{ENTRY_END:%H:%M}")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--hdf-key", default="bars_1min_eth")
    parser.add_argument("--overnight-only", action="store_true")
    parser.add_argument("--prior-rth-only", action="store_true")
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    if args.overnight_only and args.prior_rth_only:
        parser.error("--overnight-only and --prior-rth-only are mutually exclusive")
    if args.long_only and args.short_only:
        parser.error("--long-only and --short-only are mutually exclusive")
    side = "long" if args.long_only else "short" if args.short_only else "both"
    ENTRY_START = _parse_time(args.entry_start)
    ENTRY_END = _parse_time(args.entry_end)

    bars = _load_bars(args.data_path, args.hdf_key)
    logger.info("Loaded %s 1m ETH bars: %s -> %s", f"{len(bars):,}", bars.index[0], bars.index[-1])

    if args.grid:
        rows = _grid(bars, args.contracts)
        rows_sorted = sorted(
            rows,
            key=lambda x: (
                x["total_pnl"],
                x["sharpe"],
                -(abs(x["max_drawdown"]) if x["max_drawdown"] is not None else 999999),
            ),
            reverse=True,
        )
        out = {
            "strategy": "MNQ liquidity sweep reclaim grid",
            "contracts": args.contracts,
            "rows": rows,
            "top_by_pnl": rows_sorted[:20],
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print("\nTop 12 parameter sets by PnL")
        print(pd.DataFrame(rows_sorted[:12]).to_string(index=False))
        print(f"\nSaved {RESULTS_PATH}")
        return 0

    result = run_backtest(
        bars,
        contracts=args.contracts,
        confirm_bars=args.confirm_bars,
        min_sweep_ticks=args.min_sweep_ticks,
        stop_buffer_ticks=args.stop_buffer_ticks,
        min_rr=args.min_rr,
        max_trades_per_day=args.max_trades_per_day,
        time_stop_bars=args.time_stop_bars,
        use_prior_rth=not args.overnight_only,
        use_overnight=not args.prior_rth_only,
        side=side,
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("  MNQ liquidity sweep + reclaim")
    print("=" * 72)
    print(f"  Period       : {result['period']}")
    print(f"  Contracts    : {args.contracts}")
    print(f"  Trades       : {result['num_trades']}")
    print(f"  Total PnL    : ${result['total_pnl']:,.2f}")
    print(f"  Win rate     : {result['win_rate']:.1%}")
    print(f"  Profit factor: {result['profit_factor']}")
    print(f"  Sharpe       : {result['sharpe']}")
    print(f"  Max drawdown : ${result['max_drawdown']:,.2f}")
    print(f"  Avg trade    : ${result['avg_trade']:,.2f}")
    print(f"  Levels       : {result['level_counts']}")
    print(f"  Sides        : {result['side_counts']}")
    print(f"  Saved        : {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
