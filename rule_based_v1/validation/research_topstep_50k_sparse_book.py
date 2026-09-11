"""Evaluate the two surviving sparse premia under current Topstep 50K rules.

This is an account-path study, not a new alpha search.  It rebuilds Monday RTH
and the Topstep-legal portion of pre-FOMC drift from the corrected MNQ tape,
uses one fixed $250-per-contract stop, and block-bootstraps 13-week attempts.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd

NY = "America/New_York"
POINT_VALUE = 2.0
ROUND_TRIP_COST = 2.05
STOP_USD_PER_CONTRACT = 250.0
TARGET = 3_000.0
MAX_LOSS_LIMIT = 2_000.0
CONSISTENCY = 0.50


@dataclass(frozen=True)
class Trade:
    day: pd.Timestamp
    leg: str
    pnl: float
    mae: float


def account_floor(peak: float) -> float:
    """EOD-trailing MLL, locked at the starting balance once reached."""
    return min(0.0, max(-MAX_LOSS_LIMIT, peak - MAX_LOSS_LIMIT))


def stopped_long(
    bars: pd.DataFrame,
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    *,
    stop_usd: float = STOP_USD_PER_CONTRACT,
    entry_tolerance_min: int = 3,
    exit_tolerance_min: int = 90,
) -> tuple[float, float] | None:
    """One-contract long with conservative gap-through stop execution."""
    i = bars.index.searchsorted(entry_ts, side="left")
    j = bars.index.searchsorted(exit_ts, side="right") - 1
    if i >= len(bars) or j <= i:
        return None
    if bars.index[i] - entry_ts > pd.Timedelta(minutes=entry_tolerance_min):
        return None
    if exit_ts - bars.index[j] > pd.Timedelta(minutes=exit_tolerance_min):
        return None
    segment = bars.iloc[i : j + 1]
    entry = float(segment.iloc[0]["open"])
    stop_price = entry - stop_usd / POINT_VALUE
    hits = np.flatnonzero(segment["low"].to_numpy(dtype=float) <= stop_price)
    if len(hits):
        k = int(hits[0])
        fill = min(stop_price, float(segment.iloc[k]["open"]))
        before = segment.iloc[:k]
        worst_price = min(
            fill,
            float(before["low"].min()) if len(before) else entry,
        )
        exit_price = fill
    else:
        worst_price = float(segment["low"].min())
        exit_price = float(segment.iloc[-1]["close"])
    pnl = (exit_price - entry) * POINT_VALUE - ROUND_TRIP_COST
    mae = (worst_price - entry) * POINT_VALUE
    return pnl, mae


def build_events(bars: pd.DataFrame, fomc_calendar: pd.DataFrame) -> pd.DataFrame:
    bars = bars.tz_convert(NY).sort_index()
    rows: list[dict] = []

    for monday in pd.date_range(bars.index.min().date(), bars.index.max().date(), freq="W-MON"):
        day = monday.date()
        entry = pd.Timestamp.combine(day, time(9, 31)).tz_localize(NY)
        exit_ = pd.Timestamp.combine(day, time(15, 59)).tz_localize(NY)
        result = stopped_long(bars, entry, exit_, exit_tolerance_min=3)
        if result is not None:
            rows.append({"day": pd.Timestamp(day), "leg": "monday_rth", "pnl": result[0], "mae": result[1]})

    for value in fomc_calendar["announcement_date"]:
        decision = pd.Timestamp(value)
        if decision.date() < bars.index.min().date() or decision.date() > bars.index.max().date():
            continue
        prior = decision - pd.Timedelta(days=1)
        # Topstep requires the prior trading day to end at 16:10 ET.  Entering
        # after the 18:00 ET reopen keeps the position inside one trading day.
        entry = pd.Timestamp.combine(prior.date(), time(18, 1)).tz_localize(NY)
        exit_ = pd.Timestamp.combine(decision.date(), time(13, 55)).tz_localize(NY)
        result = stopped_long(bars, entry, exit_)
        if result is not None:
            rows.append({"day": decision.normalize(), "leg": "fomc_legal", "pnl": result[0], "mae": result[1]})

    out = pd.DataFrame(rows).sort_values(["day", "leg"]).reset_index(drop=True)
    out["year"] = out["day"].dt.year
    out["week"] = out["day"].dt.to_period("W-SUN").astype(str)
    return out


def historical_summary(events: pd.DataFrame) -> dict:
    result: dict[str, dict] = {}
    for leg, group in events.groupby("leg"):
        pnl = group["pnl"]
        sd = float(pnl.std(ddof=1))
        result[leg] = {
            "n": int(len(group)),
            "mean_per_contract": float(pnl.mean()),
            "win_rate": float((pnl > 0).mean()),
            "t_stat": float(pnl.mean() / (sd / np.sqrt(len(pnl)))),
            "worst_pnl": float(pnl.min()),
            "worst_mae": float(group["mae"].min()),
            "yearly_mean": {str(int(y)): float(v) for y, v in group.groupby("year")["pnl"].mean().items()},
        }
    return result


def evaluate_sequence(sequence: list[tuple[float, float]]) -> tuple[str, int, float]:
    """Apply MLL and consistency to `(pnl, mae)` daily outcomes."""
    balance = 0.0
    peak = 0.0
    floor = account_floor(peak)
    best_day = 0.0
    for n, (pnl, mae) in enumerate(sequence, start=1):
        if balance + mae <= floor:
            return "bust", n, balance + mae
        balance += pnl
        if balance <= floor:
            return "bust", n, balance
        best_day = max(best_day, pnl)
        peak = max(peak, balance)
        floor = account_floor(peak)
        if balance >= TARGET and best_day <= CONSISTENCY * balance:
            return "pass", n, balance
    return "timeout", len(sequence), balance


def weekly_blocks(events: pd.DataFrame) -> list[list[Trade]]:
    start = events["day"].min().to_period("W-SUN").start_time
    stop = events["day"].max().to_period("W-SUN").start_time
    weeks = pd.date_range(start, stop, freq="7D")
    by_week = {k: g for k, g in events.groupby(events["day"].dt.to_period("W-SUN").dt.start_time)}
    out: list[list[Trade]] = []
    for week in weeks:
        g = by_week.get(week)
        if g is None:
            out.append([])
        else:
            out.append([Trade(r.day, r.leg, float(r.pnl), float(r.mae)) for r in g.itertuples()])
    return out


def simulate(
    events: pd.DataFrame,
    *,
    contracts: int,
    weeks: int = 13,
    paths: int = 50_000,
    block_weeks: int = 4,
    decay: float = 1.0,
    seed: int = 20260910,
) -> dict:
    source = weekly_blocks(events)
    if len(source) < block_weeks:
        raise ValueError("not enough history for requested block")
    leg_means = events.groupby("leg")["pnl"].mean().to_dict()
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(weeks / block_weeks)
    status: list[str] = []
    pass_events: list[int] = []
    endings: list[float] = []
    for _ in range(paths):
        starts = rng.integers(0, len(source) - block_weeks + 1, size=n_blocks)
        sampled: list[list[Trade]] = []
        for s in starts:
            sampled.extend(source[int(s) : int(s) + block_weeks])
        sequence: list[tuple[float, float]] = []
        for week in sampled[:weeks]:
            for trade in week:
                adjusted = (trade.pnl - leg_means[trade.leg]) + decay * leg_means[trade.leg]
                sequence.append((adjusted * contracts, trade.mae * contracts))
        outcome, n, ending = evaluate_sequence(sequence)
        status.append(outcome)
        endings.append(ending)
        if outcome == "pass":
            pass_events.append(n)
    state = np.asarray(status)
    return {
        "contracts": contracts,
        "decay": decay,
        "p_pass": float(np.mean(state == "pass")),
        "p_bust": float(np.mean(state == "bust")),
        "p_timeout": float(np.mean(state == "timeout")),
        "median_event_to_pass": float(np.median(pass_events)) if pass_events else None,
        "mean_ending_balance": float(np.mean(endings)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", type=Path, required=True)
    ap.add_argument("--fomc", type=Path, required=True)
    ap.add_argument("--out-events", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    ap.add_argument("--paths", type=int, default=50_000)
    ap.add_argument("--weeks", type=int, nargs="+", default=[13, 26, 52])
    args = ap.parse_args()

    bars = pd.read_parquet(args.bars)
    calendar = pd.read_csv(args.fomc)
    events = build_events(bars, calendar)
    simulations = [
        {"horizon_weeks": weeks, **simulate(events, contracts=q, weeks=weeks, paths=args.paths, decay=decay)}
        for weeks in args.weeks
        for decay in (1.0, 0.5, 0.0)
        for q in (1, 2, 3, 4, 5)
    ]
    result = {
        "rules": {"target": TARGET, "mll": MAX_LOSS_LIMIT, "consistency": CONSISTENCY},
        "execution": {"contracts": "grid 1..5", "stop_usd_per_contract": STOP_USD_PER_CONTRACT, "round_trip_cost": ROUND_TRIP_COST},
        "horizon_weeks": args.weeks,
        "paths": args.paths,
        "historical": historical_summary(events),
        "simulations": simulations,
    }
    args.out_events.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(args.out_events, index=False)
    args.out_summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
