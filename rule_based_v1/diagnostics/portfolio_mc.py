"""Reproducible Monte Carlo for the LucidFlex 100k event-strategy book.

The committed weekend series is expressed per TWO MNQ micros. This module keeps
that unit explicit and compares two sizing policies:

* constant_2: two micros from the first event.
* daemon_ladder: one micro until +$4,500 banked, then two micros.

The max-loss floor trails end-of-period equity by $3,000 until it reaches the
starting balance, where it locks. The pass test also enforces the 50% best-day
consistency rule.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TARGET = 6_000.0
MLL = 3_000.0
CONSISTENCY = 0.50


@dataclass(frozen=True)
class AccountRules:
    target: float = TARGET
    max_loss_limit: float = MLL
    consistency_fraction: float = CONSISTENCY
    floor_lock: bool = True


@dataclass(frozen=True)
class Policy:
    name: str
    low_scale: float
    high_scale: float
    rung_balance: float

    def scale(self, balance: float) -> float:
        return self.high_scale if balance >= self.rung_balance else self.low_scale


CONSTANT_2 = Policy("constant_2", low_scale=1.0, high_scale=1.0, rung_balance=0.0)
DAEMON_LADDER = Policy("daemon_ladder", low_scale=0.5, high_scale=1.0, rung_balance=4_500.0)


def parse_strategy(spec: str) -> tuple[str, Path, float]:
    """Parse NAME=PATH[:SCALE]. SCALE multiplies the file's pnl column."""
    if "=" not in spec:
        raise ValueError(f"strategy must be NAME=PATH[:SCALE], got {spec!r}")
    name, rhs = spec.split("=", 1)
    scale = 1.0
    path_text = rhs
    maybe_path, sep, maybe_scale = rhs.rpartition(":")
    if sep:
        try:
            scale = float(maybe_scale)
            path_text = maybe_path
        except ValueError:
            path_text = rhs
    return name, Path(path_text), scale


def load_book(specs: Sequence[str]) -> pd.DataFrame:
    """Load event CSVs and aggregate strategies that share an event date."""
    pieces: list[pd.DataFrame] = []
    for spec in specs:
        name, path, scale = parse_strategy(spec)
        frame = pd.read_csv(path)
        missing = {"day", "pnl"}.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns {sorted(missing)}")
        frame = frame[["day", "pnl"]].copy()
        frame["day"] = pd.to_datetime(frame["day"], errors="raise")
        frame["pnl"] = pd.to_numeric(frame["pnl"], errors="raise") * scale
        frame["strategy"] = name
        pieces.append(frame)
    if not pieces:
        raise ValueError("at least one strategy is required")
    all_events = pd.concat(pieces, ignore_index=True).sort_values("day")
    return all_events.groupby("day", as_index=False)["pnl"].sum()


def block_bootstrap(values: np.ndarray, *, n_paths: int, periods: int, block: int, seed: int) -> np.ndarray:
    """Sample contiguous blocks, preserving medium-run regime clustering."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < block:
        raise ValueError(f"need at least block={block} observations; got {len(values)}")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(periods / block)
    starts = rng.integers(0, len(values) - block + 1, size=(n_paths, n_blocks))
    out = np.empty((n_paths, n_blocks * block), dtype=float)
    offset = np.arange(block)
    for b in range(n_blocks):
        out[:, b * block : (b + 1) * block] = values[starts[:, b, None] + offset]
    return out[:, :periods]


def loss_floor(peak: float, rules: AccountRules) -> float:
    floor = max(-rules.max_loss_limit, peak - rules.max_loss_limit)
    return min(0.0, floor) if rules.floor_lock else floor


def run_path(raw_pnl: Iterable[float], *, policy: Policy, rules: AccountRules = AccountRules()) -> tuple[str, int, float]:
    balance = 0.0
    peak = 0.0
    best_period = float("-inf")
    for period, raw in enumerate(raw_pnl, start=1):
        pnl = float(raw) * policy.scale(balance)
        balance += pnl
        peak = max(peak, balance)
        best_period = max(best_period, pnl)
        if balance <= loss_floor(peak, rules):
            return "bust", period, balance
        if balance >= rules.target and best_period <= rules.consistency_fraction * balance:
            return "pass", period, balance
    return "timeout", period if "period" in locals() else 0, balance


def evaluate_paths(paths: np.ndarray, policy: Policy, rules: AccountRules) -> dict:
    results = [run_path(path, policy=policy, rules=rules) for path in paths]
    status = np.asarray([r[0] for r in results])
    durations = np.asarray([r[1] for r in results])
    endings = np.asarray([r[2] for r in results])
    passed = durations[status == "pass"]
    return {
        "policy": policy.name,
        "p_pass": float(np.mean(status == "pass")),
        "p_bust": float(np.mean(status == "bust")),
        "p_timeout": float(np.mean(status == "timeout")),
        "median_periods_to_pass": float(np.median(passed)) if len(passed) else None,
        "p90_periods_to_pass": float(np.quantile(passed, 0.90)) if len(passed) else None,
        "mean_ending_balance": float(np.mean(endings)),
    }


def historical_stats(book: pd.DataFrame) -> dict:
    p = book["pnl"].to_numpy(float)
    std = float(np.std(p, ddof=1))
    t = float(np.mean(p) / (std / np.sqrt(len(p)))) if std > 0 and len(p) > 1 else 0.0
    return {
        "n_events": int(len(p)),
        "start": str(book["day"].min().date()),
        "end": str(book["day"].max().date()),
        "mean_pnl_base_unit": float(np.mean(p)),
        "std_pnl_base_unit": std,
        "win_rate": float(np.mean(p > 0)),
        "t_stat": t,
        "worst_event": float(np.min(p)),
    }


def run_analysis(book: pd.DataFrame, *, n_paths: int, periods: int, block: int, seed: int, zero_drift: bool) -> dict:
    values = book["pnl"].to_numpy(float)
    if zero_drift:
        values = values - values.mean()
    paths = block_bootstrap(values, n_paths=n_paths, periods=periods, block=block, seed=seed)
    rules = AccountRules()
    return {
        "historical": historical_stats(book),
        "simulation": {
            "n_paths": n_paths,
            "periods": periods,
            "block": block,
            "seed": seed,
            "zero_drift": zero_drift,
            "rules": asdict(rules),
            "results": [evaluate_paths(paths, CONSTANT_2, rules), evaluate_paths(paths, DAEMON_LADDER, rules)],
        },
    }


def print_report(result: dict) -> None:
    h = result["historical"]
    s = result["simulation"]
    print(f"events={h['n_events']} {h['start']}..{h['end']} | mean=${h['mean_pnl_base_unit']:+.2f} | WR={h['win_rate']:.1%} | t={h['t_stat']:.2f} | worst=${h['worst_event']:,.0f}")
    print(f"bootstrap: {s['n_paths']:,} paths, {s['periods']} periods, block={s['block']}, zero_drift={s['zero_drift']}")
    print(f"{'policy':16s} {'P(pass)':>9s} {'P(bust)':>9s} {'median':>9s} {'p90':>9s}")
    for row in s["results"]:
        print(f"{row['policy']:16s} {row['p_pass']:9.1%} {row['p_bust']:9.1%} {str(row['median_periods_to_pass']):>9s} {str(row['p90_periods_to_pass']):>9s}")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    default = f"WK={root / 'data/processed/weekend_hold_strict.csv'}:1.0"
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", action="append", default=None, help="NAME=CSV[:SCALE]")
    ap.add_argument("--paths", type=int, default=50_000)
    ap.add_argument("--periods", type=int, default=260)
    ap.add_argument("--block", type=int, default=13)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--zero-drift", action="store_true")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    book = load_book(args.strategy or [default])
    result = run_analysis(book, n_paths=args.paths, periods=args.periods, block=args.block, seed=args.seed, zero_drift=args.zero_drift)
    print_report(result)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
