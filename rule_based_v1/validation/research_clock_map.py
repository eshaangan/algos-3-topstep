"""Full 24h x weekday clock map on the long ETH tape — a discovery sweep.

Every time-of-day result in the ledger was produced on either 6.5y of MNQ or an
RTH-only ES series. This runs the whole 24-hour clock, by weekday, on 15.5 years
of front-month ES ETH, which is 2.4x the sample and includes the overnight hours
that RTH data cannot express.

This is explicitly a DISCOVERY pass, not a GO test. It scans many cells, so any
survivor is a candidate for a single pre-registered shot on MNQ, not a result.
Three things are reported for every cell so that nothing hides:

  gross  -- the raw move, so a real signal is not buried under a constant cost
  net    -- gross minus the round turn, which is what a GO decision needs
  halves -- the same cell in each era half; a cell that flips sign is noise

The cost line matters more than it looks. On 2 MES a round turn is 1.50 ES points
= 0.022% at ES 6900. On 2 MNQ it is 2.24 NQ points = 0.009% at NQ 25000. MNQ needs
less than half the relative move to break even, which is why confirmation belongs
there even though the long history is here.

Usage:
    python3 rule_based_v1/validation/research_clock_map.py --hours 1 2 3 4 --out runs/clock_es.json
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ET = "America/New_York"
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def load_px(path: Path, ts_col: str, tz: str | None) -> pd.Series:
    raw = pd.read_parquet(path)
    idx = pd.to_datetime(raw[ts_col])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(raw["close"].to_numpy(float), index=idx.dt.tz_convert(ET))
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


class Tape:
    """Pre-flattened tape.

    A tz-aware DatetimeIndex returns an OBJECT array from .to_numpy(), which turns
    every searchsorted into Python-level Timestamp comparisons. Converting once to
    UTC-naive datetime64 keeps the scans in C -- the difference is minutes per cell
    versus milliseconds.
    """

    def __init__(self, px: pd.Series):
        self.vals = px.to_numpy(dtype=float)
        self.keys = px.index.tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]")
        days = pd.DatetimeIndex(np.unique(px.index.normalize()))
        self.days = days
        self.day_keys = days.tz_convert("UTC").tz_localize(None).to_numpy("datetime64[ns]")

    def offsets(self, hours: float) -> np.ndarray:
        return self.day_keys + np.timedelta64(int(hours * 60), "m")


def holds(tape: "Tape", start_h: int, span_h: int) -> pd.DataFrame:
    """Long from start_h:00 to (start_h+span_h):00 ET on each session day."""
    keys, vals = tape.keys, tape.vals
    t0s, t1s = tape.offsets(start_h), tape.offsets(start_h + span_h)
    i0 = np.searchsorted(keys, t0s, "left")
    i1 = np.searchsorted(keys, t1s, "right") - 1
    ok = (i0 < len(keys)) & (i1 >= 0) & (i1 > i0)
    i0c, i1c = np.clip(i0, 0, len(keys) - 1), np.clip(i1, 0, len(keys) - 1)
    ok &= (keys[i0c] - t0s) <= np.timedelta64(10, "m")
    ok &= (t1s - keys[i1c]) <= np.timedelta64(30, "m")
    # worst adverse excursion inside each hold: the $3k buffer cares about path
    mae = np.full(len(i0c), np.nan)
    for j in np.nonzero(ok)[0]:
        a, b = i0c[j], i1c[j]
        mae[j] = vals[a:b + 1].min() - vals[a]
    out = pd.DataFrame({"day": tape.days[ok], "entry": vals[i0c][ok],
                        "exit": vals[i1c][ok], "mae_pts": mae[ok]})
    out["points"] = out["exit"] - out["entry"]
    out["dow"] = out["day"].dt.dayofweek
    return out[out["dow"] < 5]


def cell_stats(sub: pd.DataFrame, point_value: float, contracts: int, cost: float) -> dict | None:
    """Score a cell LONG and SHORT separately.

    The fee is paid in both directions, so short_net is -gross - cost, NOT the
    negation of long_net. Ranking a short by the long's t-stat therefore reads
    pure cost drag as a short edge -- a cell where the long loses $16 to fees can
    have a short that makes $1 at t=0.3. Score each side on its own PnL.
    """
    if len(sub) < 60:
        return None
    gross = sub["points"] * point_value * contracts
    long_net, short_net = gross - cost, -gross - cost
    t_long = float(stats.ttest_1samp(long_net, 0).statistic)
    t_short = float(stats.ttest_1samp(short_net, 0).statistic)
    side = "long" if t_long >= t_short else "short"
    best = long_net if side == "long" else short_net
    mae_p05 = float(sub["mae_pts"].quantile(0.05) * point_value * contracts)
    eff = float(best.mean() / abs(mae_p05)) if mae_p05 < 0 else float("nan")
    return {"n": int(len(sub)), "gross": float(gross.mean()),
            "long_net": float(long_net.mean()), "short_net": float(short_net.mean()),
            "t_long": t_long, "t_short": t_short,
            "side": side, "net": float(best.mean()), "t_net": max(t_long, t_short),
            "wr": float((best > 0).mean()), "mae_p05": mae_p05, "buffer_eff": eff}


def main() -> None:
    ap = argparse.ArgumentParser(description="24h x weekday clock map")
    ap.add_argument("--bars", type=Path, default=Path("data/processed/es_1min_eth_frontmonth.parquet"))
    ap.add_argument("--ts-col", default="et")
    ap.add_argument("--tz", default=None)
    ap.add_argument("--point-value", type=float, default=5.0)
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--tick-size", type=float, default=0.25)
    ap.add_argument("--hours", type=int, nargs="+", default=[1, 2, 3, 6],
                    help="hold lengths in hours to scan")
    ap.add_argument("--split", default="2018-01-01")
    ap.add_argument("--min-t", type=float, default=2.5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cost = 2 * args.contracts * (0.62 + 1.0 * args.tick_size * args.point_value)
    px = load_px(args.bars, args.ts_col, args.tz)
    split = pd.Timestamp(args.split, tz=ET)
    print(f"bars {len(px):,}  {px.index.min().date()} .. {px.index.max().date()}  "
          f"cost ${cost:.2f}  {args.contracts} contracts  split {args.split}")

    cells, n_trials = [], 0
    tape = Tape(px)
    for span in args.hours:
        for start in range(0, 24):
            h = holds(tape, start, span)
            if h.empty:
                continue
            for dow in [None] + list(range(5)):
                sub = h if dow is None else h[h["dow"] == dow]
                s_all = cell_stats(sub, args.point_value, args.contracts, cost)
                if s_all is None:
                    continue
                n_trials += 1
                a = sub[sub["day"] < split]
                b = sub[sub["day"] >= split]
                s_a = cell_stats(a, args.point_value, args.contracts, cost)
                s_b = cell_stats(b, args.point_value, args.contracts, cost)
                # halves must be scored on the SAME side the full sample chose,
                # otherwise each half silently picks its own winner
                side = s_all["side"]
                ta = (s_a or {}).get("t_long" if side == "long" else "t_short")
                tb = (s_b or {}).get("t_long" if side == "long" else "t_short")
                cells.append({"span": span, "start": start,
                              "dow": "ALL" if dow is None else DOW[dow],
                              **s_all,
                              "t_a": ta, "t_b": tb})

    frame = pd.DataFrame(cells)
    # 17:00-18:00 ET is the CME maintenance break: no trading, so gross is ~0 and the
    # cell degenerates to "you paid the fee and nothing moved" -- a huge |t| that is
    # pure cost drag, not signal. Same for any cell whose gross cannot clear the fee.
    frame = frame[frame["start"] != 17]
    frame["edge"] = frame["gross"].abs() - cost          # what is left after the round turn
    frame["t_best"] = frame["t_net"]                     # already the better side, signed
    tradeable = frame[frame["edge"] > 0].copy()

    print(f"\nscanned {n_trials} cells "
          f"({len(args.hours)} spans x 24 starts x 6 day-groups, minus thin cells)")
    print(f"at {n_trials} trials, |t| ~{stats.norm.ppf(1 - 0.025 / max(n_trials, 1)):.2f} "
          f"is the Bonferroni bar for 5% familywise")
    print(f"cells whose gross move exceeds the ${cost:.2f} round turn at all: "
          f"{len(tradeable)} of {len(frame)}\n")

    if tradeable.empty:
        print("NOTHING clears the cost line. No candidate.")
    else:
        print("TOP 15 CELLS THAT CLEAR COST, by |t| on net  (t_a/t_b = era halves)")
        print(f"{'span':>5}{'start':>6}{'dow':>5}{'side':>6}{'n':>6}{'gross$':>9}"
              f"{'edge$':>8}{'t':>7}{'t_a':>7}{'t_b':>7}{'eff':>7}  verdict")
        for r in tradeable.sort_values("t_best", ascending=False).head(15).itertuples():
            ta = r.t_a if r.t_a is not None else float("nan")
            tb = r.t_b if r.t_b is not None else float("nan")
            agree = ta == ta and tb == tb and ta > 0 and tb > 0
            strong = abs(r.t_net) >= args.min_t
            verdict = ("CANDIDATE" if agree and strong else
                       "halves disagree" if strong else "below gate")
            print(f"{r.span:>5}{r.start:>6}{r.dow:>5}{r.side:>6}{r.n:>6}{r.gross:>9.2f}"
                  f"{r.edge:>8.2f}{r.t_net:>7.2f}{ta:>7.2f}{tb:>7.2f}"
                  f"{r.buffer_eff:>7.3f}  {verdict}")

        keep = tradeable[tradeable["t_best"] >= args.min_t].copy()
        keep = keep[keep["t_a"].notna() & keep["t_b"].notna()]
        keep = keep[(keep["t_a"] > 0) & (keep["t_b"] > 0)]
        print(f"\nCANDIDATES (clear cost, |t|>={args.min_t}, both era halves agree): {len(keep)}")
        if len(keep):
            print(keep.sort_values("t_best", ascending=False)
                  [["span", "start", "dow", "side", "n", "gross", "edge",
                    "t_net", "t_a", "t_b", "buffer_eff"]].to_string(index=False))
        else:
            print("  none -- nothing on the 24h clock survives cost + era agreement.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"n_trials": n_trials, "cost": cost,
                                    "cells": frame.to_dict("records")}, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
