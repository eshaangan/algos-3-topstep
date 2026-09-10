"""Where could a viable edge live at all? A feasibility map, not a backtest.

WHY THIS EXISTS
---------------
`vps_disaster_stop_sizing_VERDICT.md` fixed the requirement: **~$520/wk at a size
whose worst single trade stays inside ~60% of the $3,000 MLL ($1,800)**. The book
earns $162/wk. Roughly 80 configurations have been tested and killed, one family
at a time.

Testing family 81 without first asking WHERE a qualifying edge could live is the
expensive way to do this. Every strategy is pinned by four numbers -- how often it
trades, how far price moves over its holding period, how deep the adverse
excursion goes, and what a round trip costs -- and those four already determine
the fraction of the move it would have to capture. If that fraction is absurd, the
family is dead before any signal is specified.

THE ARITHMETIC
--------------
For a family trading N times/week at S micros, capturing fraction f of the mean
absolute move M over its horizon:

    $/wk  =  N * S * (f * M  -  cost)
    S     =  floor(1800 / tail)          # tail-legal size, tail = adverse p05
    =>  f  =  (520 / (N * S) + cost) / M

`f` is the honest difficulty measure: the share of the available move the strategy
must convert into profit, after paying to cross the spread.

TWO CAVEATS THAT MAKE THIS MAP OPTIMISTIC, BOTH DELIBERATE
----------------------------------------------------------
1. **Size is set from the adverse p05, not the worst case.** So ~5% of trades
   breach the tail budget. Using the observed worst excursion instead would cut
   every `size` by roughly 3-5x and push most rows to EXCLUDED. The p05 basis is
   used because it matches how `research_disaster_stop.py` sizes with a stop in
   place; without a stop, read every row as an upper bound.
2. **Velocity assumes every block is tradeable.** A family that only fires on
   scheduled events trades far less often than its horizon allows, and its
   required `f` rises in proportion. The map bounds CONTINUOUSLY-traded families;
   event families are strictly harder than the row suggests.

WHAT THIS IS NOT
----------------
It makes no claim that an edge EXISTS anywhere on the map. It rules regions OUT.
An `open` cell means "not excluded by arithmetic", nothing more.

Usage:
    python3 rule_based_v1/validation/research_feasible_region.py \
        --out runs/feasible_region.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"

MNQ = {"path": "data/processed/mnq_1m_all.parquet", "ts": "ts",
       "tz": "America/Chicago", "pv": 2.0, "rt_cost": 2.06}

TARGET_PER_WEEK = 520.0
MLL = 3000.0
TAIL_BUDGET = 0.60 * MLL          # 1,800: worst trade must stay inside this
RTH_MINUTES = 390

# Horizons in minutes. 1950 = one RTH week; 15.5*60 = one overnight hold.
HORIZONS = (5, 15, 30, 60, 120, 240, 390, 930, 1950)


def load_px(cfg: dict) -> pd.Series:
    raw = pd.read_parquet(cfg["path"])
    idx = pd.to_datetime(raw[cfg["ts"]])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(cfg["tz"], ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(raw["close"].to_numpy(float), index=idx.dt.tz_convert(ET))
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def move_and_tail(px: pd.Series, horizon_min: int, pv: float
                  ) -> tuple[float, float, float, int]:
    """Mean |move|, adverse p05, worst adverse, and n, in dollars per micro.

    Blocks are non-overlapping and never straddle a session, so the statistics
    are not inflated by autocorrelation or contaminated by overnight gaps. The
    adverse excursion is measured along the path, not at the endpoint, because
    the trailing drawdown is charged on the path.
    """
    vals, idx = px.to_numpy(float), px.index
    day = idx.normalize()
    moves, adverse = [], []
    start = 0
    n = len(vals)
    while start < n:
        end_day = np.searchsorted(day, day[start], side="right")
        seg = vals[start:end_day]
        for i in range(0, len(seg) - horizon_min, horizon_min):
            w = seg[i:i + horizon_min + 1]
            if len(w) < horizon_min + 1:
                break
            moves.append(abs(w[-1] - w[0]) * pv)
            adverse.append((w.min() - w[0]) * pv)
        start = end_day
    if len(moves) < 50:
        return np.nan, np.nan, np.nan, len(moves)
    return (float(np.mean(moves)), float(np.quantile(adverse, 0.05)),
            float(np.min(adverse)), len(moves))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    px = load_px(MNQ)
    cost = MNQ["rt_cost"]

    print(f"MNQ feasibility map. Target ${TARGET_PER_WEEK:.0f}/wk, "
          f"tail budget ${TAIL_BUDGET:,.0f} (60% of ${MLL:,.0f} MLL), "
          f"round trip ${cost:.2f}/micro")
    print(f"tape {px.index.min().date()} .. {px.index.max().date()}\n")
    print(f"{'horizon':>8} {'trades/wk':>10} {'mean|move|':>11} {'adv p05':>9} "
          f"{'size':>5} {'$/wk @f=5%':>11} {'f needed':>9} {'verdict':>10}")

    rows = []
    for h in HORIZONS:
        m, p05, worst, n = move_and_tail(px, h, MNQ["pv"])
        if not np.isfinite(m):
            continue
        # Non-overlapping trades available per week within RTH.
        per_week = max(1.0, (RTH_MINUTES / h) * 5.0)
        tail = abs(p05)
        size = max(1, int(TAIL_BUDGET // tail)) if tail > 0 else 1
        # What f delivers the target at this velocity and size?
        f_needed = (TARGET_PER_WEEK / (per_week * size) + cost) / m
        # What the target would pay at a plausible-but-good f = 5%.
        wk_at_5 = per_week * size * (0.05 * m - cost)

        verdict = ("EXCLUDED" if f_needed > 0.25 else
                   "hard" if f_needed > 0.10 else "open")
        print(f"{h:>7}m {per_week:>10.1f} ${m:>10,.2f} ${p05:>8,.0f} "
              f"{size:>5} ${wk_at_5:>10,.0f} {f_needed:>8.1%} {verdict:>10}")
        rows.append({"horizon_min": h, "trades_per_week": per_week,
                     "mean_abs_move": m, "adverse_p05": p05,
                     "adverse_worst": worst, "n_blocks": n,
                     "tail_legal_size": size, "f_needed": f_needed,
                     "usd_per_week_at_f5pct": wk_at_5, "verdict": verdict})

    print("\nf needed = share of the mean absolute move the strategy must convert")
    print("to profit, net of crossing the spread, to reach the target.")
    print("EXCLUDED = f > 25% (not a research target). open = f <= 10%.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
