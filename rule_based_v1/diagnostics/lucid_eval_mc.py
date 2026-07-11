"""LucidFlex 100k eval + funded-phase Monte Carlo for a once-daily strategy.

Hardened vs the first pass:
  - exact daily $ series (84 hist tape days + exact recorder dailies via --recorder-csv)
  - BLOCK bootstrap (5-day contiguous blocks) — regimes cluster, iid flatters
  - zero-drift stress: same paths with the pool demeaned (edge could be 0)
  - funded-phase extraction model -> expected lifetime payout per funded account

Account rules modeled (LucidFlex 100k, verified vs live RMS 2026-07-09):
  eval: +$6,000 target, $3,000 EOD-trailing MLL, consistency: best day <= 50%
  funded: same trailing; payout min($2,500, 50% of balance) when balance >= $2,600
          and >= 5 qualifying days (day pnl >= +$100) since last payout; 90% split.
  position limit: 60 micros (does not bind at the sizes tested).

    python rule_based_v1/diagnostics/lucid_eval_mc.py --recorder-csv /tmp/rip_recorder_daily.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

TARGET, MLL = 6000.0, 3000.0
PAYOUT_CAP, PAYOUT_MIN_BAL, QUAL_DAY, QUAL_N, SPLIT = 2500.0, 2600.0, 100.0, 5, 0.90
BLOCK = 5
MAX_DAYS = 250
FUNDED_MAX_DAYS = 500


def block_paths(pool: np.ndarray, rng, n_paths: int, max_days: int) -> np.ndarray:
    """Sample contiguous BLOCK-day chunks (preserves short-run regime clustering)."""
    n = len(pool)
    starts = rng.integers(0, n - BLOCK + 1, size=(n_paths, max_days // BLOCK + 1))
    out = np.empty((n_paths, (max_days // BLOCK + 1) * BLOCK))
    for b in range(starts.shape[1]):
        idx = starts[:, b][:, None] + np.arange(BLOCK)[None, :]
        out[:, b * BLOCK:(b + 1) * BLOCK] = pool[idx]
    return out[:, :max_days]


def run_eval(daily: np.ndarray) -> tuple:
    """One eval path -> (passed, days_used, bust)."""
    bal = peak = best = 0.0
    for d in range(len(daily)):
        pnl = daily[d]
        bal += pnl
        best = max(best, pnl)
        if bal <= peak - MLL:
            return False, d + 1, True
        peak = max(peak, bal)
        if bal >= TARGET and best <= 0.5 * bal:
            return True, d + 1, False
    return False, len(daily), False


def run_funded(daily: np.ndarray) -> float:
    """One funded path -> total NET payouts extracted before bust/horizon."""
    bal = peak = 0.0
    qual = 0
    total = 0.0
    for pnl in daily:
        bal += pnl
        if bal <= peak - MLL:
            break
        peak = max(peak, bal)
        if pnl >= QUAL_DAY:
            qual += 1
        if bal >= PAYOUT_MIN_BAL and qual >= QUAL_N:
            pay = min(PAYOUT_CAP, 0.5 * bal)
            total += pay * SPLIT
            bal -= pay
            qual = 0
    return total


def table(pool: np.ndarray, label: str, sizes, n_paths=20000, seed=7):
    rng = np.random.default_rng(seed)
    print(f"\n=== {label}: pool n={len(pool)} mean=${pool.mean():+.1f} std=${pool.std():.1f} (per 2 micros) ===")
    print(f"{'micros':>7} {'P(pass)':>8} {'P(bust)':>8} {'med days':>9} {'E[payout|funded]':>17} {'EV full cycle':>14}")
    for nc in sizes:
        scale = nc / 2.0
        ep = block_paths(pool, rng, n_paths, MAX_DAYS) * scale
        res = [run_eval(ep[i]) for i in range(n_paths)]
        p_pass = np.mean([r[0] for r in res])
        p_bust = np.mean([r[2] for r in res])
        dpass = [r[1] for r in res if r[0]]
        # funded phase on fresh paths at the same size
        fp = block_paths(pool, rng, 4000, FUNDED_MAX_DAYS) * scale
        payouts = [run_funded(fp[i]) for i in range(4000)]
        e_pay = float(np.mean(payouts))
        ev = p_pass * e_pay
        print(f"{nc:>7} {p_pass:>8.1%} {p_bust:>8.1%} {int(np.median(dpass)) if dpass else '—':>9} "
              f"${e_pay:>15,.0f} ${ev:>12,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recorder-csv", help="csv with column pnl (exact recorder dailies, per 2 micros)")
    ap.add_argument("--hist-csv", default="/tmp/morning_rip_hist_daily.csv")
    ap.add_argument("--variant", default="preston 8/15/8")
    args = ap.parse_args()
    h = pd.read_csv(args.hist_csv)[args.variant].dropna().to_numpy()
    parts = [h]
    if args.recorder_csv:
        parts.append(pd.read_csv(args.recorder_csv)["pnl"].to_numpy())
    pool = np.concatenate(parts)
    sizes = (10, 16, 20, 24, 30, 40)
    table(pool, "AS MEASURED (drift +$" + f"{pool.mean():.1f}/day)", sizes)
    table(pool - pool.mean(), "ZERO-DRIFT STRESS (edge = exactly 0)", sizes)


if __name__ == "__main__":
    main()
