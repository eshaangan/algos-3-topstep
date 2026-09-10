"""Is an MNQ/MES relative-value spread cost-viable, before any edge is claimed?

WHY THIS IS THE FIRST QUESTION
------------------------------
The Track B survey concluded the binding constraint is not a shortage of alpha
but single-event tail risk against a $2-3k trailing drawdown: the book is capped
at 1 micro on its highest-velocity leg, which yields ~$132-232/wk against the
$375-500/wk the deadline needs. A market-neutral spread strips the shared index
beta, so adverse excursion per position falls, which is what would permit size.

But that argument has an obvious way of being wrong, and it must be checked
BEFORE any signal work: the spread costs MORE to trade than the outright, because
it has two legs. Cost is fixed per round trip while volatility is what shrinks.
The spread is only worth pursuing if volatility falls FASTER than cost rises.

    outright MNQ round trip : $2.06/contract   (measured 2026-09-04: mean spread
                              1.64 ticks = $0.82, commission $1.24)
    MES round trip          : $2.49/contract   (1-tick spread = $1.25 + $1.24)
    1 MNQ + h MES package   : 2.06 + h * 2.49

At h ~ 2 the package costs ~$7.04, i.e. **3.4x the outright**. So residual
volatility must fall by more than 3.4x for the spread to be the better vehicle.
This script measures that ratio and nothing else. It makes no claim that an edge
exists in the residual.

THE METRIC
----------
`cost_ratio` = round-trip cost / mean absolute dollar move over the holding
horizon. It is the same quantity the ledger used to kill the 15-minute families
at 30-70%, and the ~2-3% of the surviving overnight windows. Lower is better.

The hedge ratio h is chosen to MINIMISE RESIDUAL VARIANCE in dollar terms, fit
on a dev era and applied unchanged to a val era, so the reported residual is not
flattered by fitting the hedge to the same data it is measured on.

Usage:
    python3 rule_based_v1/validation/research_spread_cost_ratio.py \
        --out runs/spread_cost_ratio.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ET = "America/New_York"

# Point values and measured round-trip costs, per contract.
MNQ_PV, MNQ_RT = 2.0, 2.06
MES_PV, MES_RT = 5.0, 2.49

HORIZONS_MIN = (1, 5, 15, 30, 60, 240, 960)


def load(path: str, ts_col: str, tz: str | None,
         shift_min: int = 0) -> pd.Series:
    """1-minute closes on an ET index, optionally shifted to fix bar labelling.

    THE TWO FEEDS DO NOT LABEL BARS THE SAME WAY, and it is not visible in either
    series alone. Measured on 2025-06 by scanning the cross-correlation over
    lags: MNQ agrees with ES at a **+59 minute** shift of its raw stamp, at
    corr 0.9411, and at essentially zero (0.006) with no shift.

        +60 min  = the Chicago-to-Eastern conversion
         -1 min  = MNQ bars are stamped at bar CLOSE, ES bars at bar OPEN

    Without the one-minute correction the measured 1-minute correlation is
    **0.014** instead of 0.941, which understates the achievable hedge by a
    factor of three and would have killed the spread on a measurement artifact.
    A whole-sample correlation of 0.56 looks plausible enough to pass unnoticed;
    it comes entirely from the ~0.3% of rows that straddle a session boundary,
    where both legs jump together.

    This only matters when COMBINING the two tapes. Each is internally
    consistent, so single-instrument results are unaffected.
    """
    raw = pd.read_parquet(path)
    idx = pd.to_datetime(raw[ts_col])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    idx = idx.dt.tz_convert(ET)
    if shift_min:
        idx = idx + pd.Timedelta(minutes=shift_min)
    s = pd.Series(raw["close"].to_numpy(float), index=idx)
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def rth_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    m = idx.hour * 60 + idx.minute
    return (m >= 9 * 60 + 30) & (m <= 16 * 60)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mnq", default="data/processed/mnq_1m_all.parquet")
    ap.add_argument("--es", default="data/processed/es_1min_eth_frontmonth.parquet")
    ap.add_argument("--dev-end", default="2023-12-31")
    ap.add_argument("--rth-only", action="store_true", default=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    # -1 min aligns MNQ close-stamped bars onto ES open-stamped labels.
    nq = load(a.mnq, "ts", "America/Chicago", shift_min=-1)
    es = load(a.es, "et", None)
    both = pd.DataFrame({"nq": nq, "es": es}).dropna()
    if a.rth_only:
        both = both[rth_mask(both.index)]
    print(f"aligned 1-min bars: {len(both):,}  "
          f"{both.index.min().date()} .. {both.index.max().date()}")

    # Dollar P&L per contract, per minute.
    d_nq = both["nq"].diff() * MNQ_PV
    d_es = both["es"].diff() * MES_PV
    frame = pd.DataFrame({"d_nq": d_nq, "d_es": d_es}).dropna()

    cut = pd.Timestamp(a.dev_end, tz=ET)
    dev = frame[frame.index <= cut]
    val = frame[frame.index > cut]
    if len(dev) < 10_000 or len(val) < 10_000:
        raise SystemExit("not enough data either side of the dev/val cut")

    # Variance-minimising hedge: h = cov(d_nq, d_es) / var(d_es), fit on DEV.
    h = float(np.cov(dev["d_nq"], dev["d_es"])[0, 1] / np.var(dev["d_es"]))
    corr = float(np.corrcoef(frame["d_nq"], frame["d_es"])[0, 1])
    print(f"\nhedge fit on dev (<= {a.dev_end}): h = {h:.3f} MES per 1 MNQ")
    print(f"minute-return correlation (full sample): {corr:.4f}")

    report = {"h_dev": h, "corr": corr, "rows": []}

    for h_used, label in ((h, f"h={h:.2f} (dev-fit)"),
                          (round(h), f"h={round(h)} (tradeable integer)")):
        cost_pkg = MNQ_RT + abs(h_used) * MES_RT
        print(f"\n=== SPREAD  long 1 MNQ / short {h_used:.2f} MES   "
              f"round trip ${cost_pkg:.2f}  [{label}] ===")
        print(f"{'horizon':>9} {'|move| outright':>16} {'ratio':>7} "
              f"{'|move| spread':>15} {'ratio':>7} {'vol cut':>8} {'verdict':>9}")

        for m in HORIZONS_MIN:
            # Sum minute P&L over the horizon, VAL era only (hedge is out of
            # sample there). Non-overlapping blocks so the mean absolute move is
            # not autocorrelation-inflated.
            # Blocks are built WITHIN a session. Reshaping the whole val array
            # would let a block straddle an overnight gap, which silently mixes
            # two different return distributions into the same statistic.
            nq_parts, es_parts = [], []
            for _, g in val.groupby(val.index.normalize()):
                k = len(g) // m
                if k == 0:
                    continue
                nq_parts.append(g["d_nq"].to_numpy()[:k * m].reshape(k, m).sum(1))
                es_parts.append(g["d_es"].to_numpy()[:k * m].reshape(k, m).sum(1))
            if not nq_parts:
                continue
            nq_blk = np.concatenate(nq_parts)
            es_blk = np.concatenate(es_parts)
            n_blocks = len(nq_blk)
            if n_blocks < 100:
                continue
            sp_blk = nq_blk - h_used * es_blk

            mv_out = float(np.mean(np.abs(nq_blk)))
            mv_spr = float(np.mean(np.abs(sp_blk)))
            r_out = MNQ_RT / mv_out if mv_out else np.nan
            r_spr = cost_pkg / mv_spr if mv_spr else np.nan
            vol_cut = mv_out / mv_spr if mv_spr else np.nan
            better = "SPREAD" if r_spr < r_out else "outright"

            print(f"{m:>7}m  ${mv_out:>14,.2f} {r_out:>6.1%} "
                  f"${mv_spr:>13,.2f} {r_spr:>6.1%} {vol_cut:>7.2f}x "
                  f"{better:>9}")
            report["rows"].append({
                "h": h_used, "horizon_min": m, "n_blocks": int(n_blocks),
                "abs_move_outright": mv_out, "cost_ratio_outright": r_out,
                "abs_move_spread": mv_spr, "cost_ratio_spread": r_spr,
                "vol_cut": vol_cut, "cheaper": better,
            })

    print("\nREAD THIS AS: the spread is only the better vehicle where its cost "
          f"ratio is LOWER.\nThe package costs {(MNQ_RT + round(h) * MES_RT) / MNQ_RT:.1f}x "
          "the outright, so it needs a bigger volatility cut than that to win.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
