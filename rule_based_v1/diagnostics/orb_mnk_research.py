"""Micro Nikkei (USD) MNK — ORB research / small grid sweep.

Uses the same U.S. RTH 5-minute session as MNQ ORB (09:30–16:00 ET bars).
CME Micro Nikkei (USD): Globex symbol MNK; Databento continuous MNK.c.0 on GLBX.MDP3.

Run from repo root:
  python rule_based_v1/diagnostics/orb_mnk_research.py
  python rule_based_v1/diagnostics/orb_mnk_research.py --backtest-only --start 2026-01-01

Requires DATABENTO_API_KEY in .env for fetch (unless cached HDF exists).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in (str(ROOT), str(RBV1)):
    if p not in sys.path:
        sys.path.insert(0, p)

from diagnostics.ifr_backtest import (  # noqa: E402
    OR_END_TIME,
    ENTRY_CUTOFF,
    MIN_OR_BARS,
    PT_MULT,
    SL_MULT,
    TIME_STOP_BARS,
    ATR_PERIOD,
    _filter_ytd,
    _load_bars,
    _topstep_check,
    fetch_and_save,
    run_backtest_instrument,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MNK_PATH = ROOT / "data" / "processed" / "mnk_2026ytd_5min.h5"
RESULTS_JSON = ROOT / "rule_based_v1" / "diagnostics" / "orb_mnk_sweep_results.json"

# CME Micro Nikkei (USD): 0.5 USD × index / pt; min tick 5 pts = $2.50
MNK_SPEC = {
    "symbol": "MNK.c.0",
    "point_value": 0.5,
    "tick_size": 5.0,
    "tick_value": 2.5,
    "commission": 0.62,
    "slippage_ticks": 1,
}


def _sweep(
    bars: pd.DataFrame,
    n_contracts: int,
) -> list[dict]:
    rows: list[dict] = []
    or_ends = ["10:00", "10:04"]
    min_range_grid = [0.25, 0.3, 0.35, 0.4, 0.45]
    min_bars_opts = [6, 7]

    for or_end in or_ends:
        for mra in min_range_grid:
            for mb in min_bars_opts:
                r = run_backtest_instrument(
                    bars,
                    MNK_SPEC,
                    n_contracts,
                    or_end_time=or_end,
                    min_or_bars=mb,
                    min_range_atr=mra,
                    entry_cutoff=ENTRY_CUTOFF,
                    pt_mult=PT_MULT,
                    sl_mult=SL_MULT,
                    time_stop_bars=TIME_STOP_BARS,
                    atr_period=ATR_PERIOD,
                    long_only=True,
                    session_start_time="09:30",
                )
                tc = _topstep_check(r)
                rows.append(
                    {
                        "or_end_time": or_end,
                        "min_or_bars": mb,
                        "min_range_atr": mra,
                        "n_contracts": n_contracts,
                        "num_trades": r["num_trades"],
                        "total_pnl": r["total_pnl"],
                        "sharpe": r["sharpe"],
                        "max_drawdown": r["max_drawdown"],
                        "win_rate": r.get("win_rate"),
                        "combine_pass_likely": tc["combine_pass_likely"],
                        "worst_day": tc["worst_day"],
                    }
                )
    return rows


def _pick_best(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    viable = [x for x in rows if x["num_trades"] >= 8]
    pool = viable if viable else rows
    return max(pool, key=lambda x: (x["sharpe"], x["total_pnl"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="MNK ORB sweep (RTH, LONG-only)")
    parser.add_argument("--backtest-only", action="store_true")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-04-16T00:00:00+00:00")
    parser.add_argument("--contracts", type=int, default=1, help="Contracts for sim (default 1)")
    args = parser.parse_args()

    df = None
    if not args.backtest_only:
        try:
            df = fetch_and_save("MNK.c.0", MNK_PATH, args.start, args.end)
        except Exception as e:
            logger.error("MNK fetch failed: %s", e)

    if df is None and MNK_PATH.exists():
        logger.info("Loading cached MNK …")
        df = _load_bars(MNK_PATH, "bars_5min")

    if df is None or len(df) == 0:
        logger.error("No MNK bars. Set DATABENTO_API_KEY or place HDF at %s", MNK_PATH)
        sys.exit(1)

    df = _filter_ytd(df, args.start)
    logger.info(
        "MNK RTH 5m: %s bars  %s → %s",
        f"{len(df):,}",
        df.index[0],
        df.index[-1],
    )

    rows = _sweep(df, args.contracts)
    best = _pick_best(rows)

    print("\n" + "=" * 72)
    print("  MNK ORB sweep — top rows by Sharpe (LONG-only, PT=3x SL=1.5x ATR)")
    print("=" * 72)
    top = sorted(rows, key=lambda x: (x["sharpe"], x["total_pnl"]), reverse=True)[:12]
    for t in top:
        print(
            f"  or_end={t['or_end_time']}  mra={t['min_range_atr']:.2f}  "
            f"min_bars={t['min_or_bars']}  N={t['num_trades']:>3}  "
            f"PnL=${t['total_pnl']:>8,.0f}  Sharpe={t['sharpe']:>6.2f}  "
            f"MaxDD=${t['max_drawdown']:>8,.0f}  pass={t['combine_pass_likely']}"
        )

    if best:
        print("\n  Suggested starting config (highest Sharpe with enough trades):")
        print(
            f"    or_end_time: \"{best['or_end_time']}\"\n"
            f"    min_or_bars: {best['min_or_bars']}\n"
            f"    min_range_atr: {best['min_range_atr']}"
        )

    out = {
        "instrument": "MNK",
        "period_start": args.start,
        "period_end": str(args.end),
        "defaults_reference": {
            "pt_mult": PT_MULT,
            "sl_mult": SL_MULT,
            "entry_cutoff": ENTRY_CUTOFF,
            "baseline_or_end": OR_END_TIME,
            "min_or_bars_baseline": MIN_OR_BARS,
        },
        "sweep": rows,
        "best": best,
    }
    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Wrote %s", RESULTS_JSON)


if __name__ == "__main__":
    main()
