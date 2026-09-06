"""How much edge would 85% in 16 weeks actually require? (the NO-GO, quantified)

Every route tried in this session -- a different firm, the recovered legal FOMC window,
a disaster stop that genuinely caps the tail -- lands in the same place: ~92-94% P(pass)
but 30-45 weeks, never 85% inside 16. Rather than report that as a list of failures,
this measures the gap directly.

Two knobs, applied to the real (stopped) weekend + FOMC book:

  SIZE     scales mean, dispersion and tail together. More size = faster but bustier.
           This is the knob the project actually has.
  QUALITY  scales the mean while holding dispersion and the tail fixed -- i.e. a better
           edge, not a bigger bet. This is the knob the project would need to FIND.

Reporting the quality multiplier required for 85%/16wk converts "no" into "how far",
which is the only version of a negative result that is actionable.

Usage:
    python3 rule_based_v1/validation/research_required_edge.py --out runs/required_edge.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tzguard import load_firm_rules
from research_firm_choice_mc import mc
from research_disaster_stop import load_ohlc, weekend_rows, fomc_rows, sized, PV


def scale_quality(df: pd.DataFrame, k: float) -> pd.DataFrame:
    """Multiply the MEAN by k, leaving dispersion and the left tail untouched."""
    out = df.copy()
    m = out["pnl"].mean()
    out["pnl"] = (out["pnl"] - m) + m * k
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="required-edge frontier")
    ap.add_argument("--bars", type=Path, default=Path("data/processed/mnq_1m_all.parquet"))
    ap.add_argument("--ts-col", default="ts")
    ap.add_argument("--tz", default="America/Chicago")
    ap.add_argument("--calendar", type=Path,
                    default=Path("data/processed/fomc_announcements.csv"))
    ap.add_argument("--firm", default="lucid_flex_100k")
    ap.add_argument("--stop", type=float, default=600.0, help="USD per micro")
    ap.add_argument("--weeks", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    r = load_firm_rules(args.firm)
    target, mll, cons = float(r["profit_target"]), float(r["max_loss_limit"]), r["consistency"]
    budget = 0.60 * mll
    df = load_ohlc(args.bars, args.ts_col, args.tz)
    sp = args.stop / PV
    wk_raw = weekend_rows(df, sp)
    fo_raw = fomc_rows(df, args.calendar, sp, 18.0)
    print(f"\n{r['label']}  target=${target:,.0f}  MLL=${mll:,.0f}  "
          f"stop=${args.stop:,.0f}/micro  horizon={args.weeks}wk  tail budget=${budget:,.0f}")

    base = dict(target=target, mll=mll, consistency=cons,
                min_days=int(r["min_trading_days"]), daily_loss_limit=None,
                fomc_per_year=8.0, max_weeks=args.weeks)

    print(f"\nKNOB 1 -- SIZE (what the project has). Mean, dispersion and tail all scale.")
    print(f"  {'wk':>3}{'fomc':>6}{'$/wk':>9}{'worst wk$':>11}{'P(pass)':>10}{'median':>8}"
          f"{'  tail ok':>9}")
    rows = []
    for wn in (1, 2, 3, 4, 5, 6, 8):
        fn = min(2 * wn, 6)
        wk, fo = sized(wk_raw, wn), sized(fo_raw, fn)
        w_worst = float(wk["pnl"].min())
        m = mc(wk, fo, **base)
        ok = abs(w_worst) <= budget and abs(float(fo["pnl"].min())) <= budget
        rows.append({"knob": "size", "weekend": wn, "fomc": fn,
                     "per_week": float(wk["pnl"].mean() + fo["pnl"].mean() * 8 / 52),
                     "worst_weekend": w_worst, "p_pass": m["p_pass"],
                     "median_weeks": m["median_weeks"], "tail_ok": ok})
        print(f"  {wn:>3}{fn:>6}{rows[-1]['per_week']:>9.0f}{w_worst:>11,.0f}"
              f"{m['p_pass']:>10.1%}{m['median_weeks']:>8.0f}{'  yes' if ok else '   NO':>9}")

    print(f"\nKNOB 2 -- QUALITY (what the project would need to FIND). Mean scaled;")
    print(f"dispersion and the left tail held FIXED, at the largest tail-legal size.")
    best_size = max((r_ for r_ in rows if r_["tail_ok"]), key=lambda z: z["weekend"])
    wn, fn = best_size["weekend"], best_size["fomc"]
    print(f"  [size fixed at weekend x{wn} + fomc x{fn}, worst weekend "
          f"${best_size['worst_weekend']:,.0f} = {abs(best_size['worst_weekend'])/mll:.0%} of MLL]")
    print(f"  {'quality':>9}{'$/wk':>9}{'P(pass)':>10}{'median':>8}")
    need = None
    for k in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0):
        wk = scale_quality(sized(wk_raw, wn), k)
        fo = scale_quality(sized(fo_raw, fn), k)
        m = mc(wk, fo, **base)
        pw = float(wk["pnl"].mean() + fo["pnl"].mean() * 8 / 52)
        rows.append({"knob": "quality", "k": k, "per_week": pw, **m})
        print(f"  {f'x{k:.1f}':>9}{pw:>9.0f}{m['p_pass']:>10.1%}{m['median_weeks']:>8.0f}"
              + ("   <== reaches 85%" if m["p_pass"] >= 0.85 and need is None else ""))
        if m["p_pass"] >= 0.85 and need is None:
            need = (k, pw)

    print()
    if need:
        k, pw = need
        cur = best_size["per_week"]
        print(f"VERDICT: 85% in {args.weeks} weeks requires roughly a {k:.1f}x better edge "
              f"at the same risk\n         -- about ${pw:,.0f}/week versus the "
              f"${cur:,.0f}/week the book actually earns\n         at the largest size "
              f"whose worst historical trade stays inside 60% of the MLL.")
    else:
        print(f"VERDICT: even an 8x edge at fixed risk does not reach 85% in "
              f"{args.weeks} weeks.\n         The $6,000-target / $3,000-trailing-MLL "
              f"geometry, not the edge, is binding.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
