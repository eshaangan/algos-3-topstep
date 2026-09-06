"""What survives under the AUDITED LucidFlex rules? (2026-09-05)

The external audit returned NO-GO and, more importantly, corrected the account
rules this project had been assuming. The corrections are not statistical -- they
change what is physically tradeable:

  assumed (ledger)                    audited
  ----------------------------------  -----------------------------------------
  $3,000 trailing buffer              $2,000 MLL
  no consistency rule                 50% evaluation consistency
  "~23h holds allowed"                mandatory 16:45 ET flatten every day

The flatten rule is decisive. fomc_drift_v1 holds from the prior session (14:00 or
16:00 ET) through 13:55 the next day, so it crosses 16:45 ET on the entry evening
and would be force-closed. The load-bearing leg is therefore untradeable AS
SPECIFIED, regardless of its statistics.

weekend_hold_v1 survives the flatten rule on its face: Sunday 18:00 -> Monday 15:59
crosses no 16:45 boundary. So the only live question left is whether a weekend-only
book clears anything under a $2,000 MLL plus a 50% consistency rule.

Consistency is modelled the standard way: at the moment equity reaches the target,
the single largest winning day must be no more than 50% of total profit. If it is
not, the account cannot pass yet and must keep trading -- which for a book whose
edge is a handful of big weekends is a real constraint, not a formality.

Usage:
    python3 rule_based_v1/validation/research_corrected_rules_mc.py --out runs/corrected_rules.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tzguard import assert_et_index, check_window_legal, load_account_rules

ET = "America/New_York"


def load_px(path: Path, ts_col: str, tz: str | None) -> pd.Series:
    raw = pd.read_parquet(path)
    idx = pd.to_datetime(raw[ts_col])
    if idx.dt.tz is None:
        idx = idx.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    s = pd.Series(raw["close"].to_numpy(float), index=idx.dt.tz_convert(ET))
    s = s[~s.index.isna()].sort_index()
    return s[~s.index.duplicated(keep="last")]


def weekend_trades(px: pd.Series, pv: float, contracts: int,
                   tick: float = 0.25, comm: float = 0.62) -> pd.DataFrame:
    cost = 2 * contracts * (comm + tick * pv)
    days = pd.DatetimeIndex(np.unique(px.index.normalize()))
    rows = []
    for sun in days[days.dayofweek == 6]:
        mon = sun + pd.Timedelta(days=1)
        a = px[sun + pd.Timedelta(hours=18):sun + pd.Timedelta(hours=18, minutes=10)]
        b = px[mon + pd.Timedelta(hours=15, minutes=29):mon + pd.Timedelta(hours=15, minutes=59)]
        if a.empty or b.empty:
            continue
        entry = float(a.iloc[0])
        path = px[a.index[0]:b.index[-1]]
        rows.append({"pnl": (float(b.iloc[-1]) - entry) * pv * contracts - cost,
                     "mae": (float(path.min()) - entry) * pv * contracts})
    return pd.DataFrame(rows)


def mc(book: pd.DataFrame, target: float, mll: float, consistency: float | None,
       n_paths: int = 20000, max_weeks: int = 156, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    pnl, mae = book["pnl"].to_numpy(), np.minimum(book["mae"].to_numpy(), 0.0)
    draws = rng.integers(0, len(pnl), size=(n_paths, max_weeks))
    eq = np.zeros(n_paths)
    floor = np.full(n_paths, -mll)
    peak = np.zeros(n_paths)
    best_day = np.zeros(n_paths)        # largest single winning week
    done = np.zeros(n_paths, dtype=bool)
    passed = np.zeros(n_paths, dtype=bool)
    weeks = np.full(n_paths, max_weeks, dtype=int)
    for w in range(max_weeks):
        if done.all():
            break
        live = ~done
        i = draws[:, w]
        done |= live & ((eq + mae[i]) < floor)          # intra-trade breach
        live = ~done
        eq = np.where(live, eq + pnl[i], eq)
        best_day = np.where(live, np.maximum(best_day, np.maximum(pnl[i], 0.0)), best_day)
        ok_consistency = (best_day <= consistency * np.maximum(eq, 1e-9)) if consistency else True
        hit = live & (eq >= target) & ok_consistency
        passed |= hit
        weeks = np.where(hit & (weeks == max_weeks), w + 1, weeks)
        done |= hit
        live = ~done
        done |= live & (eq < floor)
        live = ~done
        peak = np.where(live, np.maximum(peak, eq), peak)
        floor = np.where(live, np.maximum(floor, peak - mll), floor)
    return {"p_pass": float(passed.mean()),
            "median_weeks": float(np.median(weeks[passed])) if passed.any() else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description="MC under audited LucidFlex rules")
    ap.add_argument("--bars", type=Path, default=Path("data/processed/mnq_1m_all.parquet"))
    ap.add_argument("--ts-col", default="ts")
    ap.add_argument("--tz", default="America/Chicago")
    ap.add_argument("--point-value", type=float, default=2.0)
    ap.add_argument("--account", default=None, help="override active_account in the config")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rules = load_account_rules(account=args.account)
    target, mll_real = rules["profit_target"], rules["max_loss_limit"]
    print(f"account={rules['account']}  balance=${rules['balance']:,}  "
          f"target=${target:,.0f}  MLL=${mll_real:,.0f}  "
          f"consistency={rules['consistency']:.0%}  flatten={rules['flatten_et']} ET\n")
    for name, (ein, eout, span) in {
            "fomc_drift_v1  (prior 16:00 -> 13:55)": ("16:00", "13:55", 1),
            "weekend_hold_v1 (Sun 18:00 -> Mon 15:59)": ("18:00", "15:59", 1),
    }.items():
        ok, why = check_window_legal(ein, eout, span, rules)
        print(f"  {'LEGAL  ' if ok else 'ILLEGAL'}  {name}: {why}")
    print()

    px = load_px(args.bars, args.ts_col, args.tz)
    print("weekend_hold_v1 only -- fomc_drift_v1 is excluded because a mandatory")
    print("16:45 ET flatten force-closes it before the tested window completes.\n")
    print(f"{'size':>5}{'MLL':>8}{'consist':>9}{'$/wk':>9}{'P(pass)':>10}{'median':>9}   scenario")
    rows = []
    for nc in (1, 2):
        book = weekend_trades(px, args.point_value, nc)
        for tgt, mll, cons, tag in (
                (3000.0, 3000.0, None, "what I modelled (WRONG: 100k MLL + 50k target, no consistency)"),
                (target, mll_real, None, f"real target ${target:,.0f} + real MLL, no consistency"),
                (target, mll_real, rules["consistency"], "REAL RULES (target + MLL + consistency)")):
            r = mc(book, tgt, mll, cons)
            rows.append({"contracts": nc, "target": tgt, "mll": mll, "consistency": cons,
                         "per_week": float(book["pnl"].mean()), **r, "tag": tag})
            print(f"{nc:>5}{mll:>8,.0f}{(str(int(cons * 100)) + '%') if cons else '   none':>9}"
                  f"{book['pnl'].mean():>9.2f}{r['p_pass']:>10.1%}{r['median_weeks']:>9.0f}   {tag}")
        # worst single weekend against the corrected cushion
        print(f"      worst single weekend at {nc} micro(s): ${book['mae'].min():,.0f} "
              f"= {abs(book['mae'].min()) / mll_real:.0%} of the ${mll_real:,.0f} MLL\n")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
