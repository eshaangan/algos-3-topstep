"""TRACK B: which prop firm's rules let the ALREADY-VALIDATED book pass?

No new alpha. The premise of Track B is that fomc_drift_v1 (t=+3.53) and
weekend_hold_v1 (t=+4.47) are statistically real and are killed only by LucidFlex's
mandatory 16:45 ET flatten. If some firm imposes no such flatten, the existing book
becomes tradeable at its full specification and no new edge is needed.

For each surveyed firm this:
  1. machine-checks which of the book's windows are LEGAL under that firm's rules,
  2. picks the best LEGAL fomc entry hour for that firm (never assumes one),
  3. runs the path-aware evaluation MC with that firm's real target, MLL and
     consistency rule,
  4. reports single-event risk -- the worst historical trade against the MLL --
     because a 92% P(pass) is worthless if one event can end the account.

Trade construction is IMPORTED from research_portfolio_mc rather than reimplemented;
this project has already been burned once by a second MC with different rules.

Usage:
    python3 rule_based_v1/validation/research_firm_choice_mc.py --out runs/firm_choice.json
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
from tzguard import check_window_legal, load_firm_rules
from research_portfolio_mc import load_px, weekend_trades, fomc_trades, sized

ET = "America/New_York"
FIRMS = ["lucid_flex_100k", "myfundedfutures_pro_100k",
         "phidias_swing_100k", "elite_trader_funding_diamond_100k"]
FOMC_ENTRY_HOURS = [14.0, 16.0, 18.0, 20.0]


def mc(wk: pd.DataFrame, fo: pd.DataFrame, target: float, mll: float,
       consistency: float | None, min_days: int, daily_loss_limit: float | None,
       fomc_per_year: float = 8.0, n_paths: int = 20000, max_weeks: int = 104,
       seed: int = 17, weekend_decay: float = 1.0) -> dict:
    """Path-aware evaluation MC with the consistency rule modelled explicitly.

    Consistency is checked at the MOMENT of passing: the largest single winning DAY
    must be <= `consistency` x total profit. The weekend trade closes Monday and the
    FOMC trade closes Wednesday, so they are distinct days and are tracked as such.
    An account that reaches the target but fails consistency does not pass -- it keeps
    trading, which for a lumpy book is a real constraint, not a formality.
    """
    rng = np.random.default_rng(seed)
    wp = wk["pnl"].to_numpy()
    if weekend_decay != 1.0:
        wp = (wp - wp.mean()) + wp.mean() * weekend_decay
    wm = np.minimum(wk["mae"].to_numpy(), 0.0)
    fp, fm = fo["pnl"].to_numpy(), np.minimum(fo["mae"].to_numpy(), 0.0)
    p_fomc = fomc_per_year / 52.0

    eq = np.zeros(n_paths)
    floor = np.full(n_paths, -mll)
    peak = np.zeros(n_paths)
    best_day = np.zeros(n_paths)
    ndays = np.zeros(n_paths, dtype=int)
    done = np.zeros(n_paths, dtype=bool)
    passed = np.zeros(n_paths, dtype=bool)
    weeks = np.full(n_paths, max_weeks, dtype=int)

    for w in range(max_weeks):
        if done.all():
            break
        i = rng.integers(0, len(wp), n_paths)
        has_f = (rng.random(n_paths) < p_fomc) & (len(fp) > 1)
        j = rng.integers(0, len(fp), n_paths)
        wk_pnl = wp[i] if len(wp) > 1 else np.zeros(n_paths)
        wk_mae = wm[i] if len(wp) > 1 else np.zeros(n_paths)
        fo_pnl = np.where(has_f, fp[j], 0.0)
        fo_mae = np.where(has_f, fm[j], 0.0)

        # Separate DAYS: the account can bust on either day's excursion, and the
        # trailing floor only ratchets on closed end-of-day equity.
        live = ~done
        done |= live & ((eq + wk_mae) < floor)
        live = ~done
        eq = np.where(live, eq + wk_pnl, eq)
        best_day = np.where(live, np.maximum(best_day, np.maximum(wk_pnl, 0.0)), best_day)
        ndays += live & (wk_pnl != 0)
        if daily_loss_limit is not None:
            done |= live & (wk_pnl < -daily_loss_limit)
        live = ~done
        done |= live & ((eq + fo_mae) < floor)
        live = ~done
        eq = np.where(live, eq + fo_pnl, eq)
        best_day = np.where(live, np.maximum(best_day, np.maximum(fo_pnl, 0.0)), best_day)
        ndays += live & has_f
        if daily_loss_limit is not None:
            done |= live & (fo_pnl < -daily_loss_limit)

        live = ~done
        ok_cons = (best_day <= consistency * np.maximum(eq, 1e-9)) if consistency else np.ones(n_paths, bool)
        hit = live & (eq >= target) & ok_cons & (ndays >= min_days)
        passed |= hit
        weeks = np.where(hit & (weeks == max_weeks), w + 1, weeks)
        done |= hit
        live = ~done
        done |= live & (eq < floor)
        live = ~done
        peak = np.where(live, np.maximum(peak, eq), peak)
        # LOCK: LucidFlex's EOD-trailing drawdown STOPS trailing once the floor
        # reaches the starting balance (account_rules.yaml::drawdown.
        # locks_at_starting_balance). Equity is relative here, so the starting
        # balance is 0 and the floor may never exceed it. Trailing past 0 -- which
        # every MC in this repo did until 2026-09-05 -- silently understates
        # P(pass), and it understates it MOST at large sizes, because those are
        # the paths that build a big peak.
        floor = np.where(live, np.maximum(floor, np.minimum(peak - mll, 0.0)), floor)

    return {"p_pass": float(passed.mean()),
            "median_weeks": float(np.median(weeks[passed])) if passed.any() else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Track B: firm choice under the existing book")
    ap.add_argument("--bars", type=Path, default=Path("data/processed/mnq_1m_all.parquet"))
    ap.add_argument("--ts-col", default="ts")
    ap.add_argument("--tz", default="America/Chicago")
    ap.add_argument("--point-value", type=float, default=2.0)
    ap.add_argument("--tick", type=float, default=0.25)
    ap.add_argument("--comm", type=float, default=0.62)
    ap.add_argument("--calendar", type=Path, default=Path("data/processed/fomc_announcements.csv"))
    ap.add_argument("--max-weeks", type=int, default=16, help="the deliverable's deadline")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    px = load_px(args.bars, args.ts_col, args.tz)
    wk_raw = weekend_trades(px)
    fo_raw = {h: fomc_trades(px, args.calendar, h) for h in FOMC_ENTRY_HOURS}

    print(f"\nweekend  n={len(wk_raw):<4} mean={wk_raw['points'].mean():+7.2f} pts  "
          f"t={stats.ttest_1samp(wk_raw['points'], 0).statistic:+5.2f}")
    for h, d in fo_raw.items():
        print(f"fomc@{int(h):02d} n={len(d):<4} mean={d['points'].mean():+7.2f} pts  "
              f"t={stats.ttest_1samp(d['points'], 0).statistic:+5.2f}")

    results = []
    for firm in FIRMS:
        r = load_firm_rules(firm)
        target, mll = float(r["profit_target"]), float(r["max_loss_limit"])
        cons = r["consistency"]
        dll = r.get("daily_loss_limit")
        print(f"\n{'=' * 104}\n{r['label']}")
        print(f"  target=${target:,.0f}  MLL=${mll:,.0f}  consistency="
              f"{f'{cons:.0%}' if cons else 'none'}  daily_loss="
              f"{f'${dll:,.0f}' if dll else 'none'}  flatten={r['flatten_et'] or 'NONE'}  "
              f"min_days={r['min_trading_days']}")

        legal_hours = [h for h in FOMC_ENTRY_HOURS
                       if check_window_legal(f"{int(h):02d}:00", "13:55", 1, r)[0]]
        wk_ok = check_window_legal("18:00", "15:59", 1, r)[0]
        print(f"  legal fomc entry hours: {[f'{int(h)}:00' for h in legal_hours] or 'NONE'}"
              f"   weekend_hold legal: {wk_ok}")
        if not legal_hours:
            print("  -> no legal fomc window at this firm")

        # single-event risk FIRST -- a size whose worst historical trade breaches the
        # MLL is disqualified regardless of what the MC says.
        print(f"\n  single-event risk (worst historical trade vs ${mll:,.0f} MLL):")
        for nc in (1, 2, 3, 4):
            wmae = sized(wk_raw, nc, args.point_value, args.tick, args.comm)["mae"].min()
            line = f"    {nc} micro(s):  weekend worst ${wmae:>8,.0f} ({abs(wmae)/mll:>5.0%})"
            if legal_hours:
                fmae = sized(fo_raw[legal_hours[0]], nc, args.point_value,
                             args.tick, args.comm)["mae"].min()
                line += f"   fomc@{int(legal_hours[0])}:00 worst ${fmae:>8,.0f} ({abs(fmae)/mll:>5.0%})"
            print(line + ("   <-- BREACH" if abs(wmae) > mll else ""))

        print(f"\n  P(pass) within {args.max_weeks} weeks / median weeks   "
              f"[fomc entry = best legal hour]")
        print(f"  {'wk':>3}{'fomc':>6}{'entry':>7}{'$/wk':>9}{'P(pass)':>10}{'median':>8}"
              f"{'  decay x0.50':>14}{'x0.00':>9}")
        best = None
        for h in legal_hours or [None]:
            for wn in (0, 1, 2):
                for fn in (0, 2, 3, 4):
                    if wn == 0 and fn == 0:
                        continue
                    if not wk_ok and wn:
                        continue
                    wk = (sized(wk_raw, wn, args.point_value, args.tick, args.comm)
                          if wn else pd.DataFrame({"pnl": [0.0], "mae": [0.0]}))
                    fo = (sized(fo_raw[h], fn, args.point_value, args.tick, args.comm)
                          if fn and h is not None else pd.DataFrame({"pnl": [0.0], "mae": [0.0]}))
                    # disqualify sizes with a single-event account kill
                    if (wn and abs(wk["mae"].min()) > mll) or (fn and abs(fo["mae"].min()) > mll):
                        continue
                    kw = dict(target=target, mll=mll, consistency=cons,
                              min_days=int(r["min_trading_days"]), daily_loss_limit=dll,
                              fomc_per_year=8.0 if fn else 0.0, max_weeks=args.max_weeks)
                    m = mc(wk, fo, **kw)
                    d50 = mc(wk, fo, **kw, weekend_decay=0.50)["p_pass"] if wn else m["p_pass"]
                    d00 = mc(wk, fo, **kw, weekend_decay=0.00)["p_pass"] if wn else m["p_pass"]
                    per_wk = ((wk["pnl"].mean() if wn else 0.0)
                              + (fo["pnl"].mean() * 8 / 52 if fn else 0.0))
                    rec = {"firm": firm, "weekend": wn, "fomc": fn,
                           "fomc_entry_hour": h, "per_week": per_wk,
                           "p_pass": m["p_pass"], "median_weeks": m["median_weeks"],
                           "p_pass_decay50": d50, "p_pass_decay00": d00}
                    results.append(rec)
                    flag = "  <== clears 85%" if m["p_pass"] >= 0.85 else ""
                    print(f"  {wn:>3}{fn:>6}{(f'{int(h)}:00' if h else '-'):>7}{per_wk:>9.0f}"
                          f"{m['p_pass']:>10.1%}{m['median_weeks']:>8.0f}"
                          f"{d50:>14.1%}{d00:>9.1%}{flag}")
                    if m["p_pass"] >= 0.85 and (best is None or m["median_weeks"] < best["median_weeks"]):
                        best = rec
        if best:
            print(f"\n  BEST >=85% within {args.max_weeks}wk: weekend x{best['weekend']} + "
                  f"fomc x{best['fomc']} @ {int(best['fomc_entry_hour'])}:00  -> "
                  f"{best['p_pass']:.1%} / {best['median_weeks']:.0f}wk median")
        else:
            print(f"\n  nothing clears 85% within {args.max_weeks} weeks at this firm")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
