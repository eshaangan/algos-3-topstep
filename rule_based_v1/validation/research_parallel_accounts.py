"""Campaign of parallel evaluations on ONE shared return stream (prereg: vps_parallel_accounts.yaml).

Every single-account route measured this session tops out below 85% within 16 weeks. But
the deliverable is "pass AN evaluation", and the session's frontier is that small tiers
are fast-and-fragile (25k: 8wk median, ~64% ceiling) while the 100k is safe-and-slow. A
campaign of several accounts turns that into an "at least one succeeds" question.

THE WHOLE DIFFICULTY IS CORRELATION. N accounts running the same strategy at the same size
from the same date are the SAME ACCOUNT N times -- identical weekends, they bust together,
and P(at least one) is exactly P(one). Independent per-account draws would fabricate
diversification that does not exist. So every account here trades ONE SHARED weekly
stream, and benefit can only come from:

    STAGGER  -- different phase in the same stream; an early disaster kills a fresh
                account but only dents a seasoned one holding a cushion
    SIZE MIX -- aggressive and conservative accounts fail in different states

A mandatory identity check asserts that stagger-0, same-size parallelism gives EXACTLY the
single-account probability. If that check fails the plumbing is broken and nothing else in
the run means anything.

Usage:
    python3 rule_based_v1/validation/research_parallel_accounts.py --out runs/parallel.json
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
from research_disaster_stop import load_ohlc, weekend_rows, fomc_rows, sized, PV


def run_campaign(wk_pnl: np.ndarray, wk_mae: np.ndarray,
                 fo_pnl: np.ndarray, fo_mae: np.ndarray,
                 specs: list[dict], horizon: int, n_paths: int,
                 fomc_per_year: float = 8.0, seed: int = 5) -> dict:
    """specs: [{target, mll, consistency, min_days, wk_scale, fo_scale, start}, ...]

    ONE stream of weekly draws, shared by every account. Account i is idle before its
    start week and then trades the same weeks everyone else does.
    """
    rng = np.random.default_rng(seed)
    # the shared stream: one draw per campaign week, used by ALL accounts
    wi = rng.integers(0, len(wk_pnl), size=(n_paths, horizon))
    has_f = rng.random((n_paths, horizon)) < (fomc_per_year / 52.0)
    fi = rng.integers(0, len(fo_pnl), size=(n_paths, horizon))

    any_pass = np.zeros(n_paths, dtype=bool)
    first_week = np.full(n_paths, horizon + 1, dtype=int)
    per_account = []

    for sp in specs:
        target, mll, cons = sp["target"], sp["mll"], sp["consistency"]
        eq = np.zeros(n_paths)
        floor = np.full(n_paths, -mll)
        peak = np.zeros(n_paths)
        best_day = np.zeros(n_paths)
        ndays = np.zeros(n_paths, dtype=int)
        done = np.zeros(n_paths, dtype=bool)
        passed = np.zeros(n_paths, dtype=bool)
        weeks = np.full(n_paths, horizon + 1, dtype=int)

        for w in range(sp["start"], horizon):
            if done.all():
                break
            wp = wk_pnl[wi[:, w]] * sp["wk_scale"]
            wm = np.minimum(wk_mae[wi[:, w]] * sp["wk_scale"], 0.0)
            fp = np.where(has_f[:, w], fo_pnl[fi[:, w]] * sp["fo_scale"], 0.0)
            fm = np.where(has_f[:, w], np.minimum(fo_mae[fi[:, w]] * sp["fo_scale"], 0.0), 0.0)

            live = ~done
            done |= live & ((eq + wm) < floor)
            live = ~done
            eq = np.where(live, eq + wp, eq)
            best_day = np.where(live, np.maximum(best_day, np.maximum(wp, 0.0)), best_day)
            ndays += live & (wp != 0)
            live = ~done
            done |= live & ((eq + fm) < floor)
            live = ~done
            eq = np.where(live, eq + fp, eq)
            best_day = np.where(live, np.maximum(best_day, np.maximum(fp, 0.0)), best_day)
            ndays += live & has_f[:, w]

            live = ~done
            ok_c = (best_day <= cons * np.maximum(eq, 1e-9)) if cons else np.ones(n_paths, bool)
            hit = live & (eq >= target) & ok_c & (ndays >= sp["min_days"])
            passed |= hit
            weeks = np.where(hit & (weeks == horizon + 1), w + 1, weeks)
            done |= hit
            live = ~done
            done |= live & (eq < floor)
            live = ~done
            peak = np.where(live, np.maximum(peak, eq), peak)
            # floor locks at the starting balance (relative equity 0)
            floor = np.where(live, np.maximum(floor, np.minimum(peak - mll, 0.0)), floor)

        per_account.append(float(passed.mean()))
        any_pass |= passed
        first_week = np.minimum(first_week, weeks)

    got = first_week[any_pass]
    return {"p_campaign": float(any_pass.mean()),
            "median_weeks": float(np.median(got)) if got.size else float("nan"),
            "per_account": per_account, "n_accounts": len(specs)}


def main() -> None:
    ap = argparse.ArgumentParser(description="parallel-evaluation campaign")
    ap.add_argument("--bars", type=Path, default=Path("data/processed/mnq_1m_all.parquet"))
    ap.add_argument("--ts-col", default="ts")
    ap.add_argument("--tz", default="America/Chicago")
    ap.add_argument("--calendar", type=Path,
                    default=Path("data/processed/fomc_announcements.csv"))
    ap.add_argument("--stop", type=float, default=500.0)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--paths", type=int, default=20000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    df = load_ohlc(args.bars, args.ts_col, args.tz)
    wk1 = sized(weekend_rows(df, args.stop / PV), 1)
    fo1 = sized(fomc_rows(df, args.calendar, args.stop / PV, 18.0), 1)
    wk_pnl, wk_mae = wk1["pnl"].to_numpy(), wk1["mae"].to_numpy()
    fo_pnl, fo_mae = fo1["pnl"].to_numpy(), fo1["mae"].to_numpy()

    tiers = {t: load_firm_rules(f"lucid_flex_{t}") for t in ("25k", "100k")}
    budget = {t: 0.60 * r["max_loss_limit"] for t, r in tiers.items()}
    print(f"\nstop=${args.stop:,.0f}/micro   horizon={args.horizon}wk   "
          f"worst weekend/micro=${wk_pnl.min():,.0f}  worst fomc/micro=${fo_pnl.min():,.0f}")

    def spec(tier: str, wk: int, fo: int, start: int) -> dict:
        r = tiers[tier]
        return {"target": float(r["profit_target"]), "mll": float(r["max_loss_limit"]),
                "consistency": r["consistency"], "min_days": int(r["min_trading_days"]),
                "wk_scale": float(wk), "fo_scale": float(fo), "start": start,
                "tag": f"{tier} wk{wk}/fo{fo}@w{start}"}

    def tail_ok(tier: str, wk: int, fo: int) -> bool:
        return (abs(wk_pnl.min() * wk) <= budget[tier]
                and abs(fo_pnl.min() * fo) <= budget[tier])

    # ---- MANDATORY identity check -------------------------------------------------
    print(f"\n{'=' * 92}\nIDENTITY CHECK -- stagger 0, identical accounts must give "
          f"EXACTLY the single-account P\n{'=' * 92}")
    one = run_campaign(wk_pnl, wk_mae, fo_pnl, fo_mae, [spec("25k", 1, 1, 0)],
                       args.horizon, args.paths)
    three_same = run_campaign(wk_pnl, wk_mae, fo_pnl, fo_mae,
                              [spec("25k", 1, 1, 0)] * 3, args.horizon, args.paths)
    print(f"  1 account          P={one['p_campaign']:.4%}")
    print(f"  3 identical, w0    P={three_same['p_campaign']:.4%}")
    delta = abs(one["p_campaign"] - three_same["p_campaign"])
    print(f"  difference         {delta:.2e}   "
          f"{'PASS -- correlation modelled correctly' if delta < 1e-9 else 'FAIL -- plumbing broken'}")
    if delta >= 1e-9:
        print("\n  ABORT: identity check failed; every other number below would be invalid.")
        return

    print(f"\n{'=' * 92}\nCAMPAIGNS (shared stream; benefit can only come from stagger "
          f"or size mix)\n{'=' * 92}")
    print(f"  {'campaign':<52}{'accts':>6}{'P(campaign)':>13}{'median wk':>11}")
    results = []
    campaigns: dict[str, list[dict]] = {}
    for n in (2, 3, 4):
        for st in (0, 2, 4):
            if st == 0 and n > 1:
                continue
            campaigns[f"{n}x 25k wk1/fo1, stagger {st}wk"] = [
                spec("25k", 1, 1, i * st) for i in range(n)]
    campaigns["3x 25k staggered 4wk + 1x 100k wk1/fo2"] = (
        [spec("25k", 1, 1, i * 4) for i in range(3)] + [spec("100k", 1, 2, 0)])
    campaigns["2x 100k wk1/fo2, stagger 4wk"] = [spec("100k", 1, 2, i * 4) for i in range(2)]
    campaigns["1x 25k + 1x 100k (size mix, both w0)"] = [
        spec("25k", 1, 1, 0), spec("100k", 1, 2, 0)]
    for name, sps in campaigns.items():
        bad = [s["tag"] for s, (t, w, f) in zip(sps, [(s["tag"].split()[0],
               int(s["wk_scale"]), int(s["fo_scale"])) for s in sps]) if not tail_ok(t, w, f)]
        if bad:
            continue
        res = run_campaign(wk_pnl, wk_mae, fo_pnl, fo_mae, sps, args.horizon, args.paths)
        results.append({"campaign": name, **res})
        flag = "  <== CLEARS 85%" if res["p_campaign"] >= 0.85 else ""
        print(f"  {name:<52}{res['n_accounts']:>6}{res['p_campaign']:>13.1%}"
              f"{res['median_weeks']:>11.0f}{flag}")

    best = max((r for r in results if r["p_campaign"] >= 0.85),
               key=lambda z: -z["n_accounts"], default=None)
    print()
    if best:
        print(f"CLEARS THE BAR: {best['campaign']} -> {best['p_campaign']:.1%} within "
              f"{args.horizon}wk, median {best['median_weeks']:.0f}wk, "
              f"{best['n_accounts']} evaluations consumed")
    else:
        top = max(results, key=lambda z: z["p_campaign"])
        print(f"NOTHING clears 85% within {args.horizon} weeks. Best: {top['campaign']} "
              f"at {top['p_campaign']:.1%}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
