"""Stop x size grid for the two-leg book, on the REPAIRED tape.

This is a rewrite of `research_disaster_stop.py`, which produced the frozen
spec's 73.4% and then vanished from the repo (it was never in git and survived
only as bytecode). The numbers it produced are not reproducible, and it was in
any case run on a tape missing every non-DST Sunday reopen plus the worst
weekend in the sample.

WHAT THE ORIGINAL GOT RIGHT AND IS PRESERVED HERE
-------------------------------------------------
* Punitive gap fill: a stopped trade fills at `min(stop, bar OPEN)`, so a
  Sunday or Monday gap through the stop is NOT assumed tradeable.
* A stop changes the MAE, not just the exit. The path is truncated at the stop
  bar, so the account is never charged for an excursion a stopped position was
  never in. (The original's first pass had this bug.)
* The EOD-trailing floor LOCKS at the starting balance:
  `floor = max(floor, min(peak - MLL, 0))`, per account_rules.yaml. Trailing it
  forever understates P(pass), most at large sizes.
* Intra-trade bust: equity + MAE is checked against the floor before the trade
  books, because the drawdown limit is breached live, not at settlement.

Legs are built once per stop level at ONE micro and scaled, since the stop is
specified per micro and both P/L and excursion are linear in size.
"""
import argparse
import json

import numpy as np
import pandas as pd

ET = "America/New_York"
PV = 2.0                # $/point per MNQ micro
COST = 2.06             # measured round turn at 1 lot (mbo_fill_model)
TARGET = 6000.0
MLL = 3000.0
FOMC_PER_YEAR = 8.0


def build_leg(bars, entries, stop_usd):
    """One micro. Returns (pnl, mae) arrays; mae is truncated at the stop bar."""
    pnl, mae = [], []
    for ent, ex, tol in entries:
        i = bars.index.searchsorted(ent, side="left")
        if i >= len(bars) or (bars.index[i] - ent).total_seconds() / 60 > tol:
            continue
        j = bars.index.searchsorted(ex, side="right") - 1
        if j <= i or (ex - bars.index[j]).total_seconds() / 60 > 90:
            continue
        seg = bars.iloc[i:j + 1]
        ep = float(seg["open"].iloc[0])
        lo = seg["low"].to_numpy()
        if stop_usd is None:
            xp, end = float(seg["close"].iloc[-1]), len(seg)
        else:
            stop = ep - stop_usd / PV
            hit = np.where(lo <= stop)[0]
            if len(hit):
                k = int(hit[0])
                xp, end = min(stop, float(seg["open"].iloc[k])), k + 1
            else:
                xp, end = float(seg["close"].iloc[-1]), len(seg)
        pnl.append((xp - ep) * PV - COST)
        mae.append((float(lo[:end].min()) - ep) * PV)      # <= 0
    return np.asarray(pnl), np.asarray(mae)


def weekend_entries(bars):
    out = []
    for s in pd.date_range(bars.index.min().date(), bars.index.max().date(), freq="W-SUN"):
        out.append((pd.Timestamp.combine(s.date(), pd.Timestamp("18:00").time()).tz_localize(ET),
                    pd.Timestamp.combine(s.date() + pd.Timedelta(days=1),
                                         pd.Timestamp("15:59").time()).tz_localize(ET),
                    2))                                     # strict reopen
    return out


def fomc_entries(bars, cal_path):
    days = {x.date() for x in bars.index}
    out = []
    for a in [pd.Timestamp(x).date() for x in pd.read_csv(cal_path)["announcement_date"]]:
        p = a - pd.Timedelta(days=1)
        while p not in days and (a - p).days < 5:
            p -= pd.Timedelta(days=1)
        if p not in days:
            continue
        out.append((pd.Timestamp.combine(p, pd.Timestamp("18:00").time()).tz_localize(ET),
                    pd.Timestamp.combine(a, pd.Timestamp("13:55").time()).tz_localize(ET),
                    90))
    return out


def _mc(wp, wm, fp, fm, wk_size, fo_size, weeks, n, seed, decay):
    """Bootstrap weekly paths. Returns (p_pass, p_bust, median_weeks_or_-1).

    Weekend and FOMC are handled explicitly rather than through a list of
    heterogeneous tuples, so the whole loop compiles in nopython mode.
    """
    np.random.seed(seed)
    p_f = FOMC_PER_YEAR / 52.0
    wmean = wp.mean()
    nw = len(wp)
    nf = len(fp)
    passed = 0
    busted = 0
    wks = np.empty(n, dtype=np.int64)
    nwk = 0
    for _ in range(n):
        eq = 0.0
        peak = 0.0
        floor = -MLL
        dead = False
        done = False
        for k in range(weeks):
            # ---- weekend leg
            if wk_size > 0:
                i = np.random.randint(0, nw)
                p = wp[i]
                if decay != 1.0:                      # shrink mean, keep spread
                    p = wmean * decay + (p - wmean)
                p = p * wk_size
                if eq + wm[i] * wk_size <= floor:     # intra-trade breach
                    busted += 1
                    dead = True
                    break
                eq += p
                if eq <= floor:
                    busted += 1
                    dead = True
                    break
            # ---- fomc leg, ~8 a year
            if fo_size > 0 and np.random.random() < p_f:
                j = np.random.randint(0, nf)
                if eq + fm[j] * fo_size <= floor:
                    busted += 1
                    dead = True
                    break
                eq += fp[j] * fo_size
                if eq <= floor:
                    busted += 1
                    dead = True
                    break
            peak = max(peak, eq)
            floor = max(floor, min(peak - MLL, 0.0))
            if eq >= TARGET:
                passed += 1
                wks[nwk] = k + 1
                nwk += 1
                done = True
                break
        if dead or done:
            continue
    med = np.median(wks[:nwk]) if nwk > 0 else -1.0
    return passed / n, busted / n, med


try:
    from numba import njit
    _mc = njit(cache=True)(_mc)
except Exception:       # pragma: no cover
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", default="/Users/jg/auction/data/mnq_1m_all.parquet")
    ap.add_argument("--fomc", default="/Users/jg/auction/data/fomc_announcements.csv")
    ap.add_argument("--weeks", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--paths", type=int, default=20000)
    ap.add_argument("--out", default="/Users/jg/auction/runs/disaster_stop_repaired.json")
    a = ap.parse_args()

    bars = pd.read_parquet(a.bars).tz_convert(ET).sort_index()
    we, fe = weekend_entries(bars), fomc_entries(bars, a.fomc)
    stops = [None, 300.0, 400.0, 500.0, 600.0, 800.0, 1000.0]

    legs = {}
    print("leg summary at 1 micro (mean $/event, worst, worst MAE):")
    for s in stops:
        wp, wm = build_leg(bars, we, s)
        fp, fm = build_leg(bars, fe, s)
        legs[s] = (wp, wm, fp, fm)
        print(f"  stop={str(s):>6}  weekend n={len(wp)} mean=${wp.mean():7.2f} "
              f"worst=${wp.min():8.2f} maeworst=${wm.min():8.2f} | "
              f"fomc n={len(fp)} mean=${fp.mean():7.2f} worst=${fp.min():8.2f}")

    rows = []
    for weeks in a.weeks:
        for s in stops:
            wp, wm, fp, fm = legs[s]
            for wk in (1, 2, 3, 4):
                for fo in (0, 1, 2, 3, 4):
                    p, b, med = _mc(wp, wm, fp, fm, wk, fo, weeks, a.paths, 7, 1.0)
                    p50, _, _ = _mc(wp, wm, fp, fm, wk, fo, weeks, a.paths, 7, 0.5)
                    rows.append(dict(weeks=weeks, stop=s, weekend=wk, fomc=fo,
                                     p_pass=round(p, 4), p_bust=round(b, 4),
                                     median_weeks=(None if med < 0 else med), p_pass_decay50=round(p50, 4),
                                     worst_weekend=round(float(wp.min() * wk), 2),
                                     per_week=round(float(wp.mean() * wk
                                                          + fp.mean() * fo * FOMC_PER_YEAR / 52.0), 2)))
        print(f"  {weeks}-week grid done", flush=True)

    json.dump(rows, open(a.out, "w"), indent=2)
    df = pd.DataFrame(rows)
    for weeks in a.weeks:
        d = df[df.weeks == weeks].sort_values("p_pass", ascending=False)
        print(f"\n=== best 8 cells at {weeks} weeks (repaired tape) ===")
        print(d.head(8).to_string(index=False))
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
