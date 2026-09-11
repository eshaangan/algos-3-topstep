"""Can you pass the evaluation by just being long, with no alpha at all?

THE REFRAME
-----------
The evaluation is not a contest against the market. It is a first-passage
problem: reach +$6,000 before the trailing drawdown catches you. Alpha is one
way to bias that walk, but drift is drift, and the equity risk premium is free
and abundant. The ledger already noticed this ("plain long ES for a week pays
+$69.29/wk, t=2.43, 75% positive years") and then never tested it.

WHAT "JUST BE LONG" ACTUALLY COSTS AT THIS FIRM
-----------------------------------------------
Lucid force-flattens at 16:45 ET every weekday and reopens at 18:00. So
buy-and-hold is NOT available. The closest legal thing is a chain of sessions:

    Sun 18:00 -> Mon 16:45,  Mon 18:00 -> Tue 16:45,  ...  Thu 18:00 -> Fri 16:45

which is five forced round trips a week. That is the real price of "following
the market" here, and it is charged per contract: 5 x $2.06 = $10.30/week/micro,
$330/micro over 32 weeks, against a $6,000 target.

METHOD
------
Contiguous historical windows are the primary estimator, not an iid bootstrap.
A first-passage problem is about the PATH, and bootstrapping destroys exactly
the autocorrelation and trending that decide whether a drawdown arrives before
a target. Every possible start date is used. An iid bootstrap is reported
alongside only to show how much it flatters the result.

Rules modelled: $6,000 target, $3,000 max loss limit, EOD-trailing floor that
LOCKS at the starting balance, intra-session breach checked against the low.
"""
import argparse
import json

import numpy as np
import pandas as pd

ET = "America/New_York"
TARGET = 6000.0
MLL = 3000.0


def build_sessions(bars, point_value, cost):
    """One row per legal 18:00 -> 16:45 holding session, at ONE contract."""
    days = sorted({x.date() for x in bars.index})
    rows = []
    for d in days:
        ent = pd.Timestamp.combine(d, pd.Timestamp("18:00").time()).tz_localize(ET)
        i = bars.index.searchsorted(ent, side="left")
        if i >= len(bars) or (bars.index[i] - ent).total_seconds() / 60 > 15:
            continue
        # exit at the flatten on the NEXT calendar day that trades
        nxt = None
        for add in (1, 2, 3):
            cand = d + pd.Timedelta(days=add)
            ex = pd.Timestamp.combine(cand, pd.Timestamp("16:45").time()).tz_localize(ET)
            j = bars.index.searchsorted(ex, side="right") - 1
            if j > i and (ex - bars.index[j]).total_seconds() / 60 <= 30:
                nxt, jj = ex, j
                break
        if nxt is None:
            continue
        seg = bars.iloc[i:jj + 1]
        ep = float(seg["open"].iloc[0])
        xp = float(seg["close"].iloc[-1])
        rows.append({
            "entry": seg.index[0],
            "pnl": (xp - ep) * point_value - cost,
            "mae": (float(seg["low"].min()) - ep) * point_value,   # <= 0
        })
    return pd.DataFrame(rows)


def walk(pnl, mae, size, weeks, sessions_per_week=5.0):
    """Run one contiguous path. Returns 'pass', 'bust' or 'timeout'."""
    eq = 0.0
    peak = 0.0
    floor = -MLL
    n = int(round(weeks * sessions_per_week))
    for k in range(min(n, len(pnl))):
        if eq + mae[k] * size <= floor:
            return "bust"
        eq += pnl[k] * size
        if eq <= floor:
            return "bust"
        peak = max(peak, eq)
        floor = max(floor, min(peak - MLL, 0.0))
        if eq >= TARGET:
            return "pass"
    return "timeout"


def contiguous(df, size, weeks):
    p = df["pnl"].to_numpy()
    m = df["mae"].to_numpy()
    n = int(round(weeks * 5))
    outs = {"pass": 0, "bust": 0, "timeout": 0}
    starts = range(0, max(len(p) - n, 1))
    for s in starts:
        outs[walk(p[s:s + n], m[s:s + n], size, weeks)] += 1
    tot = sum(outs.values())
    return outs["pass"] / tot, outs["bust"] / tot, tot


def iid(df, size, weeks, n_paths=20000, seed=7):
    rng = np.random.default_rng(seed)
    p = df["pnl"].to_numpy()
    m = df["mae"].to_numpy()
    n = int(round(weeks * 5))
    ok = bust = 0
    for _ in range(n_paths):
        idx = rng.integers(0, len(p), n)
        r = walk(p[idx], m[idx], size, weeks)
        ok += r == "pass"
        bust += r == "bust"
    return ok / n_paths, bust / n_paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", default="/Users/jg/auction/data/mnq_1m_all.parquet")
    ap.add_argument("--point-value", type=float, default=2.0)
    ap.add_argument("--cost", type=float, default=2.06)
    ap.add_argument("--label", default="MNQ")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10, 13])
    ap.add_argument("--weeks", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--out", default="/Users/jg/auction/runs/pure_beta.json")
    a = ap.parse_args()

    bars = pd.read_parquet(a.bars)
    if "et" in bars.columns:                 # es_1min_eth_frontmonth shape
        bars = bars.set_index(pd.DatetimeIndex(bars["et"])).drop(columns=["et"])
    bars = bars.tz_convert(ET).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    df = build_sessions(bars, a.point_value, a.cost)
    p = df["pnl"].to_numpy()
    t = p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))
    print(f"{a.label}: {len(df)} legal 18:00->16:45 sessions, "
          f"{df.entry.min().date()} .. {df.entry.max().date()}")
    print(f"  per session @1 contract: mean=${p.mean():.2f} t={t:.2f} "
          f"sd=${p.std(ddof=1):.2f} worst=${p.min():.2f} MAE p05=${np.percentile(df.mae, 5):.0f}")
    print(f"  forced round trips: 5/week = ${5*a.cost:.2f}/week/contract")
    ann = p.mean() / p.std(ddof=1) * np.sqrt(252)
    print(f"  annualised Sharpe of the raw hold: {ann:.2f}")

    rows = []
    for weeks in a.weeks:
        print(f"\n=== {a.label}, {weeks} weeks ===")
        print(f"{'size':>5} {'P(pass) contig':>15} {'P(bust)':>9} {'paths':>7} "
              f"{'P(pass) iid':>12}")
        for s in a.sizes:
            pc, bc, npaths = contiguous(df, s, weeks)
            pi, _ = iid(df, s, weeks)
            rows.append(dict(label=a.label, weeks=weeks, size=s,
                             p_pass_contig=round(pc, 4), p_bust_contig=round(bc, 4),
                             paths=npaths, p_pass_iid=round(pi, 4)))
            print(f"{s:>5} {pc:>15.3f} {bc:>9.3f} {npaths:>7} {pi:>12.3f}")
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
