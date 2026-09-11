"""Is there an intraday trend to follow at all? Measure it directly.

Trend following needs one thing: POSITIVE AUTOCORRELATION of returns at the
horizon you trade. Everything else -- entries, stops, filters, position sizing --
is a way of harvesting that autocorrelation. If it is zero, no parameter set
recovers it, and a config search will only find the ones that overfit.

So rather than add to the ~95 price/volume configs already killed, this measures
the primitive:

    corr( return over the last L minutes , return over the next H minutes )

on NON-OVERLAPPING blocks, RTH only, and then prices the naive trade that would
harvest it: go with the sign of the last L, hold H, pay the MEASURED round trip
from mbo_fill_model at the size the tail budget allows.

Reported both gross and net, because a family can have a real but sub-cost
signal, and that distinction is what the ledger's cost-floor rule is about.
"""
import argparse
import json

import numpy as np
import pandas as pd

ET = "America/New_York"
PV = 2.0
RISK_BUDGET = 1800.0

# measured round turn per contract by order size (runs/fill_costs.parquet)
COST_CURVE = {1: 2.047, 2: 2.101, 4: 2.191, 6: 2.269, 9: 2.388,
              13: 2.558, 20: 2.844, 40: 3.562, 60: 4.205}


def cost_at(size):
    xs = np.array(sorted(COST_CURVE))
    ys = np.array([COST_CURVE[k] for k in xs])
    return float(np.interp(size, xs, ys))


def blocks(rth, minutes):
    """Non-overlapping intraday blocks: (entry, exit, adverse) in $ per micro."""
    out = []
    for _, day in rth.groupby(rth.index.date):
        o = day["open"].to_numpy(); c = day["close"].to_numpy()
        hi = day["high"].to_numpy(); lo = day["low"].to_numpy()
        n = len(day) // minutes
        for b in range(n):
            s, e = b * minutes, (b + 1) * minutes
            out.append((o[s], c[e - 1],
                        max(o[s] - lo[s:e].min(), hi[s:e].max() - o[s]) * PV))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bars", default="/Users/jg/auction/data/mnq_1m_all.parquet")
    ap.add_argument("--label", default="MNQ")
    ap.add_argument("--point-value", type=float, default=2.0)
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 15, 30, 60, 120])
    ap.add_argument("--out", default="/Users/jg/auction/runs/intraday_trend.json")
    a = ap.parse_args()
    global PV
    PV = a.point_value

    bars = pd.read_parquet(a.bars)
    if "et" in bars.columns:
        bars = bars.set_index(pd.DatetimeIndex(bars["et"])).drop(columns=["et"])
    bars = bars.tz_convert(ET).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    m = bars.index.hour * 60 + bars.index.minute
    rth = bars[(m >= 570) & (m < 960) & (bars.index.dayofweek < 5)]
    print(f"{a.label}: {len(rth):,} RTH 1-min bars, "
          f"{rth.index.min().date()} .. {rth.index.max().date()}\n")

    rows = []
    print("THE PRIMITIVE: does a move predict the next move of the same length?")
    print(f"{'horizon':>8} {'n blocks':>9} {'autocorr':>10} {'t':>8} "
          f"{'gross $/trade':>14} {'size':>5} {'net $/trade':>12} {'net t':>8}")
    for H in a.horizons:
        arr = blocks(rth, H)
        ret = (arr[:, 1] - arr[:, 0]) * PV
        adv = arr[:, 2]
        prev, nxt = ret[:-1], ret[1:]
        r = float(np.corrcoef(prev, nxt)[0, 1])
        n = len(prev)
        t = r * np.sqrt(n - 2) / np.sqrt(max(1e-12, 1 - r * r))
        # the naive trend trade: go with the sign of the last block, hold one block
        gross = np.sign(prev) * nxt
        size = max(int(RISK_BUDGET // np.percentile(adv, 95)), 1)
        net = gross - cost_at(size)
        nt = net.mean() / (net.std(ddof=1) / np.sqrt(len(net)))
        rows.append(dict(horizon=H, n=n, autocorr=round(r, 5), t=round(float(t), 2),
                         gross=round(float(gross.mean()), 3), size=size,
                         cost=round(cost_at(size), 3),
                         net=round(float(net.mean()), 3), net_t=round(float(nt), 2)))
        print(f"{H:>8} {n:>9,} {r:>10.4f} {t:>8.2f} {gross.mean():>14.3f} "
              f"{size:>5} {net.mean():>12.3f} {nt:>8.2f}")

    print("\ninterpretation:")
    print("  autocorr > 0 = trend (momentum). autocorr < 0 = reversal (fade).")
    print("  gross is per micro BEFORE cost; net charges the measured round turn.")
    print("  a family is only alive if gross clears cost by a margin, at a size")
    print("  the $1,800 tail budget allows.")
    json.dump(rows, open(a.out, "w"), indent=2)
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
