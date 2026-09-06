"""Break-even cost analysis: how cheap would execution have to be for an edge to survive?

The project's meta-finding is that real structural edges keep dying to friction
rather than to absence of signal -- e.g. M6E short-session carry, t=+3.08, killed
only because gross ~$5.5/day lost to $7.48 of round-trip cost per 2 lots.

This asks the inverse question for any trade series: given the GROSS per-trade
edge, what round-trip cost would drive it to zero, and how does that compare to
what we actually pay? That converts "friction ate it" into a number you can shop
against a broker, a venue, or a contract size.

Usage:
    python3 rule_based_v1/validation/cost_floor.py --csv <trades.csv> --gross-col pnl_gross
    python3 rule_based_v1/validation/cost_floor.py --session-hold --instrument m6e ...
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

ET = "America/New_York"
# CORRECTED 2026-09-05: stamps are DST-aware America/Chicago, NOT fixed UTC-5.
# The old value shifted every winter timestamp by one hour. See tzguard.py.
RITHMIC_TZ = "America/Chicago"

# Per-instrument economics. point_value is $ per 1.00 price move for ONE contract.
INSTRUMENTS = {
    "mnq": {"dir": "data/hist_1m24v", "point_value": 2.0, "tick": 0.25, "comm": 0.62},
    "mes": {"dir": "data/hist_1m24_mes", "point_value": 5.0, "tick": 0.25, "comm": 0.62},
    "mgc": {"dir": "data/hist_1m24_mgc", "point_value": 10.0, "tick": 0.10, "comm": 0.62},
    "mcl": {"dir": "data/hist_1m24_mcl", "point_value": 100.0, "tick": 0.01, "comm": 0.62},
    "m6e": {"dir": "data/hist_1m24_m6e", "point_value": 12500.0, "tick": 0.0001, "comm": 0.62},
    "zn": {"dir": "data/hist_1m24_zn", "point_value": 1000.0, "tick": 0.015625, "comm": 0.62},
}


def load_bars(data_dir: str, cache: str | None = None) -> pd.DataFrame:
    if cache and os.path.exists(cache):
        df = pd.read_parquet(cache)
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "m_*.parquet")))
        if not files:
            raise SystemExit(f"no m_*.parquet under {data_dir}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ts = pd.to_datetime(df["ts"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)
    df["et"] = ts.dt.tz_localize(RITHMIC_TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert(ET)
    return df.drop_duplicates(subset="et").sort_values("et").reset_index(drop=True)


def session_hold_gross(
    bars: pd.DataFrame, entry_hhmm: str, exit_hhmm: str, weekdays, point_value: float,
    n_contracts: int, direction: str = "long"
) -> pd.DataFrame:
    """Hold from entry to exit each qualifying day. Returns GROSS pnl (no costs)."""
    eh, em = (int(x) for x in entry_hhmm.split(":"))
    xh, xm = (int(x) for x in exit_hhmm.split(":"))
    # BUG GUARD: search key and every bound must be in the SAME frame.
    # .dt.tz_localize(None) would strip tz keeping ET wall-clock while the bounds
    # below are UTC -- a silent 4-5h offset. Convert to UTC first, always.
    key = bars["et"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
    close = bars["close"].to_numpy(float)

    def _k(when):
        return np.datetime64(when.tz_convert("UTC").tz_localize(None), "ns")

    def at(when, after=True, window=15):
        k = _k(when)
        if after:
            i = int(np.searchsorted(key, k, "left"))
            if i >= len(key) or key[i] > _k(when + pd.Timedelta(minutes=window)):
                return None
        else:
            i = int(np.searchsorted(key, k, "right")) - 1
            if i < 0 or key[i] < _k(when - pd.Timedelta(minutes=window)):
                return None
        return float(close[i])

    rows = []
    for d in pd.Series(bars["et"].dt.date.unique()):
        if pd.Timestamp(d).weekday() not in weekdays:
            continue
        t_in = pd.Timestamp(d, tz=ET) + pd.Timedelta(hours=eh, minutes=em)
        t_out = pd.Timestamp(d, tz=ET) + pd.Timedelta(hours=xh, minutes=xm)
        if t_out <= t_in:
            t_out += pd.Timedelta(days=1)
        e, x = at(t_in, True), at(t_out, False)
        if e is None or x is None:
            continue
        sign = 1.0 if direction == "long" else -1.0
        rows.append({"date": d, "entry": e, "exit": x,
                     "gross": sign * (x - e) * point_value * n_contracts})
    return pd.DataFrame(rows)


def report(name: str, gross: np.ndarray, actual_cost: float, n_contracts: int) -> dict:
    n = len(gross)
    if n < 2:
        print(f"{name}: too few trades ({n})")
        return {}
    mean_g, sd = gross.mean(), gross.std(ddof=1)
    t_gross = mean_g / (sd / np.sqrt(n))
    # break-even round-trip cost = the gross mean itself (net mean = gross - cost)
    breakeven = mean_g
    net_mean = mean_g - actual_cost
    t_net = net_mean / (sd / np.sqrt(n))
    # cost at which t_net would reach 2.0 (a tradeable-confidence bar)
    cost_for_t2 = mean_g - 2.0 * sd / np.sqrt(n)
    print(f"\n=== {name} ===")
    print(f"  n={n}  gross mean=${mean_g:+,.2f}/trade  sd=${sd:,.0f}  t_gross={t_gross:+.2f}")
    print(f"  BREAK-EVEN round-trip cost : ${breakeven:,.2f}   (per {n_contracts} contracts)")
    print(f"  actual round-trip cost     : ${actual_cost:,.2f}")
    print(f"  net mean at actual cost    : ${net_mean:+,.2f}/trade   t_net={t_net:+.2f}")
    # NOTE: this verdict is ONLY about friction. It says nothing about whether the
    # edge is real or will persist -- these are in-sample stats on discovery data.
    # "NOT COST-LIMITED" must never be read as "tradeable".
    verdict = (
        "NOT COST-LIMITED (says nothing about whether the edge is real)"
        if net_mean > 0 and t_net >= 2
        else "marginal after costs" if net_mean > 0
        else "KILLED BY FRICTION -- significant gross edge, eaten by cost"
        if mean_g > 0 and t_gross >= 2
        else "NO SIGNIFICANT GROSS EDGE -- cost is not what is wrong" if mean_g > 0
        else "no gross edge -- friction is not the problem"
    )
    print(f"  cost needed for t_net>=2.0 : ${cost_for_t2:,.2f}"
          f"  ({'UNREACHABLE - fails t>=2 even at ZERO cost' if cost_for_t2 <= 0 else f'{cost_for_t2/actual_cost:.2f}x current'})")
    print(f"  VERDICT: {verdict}")
    return {"name": name, "n": n, "gross_mean": mean_g, "t_gross": t_gross,
            "breakeven_cost": breakeven, "actual_cost": actual_cost,
            "net_mean": net_mean, "t_net": t_net, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", default="mnq", choices=sorted(INSTRUMENTS))
    ap.add_argument("--cache", default=None, help="consolidated parquet to use instead of day files")
    ap.add_argument("--entry", required=True, help="HH:MM ET")
    ap.add_argument("--exit", dest="exit_t", required=True, help="HH:MM ET")
    ap.add_argument("--weekdays", default="0,1,2,3,4", help="0=Mon")
    ap.add_argument("--contracts", type=int, default=2)
    ap.add_argument("--slippage-ticks", type=float, default=1.0, help="per side")
    ap.add_argument("--direction", default="long", choices=["long", "short"])
    ap.add_argument("--label", default=None)
    a = ap.parse_args()

    spec = INSTRUMENTS[a.instrument]
    bars = load_bars(spec["dir"], a.cache)
    print(f"{a.instrument}: {len(bars):,} bars  {bars['et'].min()} -> {bars['et'].max()}")
    wd = {int(x) for x in a.weekdays.split(",")}
    tr = session_hold_gross(bars, a.entry, a.exit_t, wd, spec["point_value"], a.contracts, a.direction)
    cost = (2 * spec["comm"] * a.contracts
            + 2 * a.slippage_ticks * spec["tick"] * spec["point_value"] * a.contracts)
    label = a.label or f"{a.instrument.upper()} {a.direction} {a.entry}->{a.exit_t} ET x{a.contracts}"
    report(label, tr["gross"].to_numpy(), cost, a.contracts)


if __name__ == "__main__":
    main()
