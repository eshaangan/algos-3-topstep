"""Replay weekend_hold_v1 and monday_rth_v1 from MNQ 1-min history.

Reproduces exactly what the paper runners would have done, so that forward
weekends/Mondays missed by a dead runner can be recovered from history instead
of waited for.

TIMESTAMP TRUTH (the trap this project has hit before):
    Rithmic history `ts` (bar_end_datetime) is naive **fixed UTC-5**, not ET and
    not real UTC. Verified: the file for UTC day 2026-07-09 spans ts 19:01 (Jul 8)
    -> 19:01 (Jul 9), i.e. exactly UTC-5. ET is UTC-4 in EDT and UTC-5 in EST, so
    ET != ts + const. We localize to a fixed -05:00 offset and convert to
    America/New_York, which is correct in both halves of the year.

BAR CONVENTION: `ts` is the bar END. A bar labelled 17:00 covers 16:59->17:00.
At session reopen the first bar of the week ends :01, not :00 -- so an entry
"at 18:00 ET" resolves to the first bar ending at or after 18:00 ET.

Entry/exit mirror weekend_paper_runner.py: it polls once a minute inside a
window and takes the prevailing quote, so first-bar-at-or-after is the honest
executable mark.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

ET = "America/New_York"
# CORRECTED 2026-09-05. This was "-05:00" and that is wrong: it puts the RTH-open
# volume spike at 08:31 ET in winter instead of 09:31. Verified on both
# data/hist_1m24v/m_*.parquet and data/processed/mnq_1m_all.parquet -- the global
# 24h volume peak lands at 09:31 ET in BOTH seasons only under America/Chicago.
# The old fixed offset shifted every winter entry and exit by one hour.
RITHMIC_TZ = "America/Chicago"

# Cost model copied from weekend_paper_runner.py (NC=2, PV=2.0, COMM=0.62, TICK=0.25)
N_CONTRACTS = 2
POINT_VALUE = 2.0
COMMISSION_PER_SIDE = 0.62
SLIPPAGE_TICKS_PER_SIDE = 1.0
TICK_SIZE = 0.25

ROUND_TRIP_COST = (
    2 * COMMISSION_PER_SIDE * N_CONTRACTS
    + 2 * SLIPPAGE_TICKS_PER_SIDE * TICK_SIZE * POINT_VALUE * N_CONTRACTS
)  # $4.48 on 2 micros


@dataclass(frozen=True)
class Econ:
    """Contract economics. The default is the live spec: 2 MNQ micros."""

    point_value: float = POINT_VALUE
    contracts: int = N_CONTRACTS
    commission_per_side: float = COMMISSION_PER_SIDE
    slippage_ticks: float = SLIPPAGE_TICKS_PER_SIDE
    tick_size: float = TICK_SIZE

    @property
    def round_trip_cost(self) -> float:
        return (2 * self.commission_per_side * self.contracts
                + 2 * self.slippage_ticks * self.tick_size * self.point_value * self.contracts)


MNQ_ECON = Econ()
MES_ECON = Econ(point_value=5.0, contracts=2)


def _assert_open_spike(df: pd.DataFrame, label: str) -> None:
    """Guard the two-convention trap: RTH opens 09:30 ET all year, so the opening
    volume spike must land at 09:3x ET in BOTH seasons. A fixed-UTC-5 file treated
    as local (or vice versa) puts the winter spike an hour early."""
    v = pd.Series(df["vol"].to_numpy(), index=df["et"])
    bad = []
    for season, months in (("winter", [12, 1, 2]), ("summer", [6, 7, 8])):
        s = v[v.index.month.isin(months)]
        s = s[s.index.hour == 9]
        if s.empty:
            continue
        pk = int(s.groupby(s.index.minute).sum().idxmax())
        if not 29 <= pk <= 33:
            bad.append(f"{season} 09:{pk:02d}")
    if bad:
        raise SystemExit(
            f"TIMESTAMP CONVENTION ERROR in {label}: RTH-open spike at {', '.join(bad)} ET, "
            "expected 09:30-09:33 in both seasons.\n"
            "  data/processed/mnq_1m_all.parquet is true America/Chicago (do NOT apply -05:00)\n"
            "  data/hist_1m24v/m_*.parquet are fixed UTC-5 (DO apply -05:00)\n"
            "Mixing them shifts every winter entry/exit by one hour.")


def load_es_eth(path: str) -> pd.DataFrame:
    """Load the front-month ES ETH parquet, which is already true tz-aware ET.

    Built by build_es_eth_1min.py. Its `et` column is genuinely ET (Databento
    stamps are real UTC), so the fixed UTC-5 correction that Rithmic history
    needs must NOT be applied here.
    """
    df = pd.read_parquet(path)
    df["et"] = pd.to_datetime(df["et"])
    if df["et"].dt.tz is None:
        raise SystemExit("es-eth parquet must carry tz-aware timestamps")
    df["et"] = df["et"].dt.tz_convert(ET)
    df = df.rename(columns={"volume": "vol"})
    df = df.drop_duplicates(subset="et").sort_values("et").reset_index(drop=True)
    return df[["et", "open", "high", "low", "close", "vol"]]


def load_bars(data_dir: str) -> pd.DataFrame:
    """Load all day parquets, convert the fixed UTC-5 stamps to true ET."""
    if data_dir.endswith(".parquet") and os.path.exists(data_dir):
        df = pd.read_parquet(data_dir)          # consolidated cache
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "m_*.parquet")))
        if not files:
            raise SystemExit(f"no m_*.parquet under {data_dir}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ts = pd.to_datetime(df["ts"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)
    df["et"] = ts.dt.tz_localize(RITHMIC_TZ, ambiguous="NaT", nonexistent="NaT").dt.tz_convert(ET)
    df = df.drop_duplicates(subset="et").sort_values("et").reset_index(drop=True)
    df = df[["et", "open", "high", "low", "close", "vol"]]
    _assert_open_spike(df, data_dir)
    return df


class BarIndex:
    """Sorted view over the bar frame; O(log n) lookups instead of full scans."""

    def __init__(self, bars: pd.DataFrame):
        # search on UTC-naive int64 so searchsorted stays on a real datetime64 array
        self.key = bars["et"].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")
        self.et = bars["et"].reset_index(drop=True)
        self.close = bars["close"].to_numpy(dtype=float)

    @staticmethod
    def _k(when: pd.Timestamp) -> "np.datetime64":
        return np.datetime64(when.tz_convert("UTC").tz_localize(None), "ns")

    def first_at_or_after(self, when: pd.Timestamp, window_min: int):
        i = int(np.searchsorted(self.key, self._k(when), "left"))
        if i >= len(self.key):
            return None, None
        if self.key[i] > self._k(when + pd.Timedelta(minutes=window_min)):
            return None, None
        return float(self.close[i]), self.et.iloc[i]

    def last_at_or_before(self, when: pd.Timestamp, window_min: int):
        i = int(np.searchsorted(self.key, self._k(when), "right")) - 1
        if i < 0:
            return None, None
        if self.key[i] < self._k(when - pd.Timedelta(minutes=window_min)):
            return None, None
        return float(self.close[i]), self.et.iloc[i]


@dataclass
class Trade:
    key: str
    entry_px: float
    exit_px: float
    entry_at: pd.Timestamp
    exit_at: pd.Timestamp
    pnl: float


def _pnl(entry_px: float, exit_px: float, econ: Econ = MNQ_ECON) -> float:
    gross = (exit_px - entry_px) * econ.point_value * econ.contracts
    return round(gross - econ.round_trip_cost, 2)


def replay_weekend(bars: pd.DataFrame, econ: Econ = MNQ_ECON) -> pd.DataFrame:
    """Long Sun 18:00 ET -> Mon 15:59 ET, keyed by the Monday date."""
    idx = BarIndex(bars)
    days = pd.Series(bars["et"].dt.date.unique())
    sundays = [d for d in days if pd.Timestamp(d).weekday() == 6]
    out: list[Trade] = []
    for sun in sundays:
        mon = pd.Timestamp(sun) + pd.Timedelta(days=1)
        entry_t = pd.Timestamp(sun, tz=ET) + pd.Timedelta(hours=18)
        exit_t = pd.Timestamp(mon.date(), tz=ET) + pd.Timedelta(hours=15, minutes=59)
        # runner's own window: it only enters within 18:00-18:10
        e_px, e_at = idx.first_at_or_after(entry_t, 10)
        x_px, x_at = idx.last_at_or_before(exit_t, 30)
        if e_px is None or x_px is None:
            continue
        out.append(Trade(str(mon.date()), e_px, x_px, e_at, x_at, _pnl(e_px, x_px, econ)))
    return pd.DataFrame([t.__dict__ for t in out])


def replay_monday(bars: pd.DataFrame, econ: Econ = MNQ_ECON) -> pd.DataFrame:
    """Long Mon 09:30 ET -> Mon 16:00 ET."""
    idx = BarIndex(bars)
    days = pd.Series(bars["et"].dt.date.unique())
    mondays = [d for d in days if pd.Timestamp(d).weekday() == 0]
    out: list[Trade] = []
    for mon in mondays:
        entry_t = pd.Timestamp(mon, tz=ET) + pd.Timedelta(hours=9, minutes=30)
        exit_t = pd.Timestamp(mon, tz=ET) + pd.Timedelta(hours=16)
        e_px, e_at = idx.first_at_or_after(entry_t, 10)
        x_px, x_at = idx.last_at_or_before(exit_t, 30)
        if e_px is None or x_px is None:
            continue
        out.append(Trade(str(mon), e_px, x_px, e_at, x_at, _pnl(e_px, x_px, econ)))
    return pd.DataFrame([t.__dict__ for t in out])


def summarize(name: str, tr: pd.DataFrame) -> None:
    if tr.empty:
        print(f"\n{name}: no trades")
        return
    pnl = tr["pnl"].to_numpy()
    n = len(pnl)
    mean = pnl.mean()
    sd = pnl.std(ddof=1) if n > 1 else float("nan")
    t = mean / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    wr = 100.0 * (pnl > 0).mean()
    print(f"\n=== {name} ===")
    print(f"n={n}  mean=${mean:+,.2f}  total=${pnl.sum():+,.0f}  sd=${sd:,.0f}  t={t:+.2f}  WR={wr:.1f}%")
    yr = tr.assign(year=pd.to_datetime(tr["key"]).dt.year).groupby("year")["pnl"]
    print("  by year:")
    for year, grp in yr:
        print(f"    {year}: n={len(grp):3d}  mean=${grp.mean():+8.2f}  total=${grp.sum():+9,.0f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/hist_1m24v")
    ap.add_argument("--source", choices=("rithmic", "es-eth"), default="rithmic",
                    help="rithmic = fixed UTC-5 day parquets; es-eth = already-ET front-month ES parquet")
    ap.add_argument("--econ", choices=("mnq", "mes"), default="mnq",
                    help="contract economics; mes is $5/pt for the ES price series")
    ap.add_argument("--by-era", action="store_true",
                    help="also break results at 2020, the boundary the strategy was found on")
    ap.add_argument("--start", default=None, help="ET date filter, inclusive")
    ap.add_argument("--end", default=None, help="ET date filter, inclusive")
    ap.add_argument("--out-prefix", default=None, help="write <prefix>_weekend.csv / _monday.csv")
    a = ap.parse_args()

    econ = MES_ECON if a.econ == "mes" else MNQ_ECON
    bars = load_es_eth(a.data_dir) if a.source == "es-eth" else load_bars(a.data_dir)
    print(f"loaded {len(bars):,} bars  {bars['et'].min()} -> {bars['et'].max()}")
    if a.start:
        bars = bars[bars["et"] >= pd.Timestamp(a.start, tz=ET)]
    if a.end:
        bars = bars[bars["et"] <= pd.Timestamp(a.end, tz=ET) + pd.Timedelta(days=1)]

    wk = replay_weekend(bars, econ)
    mo = replay_monday(bars, econ)
    tag = f"2 {a.econ.upper()}"
    summarize(f"weekend_hold_v1  (Sun 18:00 -> Mon 15:59 ET, {tag})", wk)
    summarize(f"monday_rth_v1    (Mon 09:30 -> Mon 16:00 ET, {tag})", mo)

    if a.by_era:
        for name, tr in (("weekend_hold_v1", wk), ("monday_rth_v1", mo)):
            if tr.empty:
                continue
            key = pd.to_datetime(tr["key"])
            print(f"\n--- {name}: era split at 2020-01-01 "
                  f"(the strategy was found on 2019+ data, so pre-2020 is out of sample) ---")
            for era, sub in (("PRE-2020 ", tr[key < "2020-01-01"]),
                             ("2020+    ", tr[key >= "2020-01-01"])):
                if len(sub) < 5:
                    print(f"  {era} n={len(sub)} (too few)")
                    continue
                pnl = sub["pnl"].to_numpy()
                t = pnl.mean() / (pnl.std(ddof=1) / np.sqrt(len(pnl)))
                print(f"  {era} n={len(pnl):4d}  mean=${pnl.mean():+8.2f}  "
                      f"total=${pnl.sum():+10,.0f}  t={t:+5.2f}  WR={100 * (pnl > 0).mean():.1f}%")

    if a.out_prefix:
        wk.to_csv(f"{a.out_prefix}_weekend.csv", index=False)
        mo.to_csv(f"{a.out_prefix}_monday.csv", index=False)
        print(f"\nwrote {a.out_prefix}_weekend.csv ({len(wk)}) and _monday.csv ({len(mo)})")


if __name__ == "__main__":
    main()
