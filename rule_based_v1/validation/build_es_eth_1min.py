"""Front-month-pinned ES 1-minute ETH series from the raw Databento export.

The existing data/processed/es_bars_2010_2025.h5 is RTH-only, which cannot express
any overnight window. The raw GLBX export underneath it has all 24 hours back to
2010-06, but it carries every listed contract plus ~10% calendar-spread rows, and
the ledger's standing warning applies: mixing contract marks corrupts everything.

So: drop spread symbols, pick ONE front contract per CME trade date by that date's
traded volume, and keep only its bars. Roll days therefore switch cleanly at a
session boundary rather than mid-session.

CME trade date: the session opening 18:00 ET belongs to the NEXT calendar day, so
the trade date is (timestamp_ET + 7h).date().

Usage:
    python3 rule_based_v1/validation/build_es_eth_1min.py \
        --csv data/raw/GLBX-20251220-LWFB9HCEL5/glbx-mdp3-20100606-20251219.ohlcv-1m.csv \
        --out data/processed/es_1min_eth_frontmonth.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ET = "America/New_York"
USECOLS = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build front-month ES 1-min ETH bars")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--chunksize", type=int, default=1_000_000)
    args = ap.parse_args()

    frames = []
    total = spreads = 0
    for chunk in pd.read_csv(args.csv, usecols=USECOLS, chunksize=args.chunksize):
        total += len(chunk)
        # Calendar spreads look like "ESM0-ESU0" and are not a tradeable outright.
        keep = ~chunk["symbol"].str.contains("-", regex=False)
        spreads += int((~keep).sum())
        frames.append(chunk[keep])
    df = pd.concat(frames, ignore_index=True)
    del frames
    print(f"rows: {total:,} total, {spreads:,} spread rows dropped, {len(df):,} outright")

    ts = pd.to_datetime(df.pop("ts_event"), format="ISO8601", utc=True)
    df["et"] = ts.dt.tz_convert(ET)
    df["trade_date"] = (df["et"] + pd.Timedelta(hours=7)).dt.date

    vol = df.groupby(["trade_date", "symbol"], observed=True)["volume"].sum()
    front = vol.groupby(level=0).idxmax().map(lambda k: k[1]).rename("front")
    print(f"trade dates: {len(front):,}   distinct front contracts: {front.nunique()}")

    df = df.join(front, on="trade_date")
    df = df[df["symbol"] == df["front"]].drop(columns=["front"])
    df = df.sort_values("et").drop_duplicates(subset="et", keep="last")

    # A roll should be a same-price handoff, not a price jump. Report the worst
    # session-boundary moves so contamination is visible rather than assumed away.
    daily = df.groupby("trade_date")["close"].last()
    jumps = daily.diff().abs().sort_values(ascending=False)
    print("\nlargest close-to-close moves at trade-date boundaries (sanity, not a filter):")
    for d, j in jumps.head(5).items():
        print(f"   {d}  {j:+.2f} pts")

    out = df[["et", "open", "high", "low", "close", "volume", "symbol"]].reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\nwrote {args.out}  rows={len(out):,}  "
          f"{out['et'].min()} .. {out['et'].max()}")


if __name__ == "__main__":
    main()
