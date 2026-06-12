"""Build 1-minute order-flow bars from the raw Databento MNQ trade CSVs.

The 5-min microstructure parquet showed no edge; the order-flow edge (if any)
lives at sub-minute scale. This aggregates the raw tick prints to 1-minute bars
with order-flow features, streaming in chunks so memory stays bounded.

Input CSVs (mislabeled .gz but plain CSV): ts_recv(ns), side(B/A/N), price(1e-9), size.
Output: data/processed/mnq_microflow_1min.parquet
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = [
    ROOT / "data/processed/mnq_trades_dec2025.csv.gz",
    ROOT / "data/processed/mnq_trades_jan_feb9_2026.csv.gz",
    ROOT / "data/processed/mnq_trades_mar_may_2026.csv.gz",
]
OUT = ROOT / "data/processed/mnq_microflow_1min.parquet"
LARGE = 5            # contracts: "institutional" trade
CHUNK = 3_000_000

def agg_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["price"] > 0].copy()
    df["price"] = df["price"] / 1e9
    df["min"] = pd.to_datetime(df["ts_recv"], unit="ns", utc=True).dt.floor("1min")
    b = df["side"].values == "B"; a = df["side"].values == "A"
    sz = df["size"].values; lg = sz >= LARGE
    df["buy"] = np.where(b, sz, 0); df["sell"] = np.where(a, sz, 0)
    df["lgbuy"] = np.where(b & lg, sz, 0); df["lgsell"] = np.where(a & lg, sz, 0)
    df["lgvol"] = np.where(lg, sz, 0)
    g = df.groupby("min")
    out = g.agg(o=("price", "first"), h=("price", "max"), l=("price", "min"),
                c=("price", "last"), vol=("size", "sum"), n=("size", "count"),
                buy=("buy", "sum"), sell=("sell", "sum"),
                lgbuy=("lgbuy", "sum"), lgsell=("lgsell", "sum"), lgvol=("lgvol", "sum"))
    return out.reset_index()

def main():
    t0 = time.time(); parts = []
    for f in FILES:
        if not f.exists():
            print(f"skip missing {f.name}"); continue
        print(f"reading {f.name} …", flush=True); nrows = 0
        for chunk in pd.read_csv(f, compression=None, usecols=["ts_recv", "side", "price", "size"],
                                 chunksize=CHUNK):
            parts.append(agg_chunk(chunk)); nrows += len(chunk)
            print(f"  {nrows:,} rows ({time.time()-t0:.0f}s)", flush=True)
    allp = pd.concat(parts, ignore_index=True)
    g = allp.groupby("min")
    bars = g.agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"),
                 vol=("vol", "sum"), n=("n", "sum"), buy=("buy", "sum"), sell=("sell", "sum"),
                 lgbuy=("lgbuy", "sum"), lgsell=("lgsell", "sum"), lgvol=("lgvol", "sum"))
    bars = bars.sort_index()
    bars.index.name = "ts"
    bars.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "vol": "volume"},
                inplace=True)
    bars.to_parquet(OUT)
    print(f"\nWrote {len(bars):,} 1-min bars → {OUT}")
    print(f"range {bars.index[0]} -> {bars.index[-1]}  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
