"""
Build 1-minute OFI (Order Flow Imbalance) parquet files from Databento MNQ trade CSVs.

Auto-detects gzip vs plain CSV. Streams in chunks to handle 35M+ row files.

Output columns per 1-min bar:
  ts           - bar timestamp (UTC, minute-level)
  buy_vol      - aggressive buy volume (side=B)
  sell_vol     - aggressive sell volume (side=A)
  ofi          - buy_vol - sell_vol
  total_vol    - buy_vol + sell_vol
  n_trades     - number of trades
  vwap         - volume-weighted average price
"""

import gzip
import io
import os
import sys
from pathlib import Path

import pandas as pd

PRICE_SCALE = 1e9  # Databento fixed-point price divisor
CHUNK_ROWS = 1_000_000

DATA_DIR = Path(__file__).parents[2] / "data" / "processed"
OUT_DIR = DATA_DIR / "ofi_1min"
OUT_DIR.mkdir(exist_ok=True)

FILES = [
    "mnq_trades_dec2025.csv.gz",
    "mnq_trades_jan_feb9_2026.csv.gz",
    "mnq_trades_mar_may_2026.csv.gz",
]

USECOLS = ["ts_recv", "side", "price", "size"]
DTYPES = {"side": "category", "price": "float64", "size": "float64"}


def open_auto(path: Path):
    """Open gzip or plain CSV transparently."""
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rt")
    return open(path, "r")


def process(path: Path, out_path: Path) -> None:
    print(f"Processing {path.name} ...", flush=True)
    chunks = []

    fh = open_auto(path)
    try:
        reader = pd.read_csv(
            fh,
            usecols=USECOLS,
            dtype=DTYPES,
            chunksize=CHUNK_ROWS,
        )
        for i, chunk in enumerate(reader):
            # Drop rows with NA price/size (data quality gaps in some months)
            chunk = chunk.dropna(subset=["price", "size"])
            if chunk.empty:
                continue
            # Parse nanosecond timestamps -> UTC minute bars
            chunk["ts"] = pd.to_datetime(chunk["ts_recv"], unit="ns", utc=True).dt.floor("1min")
            chunk["price_f"] = chunk["price"] / PRICE_SCALE
            chunk["buy_vol"] = chunk["size"].where(chunk["side"] == "B", 0.0)
            chunk["sell_vol"] = chunk["size"].where(chunk["side"] == "A", 0.0)
            chunk["dv"] = chunk["price_f"] * chunk["size"]  # dollar-volume for VWAP

            agg = chunk.groupby("ts", observed=True).agg(
                buy_vol=("buy_vol", "sum"),
                sell_vol=("sell_vol", "sum"),
                total_vol=("size", "sum"),
                n_trades=("size", "count"),
                dv=("dv", "sum"),
            )
            chunks.append(agg)
            if (i + 1) % 10 == 0:
                print(f"  chunk {i+1} done ({(i+1)*CHUNK_ROWS:,} rows)", flush=True)
    finally:
        fh.close()

    df = pd.concat(chunks).groupby("ts").sum()
    df["ofi"] = df["buy_vol"] - df["sell_vol"]
    df["vwap"] = (df["dv"] / df["total_vol"]).round(4)
    df = df.drop(columns=["dv"])
    df = df.sort_index()

    df.to_parquet(out_path)
    print(f"  -> {out_path.name}  ({len(df):,} bars, {df.index[0]} to {df.index[-1]})", flush=True)


if __name__ == "__main__":
    for fname in FILES:
        src = DATA_DIR / fname
        if not src.exists():
            print(f"SKIP (not found): {src}", flush=True)
            continue
        stem = fname.replace(".csv.gz", "").replace(".csv", "")
        dst = OUT_DIR / f"{stem}_ofi_1min.parquet"
        if dst.exists():
            print(f"SKIP (already exists): {dst.name}", flush=True)
            continue
        process(src, dst)

    print("Done.", flush=True)
