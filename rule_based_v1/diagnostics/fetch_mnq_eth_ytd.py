"""Fetch MNQ 2026 YTD 1-min bars with full ETH coverage (no RTH filter).

Saves to data/processed/mnq_2026ytd_1min_eth.h5  key=bars_1min_eth

ETH bars are needed for compute_eth_metrics() in oiar_backtest.py —
overnight window = 16:01 prior day → 09:29 current day.

Usage:
    cd "algos 3 topstep"
    python rule_based_v1/diagnostics/fetch_mnq_eth_ytd.py
    python rule_based_v1/diagnostics/fetch_mnq_eth_ytd.py --start 2026-01-01
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "data" / "processed" / "mnq_2026ytd_1min_eth.h5"


def fetch(start: str = "2026-01-01") -> pd.DataFrame:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set in .env")

    import databento as db

    # Clamp end to available Databento ceiling for GLBX.MDP3
    # (dataset lags ~24-48h; adjust if stale)
    end_utc = "2026-03-25T23:40:00+00:00"

    print(f"Fetching MNQ.c.0  {start} → {end_utc}  (full ETH, 1-min) …")
    client = db.Historical(key=api_key)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["MNQ.c.0"],
        schema="ohlcv-1m",
        start=start,
        end=end_utc,
        stype_in="continuous",
    )
    df = data.to_df()
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("US/Eastern")
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    print(f"Fetched {len(df):,} 1-min bars  "
          f"{df.index[0].date()} → {df.index[-1].date()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(str(OUT_PATH), key="bars_1min_eth", mode="w", complevel=5)
    print(f"Saved → {OUT_PATH}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-01")
    args = parser.parse_args()
    fetch(start=args.start)


if __name__ == "__main__":
    main()
