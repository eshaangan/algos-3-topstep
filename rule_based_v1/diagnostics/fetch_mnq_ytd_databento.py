"""Fetch refreshed MNQ 2026 YTD bars from Databento.

Writes:
  data/processed/mnq_2026ytd_databento_1min_eth.h5  key=bars_1min_eth
  data/processed/mnq_2026ytd_databento_5min_rth.h5  key=bars_5min

Run:
  python rule_based_v1/diagnostics/fetch_mnq_ytd_databento.py --end 2026-04-17T21:00:00+00:00
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_1M = ROOT / "data" / "processed" / "mnq_2026ytd_databento_1min_eth.h5"
OUT_5M = ROOT / "data" / "processed" / "mnq_2026ytd_databento_5min_rth.h5"


def _default_end() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _rth_5min(df_1m: pd.DataFrame) -> pd.DataFrame:
    df5 = df_1m.resample("5min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open"])
    df5.index = df5.index.tz_convert("US/Eastern")
    rth = (
        ((df5.index.hour == 9) & (df5.index.minute >= 30))
        | ((df5.index.hour > 9) & (df5.index.hour < 16))
    )
    return df5.loc[rth, ["open", "high", "low", "close", "volume"]].copy()


def fetch(start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set")

    import databento as db

    print(f"Fetching Databento GLBX.MDP3 MNQ.c.0 ohlcv-1m {start} -> {end}")
    client = db.Historical(key=api_key)
    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["MNQ.c.0"],
        schema="ohlcv-1m",
        start=start,
        end=end,
        stype_in="continuous",
    )
    df = data.to_df()
    if df.empty:
        raise RuntimeError("Databento returned no MNQ bars")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("US/Eastern")
    df = df[["open", "high", "low", "close", "volume"]].sort_index()
    df5 = _rth_5min(df)

    OUT_1M.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(str(OUT_1M), key="bars_1min_eth", mode="w", complevel=5)
    df5.to_hdf(str(OUT_5M), key="bars_5min", mode="w", complevel=5)

    print(f"1m ETH bars: {len(df):,}  {df.index[0]} -> {df.index[-1]}")
    print(f"5m RTH bars: {len(df5):,}  {df5.index[0]} -> {df5.index[-1]}")
    print(f"Saved {OUT_1M}")
    print(f"Saved {OUT_5M}")
    return df, df5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default=_default_end())
    args = parser.parse_args()
    fetch(args.start, args.end)


if __name__ == "__main__":
    main()
