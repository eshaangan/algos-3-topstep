#!/usr/bin/env python3
"""
Fetch 2026 year-to-date MES data from Databento and save ETH + RTH variants.

Outputs:
- data/processed/mes_2026_ytd_1m.h5
- data/processed/mes_2026_ytd_5m.h5
- data/processed/mes_2026_ytd_rth_5m.h5
- data/processed/mes_2026_ytd_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from ml_intraday_v3.live_trading.data_fetcher import LiveDataFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _resample_5m(bars_1m: pd.DataFrame) -> pd.DataFrame:
    bars_5m = bars_1m.resample("5min").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    ).dropna()
    bars_5m.index.name = "timestamp"
    return bars_5m


def _filter_rth(bars: pd.DataFrame) -> pd.DataFrame:
    idx_et = bars.index.tz_convert("US/Eastern")
    in_rth = (
        ((idx_et.hour > 9) | ((idx_et.hour == 9) & (idx_et.minute >= 30)))
        & ((idx_et.hour < 16) | ((idx_et.hour == 16) & (idx_et.minute == 0)))
    )
    rth = bars.loc[in_rth].copy()
    # Match the existing RTH file convention: exclude the 16:00 ET bar open.
    idx_et_rth = rth.index.tz_convert("US/Eastern")
    rth = rth.loc[idx_et_rth.time < pd.Timestamp("16:00").time()].copy()
    return rth


def fetch_2026_ytd(symbol: str, start_date: str, end_date: str) -> dict[str, Path]:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    fetcher = LiveDataFetcher(symbol=symbol, bar_size="1m", lookback_bars=100)
    logger.info("Fetching %s from %s to %s", symbol, start_date, end_date)
    bars_1m = fetcher.fetch_historical(start_date=start_date, end_date=end_date)
    if bars_1m.empty:
        raise ValueError("No data returned from Databento")

    if bars_1m.index.tz is None:
        bars_1m.index = bars_1m.index.tz_localize("UTC")
    else:
        bars_1m.index = bars_1m.index.tz_convert("UTC")
    bars_1m = bars_1m.sort_index()
    bars_1m.index.name = "timestamp"

    bars_5m = _resample_5m(bars_1m)
    bars_rth_5m = _filter_rth(bars_5m)

    output_dir = PROJECT_ROOT / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_1m = output_dir / "mes_2026_ytd_1m.h5"
    output_5m = output_dir / "mes_2026_ytd_5m.h5"
    output_rth_5m = output_dir / "mes_2026_ytd_rth_5m.h5"
    metadata_path = output_dir / "mes_2026_ytd_metadata.json"

    bars_1m.to_hdf(output_1m, key="bars_1min", mode="w")
    bars_5m.to_hdf(output_5m, key="bars_5min", mode="w")
    bars_rth_5m.to_hdf(output_rth_5m, key="bars_5min", mode="w")

    metadata = {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "bars_1m": {
            "path": str(output_1m.relative_to(PROJECT_ROOT)),
            "count": len(bars_1m),
            "start": str(bars_1m.index[0]),
            "end": str(bars_1m.index[-1]),
        },
        "bars_5m": {
            "path": str(output_5m.relative_to(PROJECT_ROOT)),
            "count": len(bars_5m),
            "start": str(bars_5m.index[0]),
            "end": str(bars_5m.index[-1]),
        },
        "bars_rth_5m": {
            "path": str(output_rth_5m.relative_to(PROJECT_ROOT)),
            "count": len(bars_rth_5m),
            "start": str(bars_rth_5m.index[0]),
            "end": str(bars_rth_5m.index[-1]),
        },
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved 1m bars to %s", output_1m)
    logger.info("Saved 5m ETH bars to %s", output_5m)
    logger.info("Saved 5m RTH bars to %s", output_rth_5m)
    logger.info("Saved metadata to %s", metadata_path)

    return {
        "bars_1m": output_1m,
        "bars_5m": output_5m,
        "bars_rth_5m": output_rth_5m,
        "metadata": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch 2026 YTD MES Databento data")
    parser.add_argument("--symbol", default="MES.c.0")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument(
        "--end-date",
        default=datetime.utcnow().strftime("%Y-%m-%d"),
        help="Inclusive end date in YYYY-MM-DD",
    )
    args = parser.parse_args()

    fetch_2026_ytd(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
    )


if __name__ == "__main__":
    main()
