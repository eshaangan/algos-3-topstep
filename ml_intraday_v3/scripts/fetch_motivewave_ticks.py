"""
Convert MotiveWave tick export to Databento-compatible format.

## How to export from MotiveWave
1. Open MotiveWave, connect to Rithmic (LucidTrading gateway)
2. Load MNQ Continuous chart with Tick chart type
3. Pull history back to Jan 2022 (right-click chart → Request History)
4. Chart menu → Data → Export Data
5. Settings:
     Instrument : MNQ (continuous or individual contracts)
     Bar Size   : Tick (1 trade per row)
     Date Range : Jan 1 2022 → Nov 30 2025
     Format     : CSV
     Include    : Date/Time, Price, Volume, Bid Volume, Ask Volume
6. Save to: data/processed/motivewave_mnq_ticks_<start>_<end>.csv

## Output
data/processed/mnq_trades_motivewave_<start>_<end>.csv.gz

Compatible with build_microstructure_features.py (same columns as Databento files):
  ts_recv  — nanoseconds since epoch
  side     — 'B' (buy aggressor), 'A' (sell aggressor), 'N' (neutral)
  price    — integer (price × 1e9, matches Databento fixed-point)
  size     — float (number of contracts)

## Aggressor side from MotiveWave
MotiveWave CQG backfill provides bid_volume and ask_volume per tick.
  - ask_volume > 0 → trade at ask → buyer aggressor → side='B'
  - bid_volume > 0 → trade at bid → seller aggressor → side='A'
  - Neither            → neutral  → side='N'
If bid/ask volume columns absent, falls back to tick-rule direction (less accurate).
"""

import gzip
import logging
import os
import sys
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE     = Path(__file__).parents[2]
DATA_IN  = BASE / "data" / "processed"
DATA_OUT = BASE / "data" / "processed"

PRICE_SCALE = 1_000_000_000  # match Databento fixed-point

# RTH filter: 14:30-21:00 UTC = 9:30-16:00 ET
RTH_START_H, RTH_START_M = 14, 30
RTH_END_H = 21


def is_rth(dt_utc: pd.Series) -> pd.Series:
    h = dt_utc.dt.hour
    m = dt_utc.dt.minute
    return ((h > RTH_START_H) | ((h == RTH_START_H) & (m >= RTH_START_M))) & (h < RTH_END_H)


def detect_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect column mapping from MotiveWave export.
    Returns dict: {role: column_name}
    """
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}

    # Timestamp
    for cand in ["date time", "datetime", "date_time", "timestamp", "time", "date"]:
        if cand in cols:
            mapping["ts"] = cols[cand]
            break

    # Price
    for cand in ["price", "last", "close", "trade price", "trade_price"]:
        if cand in cols:
            mapping["price"] = cols[cand]
            break

    # Volume / size
    for cand in ["volume", "size", "qty", "quantity", "trade volume", "trade_volume"]:
        if cand in cols:
            mapping["size"] = cols[cand]
            break

    # Ask volume (buy aggressor) — trade lifted the ask
    for cand in ["ask volume", "ask_volume", "askvolume", "ask vol", "up volume", "up_volume"]:
        if cand in cols:
            mapping["ask_vol"] = cols[cand]
            break

    # Bid volume (sell aggressor) — trade hit the bid
    for cand in ["bid volume", "bid_volume", "bidvolume", "bid vol", "down volume", "down_volume"]:
        if cand in cols:
            mapping["bid_vol"] = cols[cand]
            break

    # Side column (some exports have explicit B/S column)
    for cand in ["side", "direction", "type", "buy/sell", "trade type"]:
        if cand in cols:
            mapping["side"] = cols[cand]
            break

    missing = [r for r in ("ts", "price", "size") if r not in mapping]
    if missing:
        raise ValueError(
            f"Could not detect required columns {missing} in MW export.\n"
            f"Available columns: {list(df.columns)}\n"
            "Check the export settings in MotiveWave."
        )

    return mapping


def infer_side(row_df: pd.DataFrame, mapping: dict) -> pd.Series:
    """Infer aggressor side from available columns."""

    # Explicit side column (B/S or Buy/Sell)
    if "side" in mapping:
        s = row_df[mapping["side"]].astype(str).str.strip().str.upper()
        result = s.map({"B": "B", "BUY": "B", "S": "A", "SELL": "A"}).fillna("N")
        known = (result != "N").mean()
        if known > 0.5:
            logger.info("Side from explicit column '%s' (%.0f%% classified)", mapping["side"], known * 100)
            return result

    # Ask/bid volume columns
    if "ask_vol" in mapping and "bid_vol" in mapping:
        ask = pd.to_numeric(row_df[mapping["ask_vol"]], errors="coerce").fillna(0)
        bid = pd.to_numeric(row_df[mapping["bid_vol"]], errors="coerce").fillna(0)
        result = pd.Series("N", index=row_df.index)
        result[ask > 0] = "B"   # lifted ask → buy aggressor
        result[bid > 0] = "A"   # hit bid  → sell aggressor
        classified = ((result == "B") | (result == "A")).mean()
        logger.info("Side from bid/ask volume columns (%.0f%% classified)", classified * 100)
        return result

    # Fallback: tick rule (price vs previous price)
    logger.warning(
        "No bid/ask volume or explicit side column found — using tick rule "
        "(less accurate, ~75-80%% precision). "
        "Re-export from MotiveWave with 'Bid Volume' and 'Ask Volume' included for best results."
    )
    price = pd.to_numeric(row_df[mapping["price"]], errors="coerce")
    dp    = price.diff()
    result = pd.Series("N", index=row_df.index)
    result[dp > 0] = "B"
    result[dp < 0] = "A"
    return result


def convert_file(in_path: Path, out_path: Path) -> int:
    """Convert one MotiveWave CSV export to gzip CSV in Databento format."""
    logger.info("Reading %s ...", in_path.name)

    # Try common separators and encodings
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(in_path, sep=sep, encoding="utf-8", low_memory=False)
            if len(df.columns) >= 3:
                break
        except Exception:
            continue
    else:
        raise ValueError(f"Could not parse {in_path.name} as CSV")

    logger.info("  Loaded %d rows, %d columns: %s", len(df), len(df.columns), list(df.columns))

    mapping = detect_columns(df)
    logger.info("  Column mapping: %s", mapping)

    # Parse timestamp
    ts_raw = df[mapping["ts"]]
    try:
        dt = pd.to_datetime(ts_raw, utc=False)
        # MotiveWave exports in exchange local time (CT) — convert to UTC
        if dt.dt.tz is None:
            # Assume Chicago time (UTC-6 standard, UTC-5 daylight)
            # Use a conservative approach: let pandas handle DST via US/Central
            dt = dt.dt.tz_localize("US/Central", ambiguous="NaT", nonexistent="NaT")
        dt = dt.dt.tz_convert("UTC")
    except Exception as e:
        raise ValueError(f"Could not parse timestamp column '{mapping['ts']}': {e}")

    # RTH filter
    rth_mask = is_rth(dt)
    df   = df[rth_mask].copy()
    dt   = dt[rth_mask]
    logger.info("  After RTH filter: %d rows", len(df))

    if len(df) == 0:
        logger.warning("  No RTH rows — check timestamp parsing / timezone handling")
        return 0

    # Price and size
    price = pd.to_numeric(df[mapping["price"]], errors="coerce")
    size  = pd.to_numeric(df[mapping["size"]],  errors="coerce")
    valid = price.notna() & size.notna() & (price > 0) & (size > 0)
    df    = df[valid]; dt = dt[valid]; price = price[valid]; size = size[valid]

    # Side
    side = infer_side(df, mapping)

    # Build output
    ts_ns     = (dt.astype("int64")).values   # already ns since epoch in UTC
    price_int = (price.values * PRICE_SCALE).astype("int64")

    out = pd.DataFrame({
        "ts_recv": ts_ns,
        "side":    side.values,
        "price":   price_int,
        "size":    size.values.astype(float),
    }).sort_values("ts_recv")

    out.to_csv(out_path, index=False, compression="gzip")
    logger.info("  Saved %d RTH ticks → %s (%.1f MB)",
                len(out), out_path.name, out_path.stat().st_size / 1e6)
    return len(out)


def main():
    # Find all MotiveWave CSV files in DATA_IN
    patterns = ["motivewave_*.csv", "motivewave_*.txt", "mw_*.csv", "MW_*.csv"]
    candidates = []
    for pat in patterns:
        candidates.extend(sorted(DATA_IN.glob(pat)))

    if not candidates:
        logger.error(
            "No MotiveWave export files found in %s\n"
            "Expected file names matching: motivewave_mnq_*.csv\n"
            "Export from MotiveWave and save to that directory.",
            DATA_IN,
        )
        sys.exit(1)

    total = 0
    converted = []
    for in_path in candidates:
        # Derive output filename
        stem = in_path.stem.replace("motivewave_", "").replace("mw_", "").replace("MW_", "")
        out_name = f"mnq_trades_mw_{stem}.csv.gz"
        out_path = DATA_OUT / out_name

        try:
            n = convert_file(in_path, out_path)
            total += n
            converted.append(out_path)
        except Exception as exc:
            logger.error("Failed to convert %s: %s", in_path.name, exc)

    if not converted:
        logger.error("No files converted successfully.")
        sys.exit(1)

    logger.info("\nConverted %d files, %d total RTH ticks", len(converted), total)
    logger.info("\nNext steps:")
    logger.info("1. Add converted files to FILES in build_microstructure_features.py:")
    for p in converted:
        logger.info("   DATA / '%s'", p.name)
    logger.info("2. Run: python ml_intraday_v3/scripts/build_microstructure_features.py")
    logger.info("3. Run: python ml_intraday_v3/scripts/barrier_grid_search.py")
    logger.info("4. Run: python ml_intraday_v3/scripts/ml_scalper_v5.py")


if __name__ == "__main__":
    main()
