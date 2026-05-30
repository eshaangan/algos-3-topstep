"""
Rithmic Historical Data Fetcher — free replacement for Databento subscriptions.

Two modes:
  ohlcv  — pull OHLCV time bars for any date range, save to HDF5.
            Handles quarterly MNQ contract rollover automatically.
            Typically goes back 2-3 years.

  ticks  — pull tick-bar replay → reconstruct full 39-column microstructure
            parquet (identical to the live system's feature output).
            Typically goes back 3-6 months (Rithmic/Lucid limit).

Usage examples:
  # Update microstructure parquet with last 30 days of tick data
  python scripts/rithmic_history_fetcher.py --mode ticks --days 30

  # Pull 1 year of 5-min OHLCV bars for ORB backtesting
  python scripts/rithmic_history_fetcher.py --mode ohlcv --days 365 --bar-min 5

  # Pull ticks for a specific date range
  python scripts/rithmic_history_fetcher.py --mode ticks --start 2026-04-01 --end 2026-05-28

  # Dry-run to see what would be pulled (no writes)
  python scripts/rithmic_history_fetcher.py --mode ticks --days 7 --dry-run

Environment variables (required):
  RITHMIC_USERNAME, RITHMIC_PASSWORD, RITHMIC_SYSTEM_NAME, RITHMIC_GATEWAY_URI

Output files:
  data/processed/mnq_microstructure_5min.parquet  (ticks mode — appended/merged)
  data/processed/mnq_ohlcv_{bar_min}min_rithmic.h5  (ohlcv mode)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from async_rithmic import RithmicClient
from async_rithmic.enums import TimeBarType, OrderPlacement
from core.microstructure import compute_bar_features, compute_vpin

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rithmic_fetcher")

EXCHANGE = "CME"
SYMBOL   = "MNQ"

# MNQ quarterly contract rollover schedule.
# Key = first trading day of new front month (roll happens ~8 biz days before expiry).
# Values are the new front-month contract codes.
# Update this table each quarter.
ROLL_SCHEDULE: list[tuple[datetime, str]] = [
    (datetime(2025,  6, 10, tzinfo=timezone.utc), "MNQU5"),   # Sep 2025
    (datetime(2025,  9,  9, tzinfo=timezone.utc), "MNQZ5"),   # Dec 2025
    (datetime(2025, 12,  9, tzinfo=timezone.utc), "MNQH6"),   # Mar 2026
    (datetime(2026,  3, 10, tzinfo=timezone.utc), "MNQM6"),   # Jun 2026
    (datetime(2026,  6, 10, tzinfo=timezone.utc), "MNQU6"),   # Sep 2026
    (datetime(2026,  9,  8, tzinfo=timezone.utc), "MNQZ6"),   # Dec 2026
]

MICRO_PARQUET = ROOT / "data" / "processed" / "mnq_microstructure_5min.parquet"
OHLCV_TEMPLATE = ROOT / "data" / "processed" / "mnq_ohlcv_{bar_min}min_rithmic.h5"


# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------

def contract_for_date(dt: datetime) -> str:
    """Return the front-month MNQ contract code active on a given UTC datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    code = ROLL_SCHEDULE[0][1]
    for roll_dt, contract in ROLL_SCHEDULE:
        if dt >= roll_dt:
            code = contract
    return code


def date_ranges_by_contract(
    start: datetime, end: datetime
) -> list[tuple[str, datetime, datetime]]:
    """
    Split [start, end] into sub-ranges, one per contract, to handle rollovers.

    Returns list of (contract_code, range_start, range_end).
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    # Build sorted list of rollover boundaries that fall within our range
    boundaries = [(dt, code) for dt, code in ROLL_SCHEDULE if start < dt < end]
    boundaries.sort(key=lambda x: x[0])

    ranges = []
    cursor = start
    for roll_dt, new_code in boundaries:
        old_code = contract_for_date(cursor)
        ranges.append((old_code, cursor, roll_dt))
        cursor = roll_dt

    ranges.append((contract_for_date(cursor), cursor, end))
    return ranges


# ---------------------------------------------------------------------------
# Rithmic connection (shared async context)
# ---------------------------------------------------------------------------

async def _make_client() -> RithmicClient:
    user    = os.environ["RITHMIC_USERNAME"]
    pw      = os.environ["RITHMIC_PASSWORD"]
    system  = os.getenv("RITHMIC_SYSTEM_NAME", "LucidTrading")
    app     = os.getenv("RITHMIC_APP_NAME",    "XXXX:history_fetcher")
    gateway = os.getenv("RITHMIC_GATEWAY_URI", "rprotocol.rithmic.com:443")

    client = RithmicClient(
        user=user,
        password=pw,
        system_name=system,
        app_name=app,
        app_version="1.0.0",
        url=gateway,
        manual_or_auto=OrderPlacement.AUTO,
    )
    await client.connect()
    logger.info("Connected — accounts: %s", client.accounts)
    return client


# ---------------------------------------------------------------------------
# OHLCV mode
# ---------------------------------------------------------------------------

async def fetch_ohlcv(
    start: datetime,
    end: datetime,
    bar_min: int,
    dry_run: bool,
) -> pd.DataFrame:
    """Pull OHLCV time bars from Rithmic, returns combined DataFrame."""
    all_bars: list[pd.Series] = []

    if dry_run:
        for contract, seg_start, seg_end in date_ranges_by_contract(start, end):
            logger.info("[dry-run] would call get_historical_time_bars(%s, %s → %s)",
                        contract, seg_start.date(), seg_end.date())
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    client = await _make_client()

    for contract, seg_start, seg_end in date_ranges_by_contract(start, end):
        logger.info("Pulling %s OHLCV [%s → %s] contract=%s",
                    f"{bar_min}-min", seg_start.date(), seg_end.date(), contract)

        try:
            bars = await client.get_historical_time_bars(
                contract, EXCHANGE,
                start_time=seg_start,
                end_time=seg_end,
                bar_type=TimeBarType.MINUTE_BAR,
                bar_type_periods=bar_min,
                idle_timeout=30.0,
                max_pages=2000,
            )
        except Exception as exc:
            logger.error("get_historical_time_bars failed for %s: %s", contract, exc)
            continue

        logger.info("  Received %d bars for %s", len(bars or []), contract)
        for b in (bars or []):
            end_dt = b.get("bar_end_datetime")
            if end_dt is None:
                marker = b.get("marker")
                if marker is None:
                    continue
                end_dt = datetime.fromtimestamp(int(marker), tz=timezone.utc)
            if isinstance(end_dt, datetime) and end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc)
            bar_ts = pd.Timestamp(end_dt, tz="UTC") - pd.Timedelta(minutes=bar_min)
            row = pd.Series({
                "open":   float(b.get("open_price",  b.get("open",  0))),
                "high":   float(b.get("high_price",  b.get("high",  0))),
                "low":    float(b.get("low_price",   b.get("low",   0))),
                "close":  float(b.get("close_price", b.get("close", 0))),
                "volume": float(b.get("volume", 0)),
            }, name=bar_ts)
            all_bars.append(row)

    await client.disconnect()

    if not all_bars:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    df = pd.DataFrame(all_bars)
    df.index.name = "ts"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    logger.info("OHLCV total: %d bars, %s → %s", len(df), df.index[0].date(), df.index[-1].date())
    return df


def save_ohlcv(df: pd.DataFrame, bar_min: int, dry_run: bool) -> Path:
    out = OHLCV_TEMPLATE.with_name(OHLCV_TEMPLATE.name.format(bar_min=bar_min))
    if dry_run:
        logger.info("[dry-run] would write %d bars → %s", len(df), out)
        return out

    # Merge with existing file if present
    if out.exists():
        try:
            existing = pd.read_hdf(out, key="bars_5min")
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            df = combined
            logger.info("Merged with existing: %d bars total", len(df))
        except Exception as exc:
            logger.warning("Could not read existing HDF5, overwriting: %s", exc)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_hdf(out, key="bars_5min", mode="w", complevel=5, complib="blosc")
    logger.info("Saved %d OHLCV bars → %s", len(df), out)
    return out


# ---------------------------------------------------------------------------
# Tick mode — fetch ticks → microstructure features
# ---------------------------------------------------------------------------

def _ticks_to_df(raw_ticks: list[dict]) -> pd.DataFrame:
    """
    Convert async_rithmic tick bar replay dicts to a raw ticks DataFrame.

    Rithmic 1-tick bars:
      data_bar_ssboe[0]  — epoch seconds
      data_bar_usecs[0]  — microseconds
      open_price         — trade price (== close_price for 1-tick bar)
      volume             — number of contracts
      bid_volume         — > 0 if buyer-initiated (aggressor = BUY)
      ask_volume         — > 0 if seller-initiated (aggressor = SELL)
    """
    rows = []
    for t in raw_ticks:
        ssboe_list = t.get("data_bar_ssboe", [])
        usecs_list = t.get("data_bar_usecs", [])
        if not ssboe_list:
            continue
        ssboe = int(ssboe_list[0])
        usecs = int(usecs_list[0]) if usecs_list else 0
        ts_ns = ssboe * 1_000_000_000 + usecs * 1_000

        price = float(t.get("open_price") or t.get("close_price") or 0)
        vol   = float(t.get("volume") or 0)
        bid_v = float(t.get("bid_volume") or 0)
        ask_v = float(t.get("ask_volume") or 0)

        if price <= 0 or vol <= 0:
            continue

        # Determine aggressor from bid/ask volume split
        if bid_v > 0 and ask_v == 0:
            side = "B"
        elif ask_v > 0 and bid_v == 0:
            side = "A"
        elif bid_v >= ask_v:
            side = "B"
        elif ask_v > bid_v:
            side = "A"
        else:
            side = "N"

        rows.append({
            "ts_recv": ts_ns,
            "price_f": price,
            "size":    vol,
            "side":    side,
        })

    if not rows:
        return pd.DataFrame(columns=["ts_recv", "price_f", "size", "side"])

    df = pd.DataFrame(rows).sort_values("ts_recv").reset_index(drop=True)
    return df


def _ticks_to_bars(ticks: pd.DataFrame, bar_min: int = 5) -> pd.DataFrame:
    """
    Aggregate per-tick DataFrame into bar-level microstructure features.
    Applies compute_bar_features() per 5-min bar, then adds rolling features.
    Returns DataFrame with the same 39-column schema as mnq_microstructure_5min.parquet.
    """
    ticks = ticks.copy()
    ticks["bar_ts"] = (
        pd.to_datetime(ticks["ts_recv"], unit="ns", utc=True)
          .dt.floor(f"{bar_min}min")
    )
    ticks["min_ts"] = (
        pd.to_datetime(ticks["ts_recv"], unit="ns", utc=True)
          .dt.floor("1min")
    )

    # Per-bar OFI on the first and last minute (for ofi_1m_first / ofi_1m_last)
    min_ofi = (
        ticks.assign(signed=ticks.apply(
            lambda r: r["size"] if r["side"] == "B"
                      else (-r["size"] if r["side"] == "A" else 0), axis=1
        ))
        .groupby("min_ts")["signed"].sum()
    )

    records = {}
    for bar_ts, grp in ticks.groupby("bar_ts"):
        feats = compute_bar_features(grp[["ts_recv", "price_f", "size", "side"]])
        if not feats:
            continue

        # OHLCV from ticks (already inside feats as open/high/low/close/vwap)
        bar_end_min = bar_ts + pd.Timedelta(minutes=bar_min - 1)
        feats["ofi_1m_first"] = float(min_ofi.get(bar_ts,       0.0))
        feats["ofi_1m_last"]  = float(min_ofi.get(bar_end_min,  0.0))
        records[bar_ts] = feats

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(records, orient="index")
    df.index.name = "ts"
    df = df.sort_index()

    # Rolling microstructure (needs enough history for VPIN-20)
    buy_s  = df["buy_vol"]
    sell_s = df["sell_vol"]
    vpin_d = compute_vpin(buy_s, sell_s, windows=(5, 20))
    df["vpin_5"]  = vpin_d["vpin_5"]
    df["vpin_20"] = vpin_d["vpin_20"]

    # ofi_15m = 3-bar lagged rolling OFI sum (3 × 5-min = 15-min lookback, lag-1 safe)
    df["ofi_15m"] = df["ofi"].shift(1).rolling(3, min_periods=3).sum()

    return df


async def _fetch_chunk(
    idx: int,
    total: int,
    contract: str,
    chunk_start: datetime,
    chunk_end: datetime,
    semaphore: asyncio.Semaphore,
) -> pd.DataFrame:
    """Fetch one tick chunk with its own client connection, bounded by semaphore."""
    async with semaphore:
        label = f"Chunk {idx+1}/{total} {contract} [{chunk_start.strftime('%m-%d')} → {chunk_end.strftime('%m-%d')}]"
        logger.info("START  %s", label)
        try:
            client = await _make_client()
        except Exception as exc:
            logger.error("CONNECT FAIL  %s: %s", label, exc)
            return pd.DataFrame()

        try:
            raw = await client.get_historical_tick_data(
                contract, EXCHANGE,
                start_time=chunk_start,
                end_time=chunk_end,
                idle_timeout=30.0,
                max_pages=5000,
            )
        except Exception as exc:
            logger.error("FETCH FAIL  %s: %s", label, exc)
            return pd.DataFrame()
        finally:
            try:
                await client.disconnect()
            except Exception:
                pass

        n_raw = len(raw or [])
        if n_raw == 0:
            logger.warning("EMPTY  %s (weekend/holiday or beyond history depth)", label)
            return pd.DataFrame()

        ticks_df = _ticks_to_df(raw)
        logger.info("DONE   %s — %d ticks  B=%d A=%d",
                    label, len(ticks_df),
                    (ticks_df["side"] == "B").sum(),
                    (ticks_df["side"] == "A").sum())
        return ticks_df


async def fetch_ticks(
    start: datetime,
    end: datetime,
    bar_min: int,
    dry_run: bool,
    chunk_days: int = 2,
    parallel: int = 4,
) -> pd.DataFrame:
    """
    Pull tick replay from Rithmic in parallel chunks, aggregate to microstructure bars.

    parallel: max concurrent Rithmic connections (default 4).
    chunk_days: days per chunk (default 2).
    """
    # Build chunks
    chunks: list[tuple[str, datetime, datetime]] = []
    for contract, seg_start, seg_end in date_ranges_by_contract(start, end):
        cursor = seg_start
        while cursor < seg_end:
            chunk_end = min(cursor + timedelta(days=chunk_days), seg_end)
            chunks.append((contract, cursor, chunk_end))
            cursor = chunk_end

    if dry_run:
        for i, (contract, cs, ce) in enumerate(chunks):
            logger.info("[dry-run] chunk %d/%d: %s [%s → %s]",
                        i+1, len(chunks), contract, cs.date(), ce.date())
        return pd.DataFrame()

    logger.info("Fetching %d chunks (%d-day each) on a single persistent connection",
                len(chunks), chunk_days)

    # One client for all chunks — avoids 9× connect/disconnect overhead and
    # eliminates ForcedLogout storms caused by multiple simultaneous sessions.
    client = await _make_client()
    all_ticks: list[pd.DataFrame] = []

    try:
        for i, (contract, cs, ce) in enumerate(chunks):
            label = f"Chunk {i+1}/{len(chunks)} {contract} [{cs.strftime('%m-%d')} → {ce.strftime('%m-%d')}]"
            logger.info("START  %s", label)
            try:
                raw = await client.get_historical_tick_data(
                    contract, EXCHANGE,
                    start_time=cs,
                    end_time=ce,
                    idle_timeout=30.0,
                    max_pages=5000,
                )
            except Exception as exc:
                logger.error("FETCH FAIL  %s: %s", label, exc)
                continue

            n_raw = len(raw or [])
            if n_raw == 0:
                logger.warning("EMPTY  %s (weekend/holiday or beyond history depth)", label)
                continue

            ticks_df = _ticks_to_df(raw)
            logger.info("DONE   %s — %d ticks  B=%d A=%d",
                        label, len(ticks_df),
                        (ticks_df["side"] == "B").sum(),
                        (ticks_df["side"] == "A").sum())
            all_ticks.append(ticks_df)

            # Checkpoint save after each successful chunk
            partial_ticks = pd.concat(all_ticks, ignore_index=True).sort_values("ts_recv")
            partial_bars = _ticks_to_bars(partial_ticks, bar_min=bar_min)
            if not partial_bars.empty:
                save_ticks(partial_bars, dry_run=False)
                logger.info("Checkpoint saved — %d bars through %s",
                            len(partial_bars), partial_bars.index[-1].date())
    finally:
        try:
            await client.disconnect()
        except Exception:
            pass

    if not all_ticks:
        logger.warning("No tick data received. Possible causes:\n"
                       "  (1) Date range exceeds Rithmic history depth (~3-6 months)\n"
                       "  (2) Contract code mismatch — edit ROLL_SCHEDULE in this script\n"
                       "  (3) Auth / permission issue")
        return pd.DataFrame()

    combined_ticks = pd.concat(all_ticks, ignore_index=True)
    combined_ticks = combined_ticks.sort_values("ts_recv").reset_index(drop=True)
    logger.info("Total ticks: %d, aggregating to %d-min bars…",
                len(combined_ticks), bar_min)

    bars = _ticks_to_bars(combined_ticks, bar_min=bar_min)
    logger.info("Bars built: %d, %s → %s",
                len(bars), bars.index[0].date(), bars.index[-1].date())
    return bars


def save_ticks(bars: pd.DataFrame, dry_run: bool) -> Path:
    out = MICRO_PARQUET
    if dry_run:
        logger.info("[dry-run] would write %d bars → %s", len(bars), out)
        return out

    # Merge with existing parquet
    if out.exists():
        try:
            existing = pd.read_parquet(out)
            existing.index = pd.to_datetime(existing.index, utc=True)
            bars.index     = pd.to_datetime(bars.index,     utc=True)
            combined = pd.concat([existing, bars])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
            # Re-compute VPIN and ofi_15m on the merged series for consistency
            vpin_d = compute_vpin(combined["buy_vol"], combined["sell_vol"], windows=(5, 20))
            combined["vpin_5"]  = vpin_d["vpin_5"]
            combined["vpin_20"] = vpin_d["vpin_20"]
            combined["ofi_15m"] = combined["ofi"].shift(1).rolling(3, min_periods=3).sum()
            bars = combined
            logger.info("Merged with existing parquet: %d bars total, %s → %s",
                        len(bars), bars.index[0].date(), bars.index[-1].date())
        except Exception as exc:
            logger.warning("Could not merge existing parquet, overwriting: %s", exc)

    out.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out)
    logger.info("Saved %d microstructure bars → %s", len(bars), out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["ohlcv", "ticks"], required=True,
                   help="ohlcv=OHLCV time bars (years of history), ticks=microstructure (months)")
    p.add_argument("--days",  type=int, default=None,
                   help="Number of days back from today (alternative to --start/--end)")
    p.add_argument("--start", default=None,
                   help="Start date YYYY-MM-DD (UTC). Overrides --days.")
    p.add_argument("--end",   default=None,
                   help="End date YYYY-MM-DD (UTC). Defaults to today.")
    p.add_argument("--bar-min", type=int, default=5,
                   help="Bar size in minutes (default: 5)")
    p.add_argument("--chunk-days", type=int, default=2,
                   help="Days per tick request chunk (default: 2). Smaller = less likely to drop.")
    p.add_argument("--parallel", type=int, default=4,
                   help="Max concurrent Rithmic connections for tick chunks (default: 4)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be fetched without connecting to Rithmic")
    return p.parse_args()


async def _main() -> None:
    args = _parse_args()

    # Resolve date range
    end_dt   = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    if args.end:
        end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) + timedelta(days=1)

    if args.start:
        start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    elif args.days:
        start_dt = end_dt - timedelta(days=args.days)
    else:
        start_dt = end_dt - timedelta(days=30)

    logger.info("Mode=%s  range=[%s, %s]  bar=%d-min  dry_run=%s",
                args.mode, start_dt.date(), end_dt.date(), args.bar_min, args.dry_run)

    # Show contract plan
    ranges = date_ranges_by_contract(start_dt, end_dt)
    for contract, s, e in ranges:
        logger.info("  Contract segment: %s  [%s → %s]", contract, s.date(), e.date())

    if args.mode == "ohlcv":
        df = await fetch_ohlcv(start_dt, end_dt, args.bar_min, args.dry_run)
        if not df.empty:
            save_ohlcv(df, args.bar_min, args.dry_run)

    elif args.mode == "ticks":
        df = await fetch_ticks(start_dt, end_dt, args.bar_min, args.dry_run,
                               chunk_days=args.chunk_days, parallel=args.parallel)
        if not df.empty:
            save_ticks(df, args.dry_run)


if __name__ == "__main__":
    asyncio.run(_main())
