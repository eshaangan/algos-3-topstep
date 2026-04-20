"""Fetch MNQ tick-level trades data from Databento and build true CDP per 5-min bar.

Pulls the `trades` schema (actual buy/sell-initiated volume per tick) to replace
the OHLCV proxy CDP. Budget target: ~$40 from today backwards as far as possible.

Cost reference (GLBX.MDP3, MNQ.c.0, trades schema):
  Jan 2 – Mar 19, 2026 (~55 trading days) = $68.91 → ~$1.25/trading day
  $40 budget ≈ 32 trading days → start ~Feb 10, 2026

Output:
  data/processed/mnq_trades_5min.h5   key=bars_5min   (OHLCV + cdp, buy_vol, sell_vol)
  data/processed/mnq_trades_raw.dbn   (raw tick data, ~400MB)

Usage:
  cd "algos 3 topstep"
  python rule_based_v1/diagnostics/fetch_trades_l2.py --cost-check
  python rule_based_v1/diagnostics/fetch_trades_l2.py
  python rule_based_v1/diagnostics/fetch_trades_l2.py --start 2026-02-10
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RBV1 = ROOT / "rule_based_v1"
for p in [str(ROOT), str(RBV1)]:
    if p not in sys.path:
        sys.path.insert(0, p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_PATH    = ROOT / "data" / "processed" / "mnq_trades_raw.dbn"
BARS_PATH   = ROOT / "data" / "processed" / "mnq_trades_5min.h5"

DEFAULT_START = "2026-02-10"   # ~32 trading days back ≈ $40
DEFAULT_END   = "2026-03-23"


# ---------------------------------------------------------------------------
# Databento client
# ---------------------------------------------------------------------------
def _get_client():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("DATABENTO_API_KEY")
    if not api_key:
        raise ValueError("DATABENTO_API_KEY not set in .env")
    import databento as db
    return db.Historical(key=api_key)


# ---------------------------------------------------------------------------
# Cost check
# ---------------------------------------------------------------------------
def check_cost(start: str, end: str) -> float:
    client = _get_client()
    cost = client.metadata.get_cost(
        dataset="GLBX.MDP3",
        symbols=["MNQ.c.0"],
        schema="trades",
        start=start,
        end=end,
        stype_in="continuous",
    )
    return cost


# ---------------------------------------------------------------------------
# Fetch and save raw .dbn
# ---------------------------------------------------------------------------
def fetch_raw(start: str, end: str) -> Path:
    client = _get_client()
    logger.info(f"Fetching trades schema: {start} → {end}")
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = client.timeseries.get_range(
        dataset="GLBX.MDP3",
        symbols=["MNQ.c.0"],
        schema="trades",
        start=start,
        end=end,
        stype_in="continuous",
    )
    data.to_file(str(RAW_PATH))
    logger.info(f"Raw trades saved → {RAW_PATH} ({RAW_PATH.stat().st_size / 1e6:.1f} MB)")
    return RAW_PATH


# ---------------------------------------------------------------------------
# Build 5-min bars with true CDP
# ---------------------------------------------------------------------------
def build_bars(raw_path: Path) -> pd.DataFrame:
    """Aggregate tick trades into 5-min bars with true cumulative delta."""
    import databento as db
    logger.info(f"Loading raw trades from {raw_path} ...")

    store = db.DBNStore.from_file(str(raw_path))
    df = store.to_df()

    logger.info(f"Loaded {len(df):,} ticks   {df.index[0]} → {df.index[-1]}")

    # Ensure UTC-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("US/Eastern")

    # RTH filter
    rth = (
        ((df.index.hour > 9) | ((df.index.hour == 9) & (df.index.minute >= 30)))
        & (df.index.hour < 16)
    )
    df = df.loc[rth]
    logger.info(f"After RTH filter: {len(df):,} ticks")

    # Determine buyer/seller initiated
    # Databento trades: side='A' = aggressor hit the ask → buyer initiated (buy)
    #                   side='B' = aggressor hit the bid → seller initiated (sell)
    # size is in lots (contracts) — cast to int64 BEFORE subtraction to avoid uint32 overflow
    size_int = df["size"].astype("int64")
    is_buy   = df["side"] == "A"
    df["buy_vol"]  = size_int.where(is_buy,  0)
    df["sell_vol"] = size_int.where(~is_buy, 0)

    # 5-min resample
    bars = df.resample("5min").agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("size", "sum"),
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
    ).dropna(subset=["open"])

    # True cumulative delta per bar
    bars["cdp"] = bars["buy_vol"] - bars["sell_vol"]

    # Also compute rolling cumulative delta (session-level running sum, reset each day)
    bars["cum_delta"] = bars.groupby(bars.index.date)["cdp"].cumsum()

    # Verify price scaling — Databento trades use fixed-point: divide by 1e9 for MNQ
    # If prices look like integers > 10000, they need scaling
    sample_close = bars["close"].dropna().iloc[0] if len(bars) > 0 else 0
    if sample_close > 100_000:
        logger.info(f"Scaling prices from fixed-point (sample close={sample_close:.0f})")
        for col in ["open", "high", "low", "close"]:
            bars[col] = bars[col] / 1e9
    else:
        logger.info(f"Prices look normal (sample close={sample_close:.2f})")

    logger.info(
        f"Built {len(bars):,} 5-min bars  "
        f"{bars.index[0].date()} → {bars.index[-1].date()}\n"
        f"  volume range: {bars['volume'].min():.0f} – {bars['volume'].max():.0f}\n"
        f"  cdp range:    {bars['cdp'].min():.0f} – {bars['cdp'].max():.0f}"
    )

    BARS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bars.to_hdf(str(BARS_PATH), key="bars_5min", mode="w", complevel=5)
    logger.info(f"Saved 5-min bars → {BARS_PATH}")
    return bars


# ---------------------------------------------------------------------------
# Quick diagnostics after build
# ---------------------------------------------------------------------------
def print_sample(bars: pd.DataFrame, n: int = 5) -> None:
    print("\n--- Sample 5-min bars with CDP ---")
    sample = bars[["open", "high", "low", "close", "volume", "buy_vol", "sell_vol", "cdp", "cum_delta"]]
    with pd.option_context("display.float_format", "{:,.2f}".format, "display.max_columns", 10):
        print(sample.tail(n).to_string())

    print("\n--- Daily signed volume summary ---")
    daily = bars.groupby(bars.index.date).agg(
        total_vol=("volume", "sum"),
        buy_vol=("buy_vol", "sum"),
        sell_vol=("sell_vol", "sum"),
        net_delta=("cdp", "sum"),
        close=("close", "last"),
    )
    daily["delta_pct"] = daily["net_delta"] / daily["total_vol"] * 100
    with pd.option_context("display.float_format", "{:,.1f}".format, "display.max_columns", 10):
        print(daily.tail(10).to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=DEFAULT_END,   help="End date YYYY-MM-DD")
    parser.add_argument("--cost-check", action="store_true", help="Print cost estimate only, do not fetch")
    parser.add_argument("--build-only", action="store_true", help="Skip fetch, rebuild bars from cached .dbn")
    args = parser.parse_args()

    if args.cost_check:
        logger.info(f"Checking cost for trades schema: {args.start} → {args.end}")
        cost = check_cost(args.start, args.end)
        print(f"\n  Estimated cost: ${cost:.2f}")
        print(f"  Date range:     {args.start} → {args.end}")
        return

    if args.build_only:
        if not RAW_PATH.exists():
            logger.error(f"No raw data at {RAW_PATH}. Run without --build-only first.")
            sys.exit(1)
        bars = build_bars(RAW_PATH)
        print_sample(bars)
        return

    # Full flow: cost check → fetch → build
    logger.info(f"Checking cost for {args.start} → {args.end} ...")
    cost = check_cost(args.start, args.end)
    print(f"\n  Estimated cost: ${cost:.2f}")

    if cost > 50.0:
        logger.warning(f"Cost ${cost:.2f} exceeds $50 budget. Tightening date range.")
        # Adjust start forward by ~2 weeks to stay under budget
        adjusted_start = pd.Timestamp(args.start) + pd.Timedelta(days=14)
        args.start = adjusted_start.strftime("%Y-%m-%d")
        cost2 = check_cost(args.start, args.end)
        print(f"  Adjusted start: {args.start}  Cost: ${cost2:.2f}")
        if cost2 > 50.0:
            logger.error(f"Still over budget (${cost2:.2f}). Aborting. Use --start to set a later date.")
            sys.exit(1)
        cost = cost2

    logger.info(f"Proceeding with fetch: {args.start} → {args.end}  (cost: ${cost:.2f})")
    raw_path = fetch_raw(args.start, args.end)
    bars = build_bars(raw_path)
    print_sample(bars)

    print(f"\n  Data saved to {BARS_PATH}")
    print(f"  Use key='bars_5min' to load. Columns: open/high/low/close/volume/buy_vol/sell_vol/cdp/cum_delta")
    print(f"  True CDP is 'cdp' column: buyer-initiated minus seller-initiated volume per 5-min bar")


if __name__ == "__main__":
    main()
