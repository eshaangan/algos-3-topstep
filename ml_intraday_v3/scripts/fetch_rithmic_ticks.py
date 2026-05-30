"""
Download MNQ historical tick data via Rithmic R|Protocol (async_rithmic).

Pulls up to ~1 year of tick-by-tick data (price, size, aggressor side) from
Rithmic's HISTORY_PLANT and saves as gzip-compressed CSV in the same format as
the existing Databento files (ts_recv, side, price, size) so the existing
build_microstructure_features.py pipeline can consume them unchanged.

Usage:
    python ml_intraday_v3/scripts/fetch_rithmic_ticks.py

Requires in .env:
    RITHMIC_USERNAME
    RITHMIC_PASSWORD
    RITHMIC_SYSTEM_NAME   (e.g. "Lucid Trading")
    RITHMIC_GATEWAY_URI   (e.g. "rituz00100.rithmic.com:443")
    RITHMIC_APP_NAME      (e.g. "XXXX:mnq_ml_trader")

Output:
    data/processed/mnq_trades_rithmic_<start>_<end>.csv.gz
"""

import asyncio
import gzip
import logging
import os
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE      = Path(__file__).parents[2]
DATA_OUT  = BASE / "data" / "processed"
DATA_OUT.mkdir(parents=True, exist_ok=True)

SYMBOL    = "MNQ"
EXCHANGE  = "CME"

# RTH filter — same as Databento files (14:30-21:00 UTC = 9:30-16:00 ET)
RTH_START_H, RTH_START_M = 14, 30
RTH_END_H = 21


def load_env():
    env_path = BASE / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.split("#")[0].strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip().strip('"').strip("'")
                    os.environ.setdefault(k.strip(), v)


def is_rth(dt: datetime) -> bool:
    h, m = dt.hour, dt.minute
    after_open  = (h > RTH_START_H) or (h == RTH_START_H and m >= RTH_START_M)
    before_close = h < RTH_END_H
    return after_open and before_close


async def fetch_ticks(start: datetime, end: datetime) -> list[dict]:
    """Fetch ticks for a single time window via Rithmic HISTORY_PLANT."""
    from async_rithmic import RithmicClient
    from async_rithmic.enums import OrderPlacement, SysInfraType

    client = RithmicClient(
        user=os.environ["RITHMIC_USERNAME"],
        password=os.environ["RITHMIC_PASSWORD"],
        system_name=os.environ.get("RITHMIC_SYSTEM_NAME", "Lucid Trading"),
        app_name=os.environ.get("RITHMIC_APP_NAME", "XXXX:mnq_ml_trader"),
        app_version="1.0.0",
        url=os.environ["RITHMIC_GATEWAY_URI"],
        manual_or_auto=OrderPlacement.AUTO,
    )

    # Only connect the plants we need (history + ticker for front-month resolution)
    await client.connect(plants=[SysInfraType.TICKER_PLANT, SysInfraType.HISTORY_PLANT])

    try:
        contract = await client.get_front_month_contract(SYMBOL, EXCHANGE)
        logger.info("Front-month contract: %s", contract)

        ticks_collected = []

        async def on_tick(data: dict):
            dt = data.get("datetime")
            if dt and is_rth(dt):
                ticks_collected.append(data)

        client.on_historical_tick += on_tick

        logger.info("Requesting tick replay %s → %s ...", start.date(), end.date())
        result = await client.get_historical_tick_data(
            symbol=contract,
            exchange=EXCHANGE,
            start_time=start,
            end_time=end,
            wait=True,
            idle_timeout=60.0,
            max_pages=10_000,
        )

        if result:
            for data in result:
                dt = data.get("datetime")
                if dt and is_rth(dt):
                    ticks_collected.append(data)

        logger.info("Collected %d RTH ticks", len(ticks_collected))
        return ticks_collected

    finally:
        await client.disconnect()


def rithmic_side_to_databento(side_value) -> str:
    """
    Map Rithmic aggressor side to Databento convention.
    Rithmic: 1=BUY aggressor, 2=SELL aggressor, 0=no aggressor
    Databento: 'B'=bid-side aggressor (buy), 'A'=ask-side aggressor (sell), 'N'=neutral
    """
    if side_value == 1:
        return "B"
    if side_value == 2:
        return "A"
    return "N"


def save_ticks(ticks: list[dict], out_path: Path) -> int:
    """Save ticks to gzip CSV in Databento-compatible format."""
    if not ticks:
        logger.warning("No ticks to save.")
        return 0

    PRICE_SCALE = 1_000_000_000  # match existing Databento files

    written = 0
    with gzip.open(out_path, "wt", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts_recv", "side", "price", "size"])

        for tick in sorted(ticks, key=lambda t: t.get("datetime", datetime.min)):
            dt: datetime = tick.get("datetime")
            price_f: float = tick.get("trade_price") or tick.get("price") or 0.0
            size: int = int(tick.get("trade_size") or tick.get("size") or 0)
            side_raw = tick.get("aggressor") or tick.get("transaction_type") or 0
            side_str = rithmic_side_to_databento(side_raw)

            if dt is None or price_f == 0 or size == 0:
                continue

            # ts_recv: nanoseconds since epoch (match Databento format)
            ts_ns = int(dt.timestamp() * 1e9)
            price_int = int(round(price_f * PRICE_SCALE))

            writer.writerow([ts_ns, side_str, price_int, size])
            written += 1

    logger.info("Saved %d ticks → %s (%.1f MB)", written, out_path.name,
                out_path.stat().st_size / 1e6)
    return written


async def main_async():
    load_env()

    missing = [v for v in ["RITHMIC_USERNAME", "RITHMIC_PASSWORD", "RITHMIC_GATEWAY_URI"]
               if not os.environ.get(v)]
    if missing:
        logger.error("Missing env vars: %s", missing)
        logger.error("Add them to .env and re-run.")
        return

    # Pull in 30-day chunks to avoid Rithmic's per-request limits
    # Adjust end_date to yesterday; start = 1 year back
    end_date   = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=365)

    logger.info("Fetching MNQ ticks from %s to %s (1 year)", start_date.date(), end_date.date())

    all_ticks: list[dict] = []
    chunk_start = start_date
    chunk_days  = 30

    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end_date)
        logger.info("Chunk: %s → %s", chunk_start.date(), chunk_end.date())

        try:
            ticks = await fetch_ticks(chunk_start, chunk_end)
            all_ticks.extend(ticks)
            logger.info("  Got %d ticks (total so far: %d)", len(ticks), len(all_ticks))
        except Exception as exc:
            logger.error("Chunk failed: %s — skipping", exc)

        chunk_start = chunk_end

    if not all_ticks:
        logger.error("No ticks collected. Check credentials and gateway URI.")
        return

    start_str = start_date.strftime("%Y%m%d")
    end_str   = (end_date - timedelta(days=1)).strftime("%Y%m%d")
    out_path  = DATA_OUT / f"mnq_trades_rithmic_{start_str}_{end_str}.csv.gz"

    saved = save_ticks(all_ticks, out_path)
    logger.info("Done. %d total RTH ticks → %s", saved, out_path)
    logger.info("")
    logger.info("Next step: add this file to FILES in build_microstructure_features.py")
    logger.info("  FILES = [ ..., DATA / '%s' ]", out_path.name)


if __name__ == "__main__":
    asyncio.run(main_async())
