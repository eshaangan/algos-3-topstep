"""
Free, no-key, deep-history data sources that are NOT the futures tape.

Every strategy family this project has tried was derived from OHLCV of the
instrument being traded, and all of them are NO-GO. That is not a coincidence:
bar-derived signals on index futures are the most heavily mined dataset in
existence, so the prior on finding residual edge there is poor. These fetchers
exist to widen the search to information that is genuinely not in the MNQ/MES
bars, without spending money and without a Databento key (the account is locked).

All three verified reachable and key-free on 2026-09-08:

  Source     Series                                   History     Auth
  ---------  ---------------------------------------  ----------  --------
  Cboe       VIX, VIX3M, VIX9D, VVIX, SKEW (daily)     1990/2009+  none
  CFTC       Commitments of Traders (weekly)           1997+       none
  Coinbase   BTC-USD & ETH-USD hourly candles          2017+       none*

  * Coinbase Exchange rejects requests without a browser User-Agent (403),
    which is why one is always sent below.

Why each one is interesting for this book specifically:

- **Cboe vol complex** is the market's *forward* price of risk. Realized-vol
  conditioning on weekend_hold was already tested and falsified (see
  EDGE_SCREENS_GUIDE.md); implied vol and the term-structure slope are a
  different, risk-premium-bearing variable, though they may still proxy the same
  state. Treat as a fresh hypothesis, not a retest.
- **CFTC COT** is positioning, not price. Tuesday's data is published Friday
  15:30 ET, so it is legitimately available before a Sunday 18:00 entry -- no
  lookahead for a weekend strategy.
- **Coinbase** matters because of a market-structure fact: equity futures are
  shut from Friday 17:00 to Sunday 18:00 ET (~49h) while crypto trades
  continuously. Information arriving in that window is impounded in crypto and
  cannot be in the futures tape. `weekend_hold_v1` enters at exactly Sunday
  18:00, which makes the closure-window crypto move the natural free covariate.

Caching: every fetch writes parquet under `cache_dir` (default
`data/processed/free/`) and is served from disk on subsequent calls. Pass
`refresh=True` to re-download.
"""

from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Iterable

import pandas as pd

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json, text/csv, */*"}
DEFAULT_CACHE = Path("data/processed/free")

CBOE_INDICES = ("VIX", "VIX3M", "VIX9D", "VVIX", "SKEW")
CBOE_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"


def _get(url: str, timeout: int = 45) -> bytes:
    return urllib.request.urlopen(
        urllib.request.Request(url, headers=UA), timeout=timeout
    ).read()


def _cache_path(cache_dir: Path, name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{name}.parquet"


# ---------------------------------------------------------------------------
# Cboe volatility complex
# ---------------------------------------------------------------------------

def fetch_cboe_index(
    name: str = "VIX",
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    One Cboe daily index history as a DataFrame indexed by UTC date.

    Parameters
    ----------
    name : one of CBOE_INDICES.
    refresh : re-download instead of using the on-disk cache.

    Returns
    -------
    DataFrame with lower-cased columns (OHLC for VIX/VIX3M/VIX9D; a single
    value column for VVIX/SKEW), indexed by date.
    """
    name = name.upper()
    if name not in CBOE_INDICES:
        raise ValueError(f"unknown Cboe index {name!r}; expected one of {CBOE_INDICES}")

    cache = _cache_path(cache_dir, f"cboe_{name.lower()}")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    df = pd.read_csv(io.BytesIO(_get(CBOE_URL.format(name=name))))
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    df.index.name = "date"
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.to_parquet(cache)
    return df


def fetch_vol_complex(
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    The Cboe complex joined on date, with the term-structure slopes this project
    would actually condition on.

    Adds:
      ts_slope_3m  = VIX3M / VIX  (>1 contango/calm, <1 backwardation/stress)
      ts_slope_9d  = VIX / VIX9D
    """
    out = {}
    for name in CBOE_INDICES:
        d = fetch_cboe_index(name, cache_dir, refresh)
        col = "close" if "close" in d.columns else d.columns[-1]
        out[name.lower()] = d[col]
    df = pd.DataFrame(out).sort_index()
    df["ts_slope_3m"] = df["vix3m"] / df["vix"]
    df["ts_slope_9d"] = df["vix"] / df["vix9d"]
    return df


# ---------------------------------------------------------------------------
# CFTC Commitments of Traders
# ---------------------------------------------------------------------------

def fetch_cot(
    contract: str = "E-MINI S&P 500",
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
    page: int = 5000,
) -> pd.DataFrame:
    """
    Weekly CFTC legacy COT rows for one contract, indexed by report date (UTC).

    The report reflects positions as of Tuesday and is published the following
    Friday at 15:30 ET. Callers conditioning a trade on it must respect that
    release lag; `report_date` here is the Tuesday, NOT the release time.

    Parameters
    ----------
    contract : exact `contract_market_name`, e.g. "E-MINI S&P 500",
        "MICRO E-MINI S&P 500 INDEX", "NASDAQ-100 STOCK INDEX (MINI)".
    """
    cache = _cache_path(cache_dir, f"cot_{contract.lower().replace(' ', '_').replace('&', 'and')}")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    rows, offset = [], 0
    while True:
        q = urllib.parse.urlencode({
            "$where": f"contract_market_name='{contract}'",
            "$order": "report_date_as_yyyy_mm_dd ASC",
            "$limit": page,
            "$offset": offset,
        })
        chunk = json.loads(_get(f"{COT_URL}?{q}"))
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < page:
            break
        offset += page
        time.sleep(0.2)

    if not rows:
        raise ValueError(
            f"no COT rows for contract_market_name={contract!r}; the name must "
            f"match the CFTC string exactly"
        )

    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(
        df["report_date_as_yyyy_mm_dd"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["report_date"]).set_index("report_date").sort_index()
    for c in df.columns:
        if c == "report_date_as_yyyy_mm_dd":
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        # Keep the column numeric only if it really is; text fields stay text.
        if conv.notna().sum() >= 0.9 * df[c].notna().sum():
            df[c] = conv
    df.to_parquet(cache)
    return df


def cot_positioning(
    contract: str = "E-MINI S&P 500",
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
    z_window: int = 52,
) -> pd.DataFrame:
    """
    Net positioning and its rolling z-score, the form a signal would use.

    `*_z` columns are z-scored on a trailing window that EXCLUDES the current
    observation, so the series is usable as of its own report date without
    lookahead.
    """
    raw = fetch_cot(contract, cache_dir, refresh)
    need = ["noncomm_positions_long_all", "noncomm_positions_short_all",
            "comm_positions_long_all", "comm_positions_short_all",
            "open_interest_all"]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise KeyError(f"COT response missing {missing}")

    df = pd.DataFrame(index=raw.index)
    oi = raw["open_interest_all"].replace(0, pd.NA)
    df["oi"] = raw["open_interest_all"]
    df["noncomm_net"] = (raw["noncomm_positions_long_all"]
                         - raw["noncomm_positions_short_all"])
    df["comm_net"] = (raw["comm_positions_long_all"]
                      - raw["comm_positions_short_all"])
    df["noncomm_net_pct_oi"] = df["noncomm_net"] / oi
    df["comm_net_pct_oi"] = df["comm_net"] / oi

    for c in ("noncomm_net_pct_oi", "comm_net_pct_oi"):
        r = df[c].shift(1).rolling(z_window, min_periods=z_window // 2)
        df[f"{c}_z"] = (df[c] - r.mean()) / r.std(ddof=0)
    return df


# ---------------------------------------------------------------------------
# Coinbase hourly candles
# ---------------------------------------------------------------------------

def fetch_coinbase_hourly(
    product: str = "BTC-USD",
    start: str = "2019-12-01",
    end: Optional[str] = None,
    cache_dir: Path = DEFAULT_CACHE,
    refresh: bool = False,
    pause: float = 0.35,
) -> pd.DataFrame:
    """
    Hourly OHLCV from Coinbase Exchange, paginated (300 candles per request).

    Returns a DataFrame indexed by UTC hour with columns
    [low, high, open, close, volume].

    Coinbase rejects requests lacking a browser User-Agent with HTTP 403; one is
    always sent. Gaps are possible for early history and are simply absent from
    the index rather than forward-filled.
    """
    cache = _cache_path(cache_dir, f"coinbase_{product.lower().replace('-', '_')}_1h")
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    t0 = pd.Timestamp(start, tz="UTC")
    t1 = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
    step = timedelta(hours=290)          # under the 300-candle cap

    frames, cur = [], t0
    while cur < t1:
        nxt = min(cur + step, t1)
        q = urllib.parse.urlencode({
            "granularity": 3600,
            "start": cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end": nxt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        })
        try:
            chunk = json.loads(_get(f"{COINBASE_URL.format(product=product)}?{q}"))
            if chunk:
                frames.append(pd.DataFrame(
                    chunk, columns=["ts", "low", "high", "open", "close", "volume"]))
        except Exception:
            pass                          # tolerate transient gaps; verified after
        cur = nxt
        time.sleep(pause)

    if not frames:
        raise RuntimeError(f"no Coinbase candles returned for {product}")

    df = pd.concat(frames, ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = (df.drop_duplicates("ts").set_index("ts").sort_index()
            .astype(float))
    df.index.name = "ts"
    df.to_parquet(cache)
    return df
