"""Historical earnings dates for the mega-cap NDX constituents, from Alpha Vantage.

The set is fixed to the "Magnificent 7" -- AAPL, MSFT, NVDA, AMZN, GOOGL, META,
TSLA. That matters for overfit control: it is an externally-defined, widely-used
grouping chosen for its index weight, not a basket selected by looking at which
tickers happened to move MNQ. Together they are roughly 40-45% of the Nasdaq-100,
which is the entire reason an index-level effect is plausible at all.

Alpha Vantage's EARNINGS endpoint returns `reportedDate` per quarter, which is the
calendar date the report was released. All seven report AFTER the close, so the
uncertainty is resolved overnight and the pre-announcement window has to end at
that day's close.

Usage:
    python3 rule_based_v1/validation/fetch_megacap_earnings.py \
        --out data/processed/megacap_earnings.csv
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import requests

MAG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
URL = "https://www.alphavantage.co/query"


def fetch(symbol: str, api_key: str, cache_dir: Path) -> dict:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / f"{symbol}.json"
    if cached.exists():
        return json.loads(cached.read_text())
    r = requests.get(URL, params={"function": "EARNINGS", "symbol": symbol,
                                  "apikey": api_key}, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "quarterlyEarnings" not in data:
        raise SystemExit(f"{symbol}: unexpected response {list(data)[:3]} -- rate limited?")
    cached.write_text(json.dumps(data))
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="mega-cap earnings dates")
    ap.add_argument("--symbols", nargs="+", default=MAG7)
    ap.add_argument("--api-key", default=os.environ.get("ALPHAVANTAGE_API_KEY",
                                                        "3XBV9VY0KQRJ3LO8"))
    ap.add_argument("--cache-dir", type=Path, default=Path("data/raw/av_earnings"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for i, sym in enumerate(args.symbols):
        data = fetch(sym, args.api_key, args.cache_dir)
        for q in data["quarterlyEarnings"]:
            rd = q.get("reportedDate")
            if not rd:
                continue
            rows.append({"symbol": sym, "report_date": rd,
                         "fiscal_end": q.get("fiscalDateEnding"),
                         "reported_eps": q.get("reportedEPS"),
                         "estimated_eps": q.get("estimatedEPS"),
                         "surprise_pct": q.get("surprisePercentage")})
        if i < len(args.symbols) - 1:
            time.sleep(1)

    df = pd.DataFrame(rows)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")
    df = df.dropna(subset=["report_date"]).sort_values(["report_date", "symbol"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"{len(df)} earnings dates -> {args.out}")
    print(f"span {df['report_date'].min().date()} .. {df['report_date'].max().date()}")
    per = df[df["report_date"] >= "2020-01-01"].groupby("symbol").size()
    print("\nreports since 2020 per symbol:")
    print(per.to_string())
    yr = df[df["report_date"] >= "2020-01-01"].groupby(df["report_date"].dt.year).size()
    print("\nreports per year (all symbols):")
    print(yr.to_string())


if __name__ == "__main__":
    main()
