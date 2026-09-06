"""Scheduled FOMC announcement dates, straight from federalreserve.gov.

The edge-hunt ledger records that earlier FOMC work used dates recalled from
model memory. This builds the calendar from the Fed's own pages instead, taking
the announcement date from the statement-release URL (``monetary<YYYYMMDD>a``)
inside each meeting block rather than from the heading text, so two-day and
month-straddling meetings resolve to the day the market actually heard.

Excluded, by the Fed's own labels: conference calls, unscheduled meetings,
notation votes, and the cancelled March 17-18 2020 meeting. Those are real Fed
actions but they are not the scheduled cycle, and including them corrupts
FOMC cycle time.

Usage:
    python3 rule_based_v1/validation/fetch_fomc_calendar.py \
        --start 2010 --end 2025 --out data/processed/fomc_announcements.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import requests

HISTORICAL = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
CURRENT = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; contact via repo owner)"}

# Statement URLs appear as /newsevents/pressreleases/monetary20240131a.htm on
# modern pages and /newsevents/press/monetary/20100127a.htm on pre-2011 pages.
STATEMENT = re.compile(r"(?:monetary|/monetary/)(\d{8})a")
EXCLUDE = ("cancelled", "unscheduled", "notation", "conference call")


def _get(url: str, cache_dir: Path) -> str | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / (re.sub(r"\W+", "_", url) + ".html")
    if cached.exists():
        return cached.read_text(errors="replace")
    r = requests.get(url, headers=HEADERS, timeout=60)
    if r.status_code != 200 or "FOMC" not in r.text:
        return None
    cached.write_text(r.text, errors="replace")
    return r.text


def parse_historical(html: str, year: int) -> tuple[set[str], list[str]]:
    keep, dropped = set(), []
    for block in re.split(r"<h5[^>]*>", html)[1:]:
        label = re.sub(r"<[^>]+>", "", re.split(r"</h5>", block)[0]).strip()
        low = label.lower()
        if "meeting" not in low or any(k in low for k in EXCLUDE):
            dropped.append(label)
            continue
        hits = [d for d in STATEMENT.findall(block[:20000]) if d.startswith(str(year))]
        if not hits:
            dropped.append(f"{label} [no statement link]")
            continue
        keep.add(sorted(set(hits))[-1])
    return keep, dropped


def parse_current(html: str, lo: int, hi: int) -> tuple[set[str], list[str]]:
    keep, dropped = set(), []
    chunks = re.split(r'<h4><a id="\d+">(\d{4}) FOMC Meetings</a></h4>', html)
    for i in range(1, len(chunks), 2):
        year = int(chunks[i])
        if not lo <= year <= hi:
            continue
        # The layout packs two meetings per row, so work row by row and read the
        # date cell to catch the "(notation vote)" style labels.
        # Split on the row container only -- the inner fomc-meeting__month and
        # __date divs also carry a "fomc-meeting" prefix and would fragment rows.
        for row in re.split(r'<div class="[^"]*row fomc-meeting[^"]*"', chunks[i + 1])[1:]:
            cell = re.search(r'fomc-meeting__date[^>]*>([^<]*)<', row)
            label = (cell.group(1) if cell else "").strip()
            if any(k in label.lower() for k in EXCLUDE):
                dropped.append(f"{year} {label}")
                continue
            for d in STATEMENT.findall(row):
                if d.startswith(str(year)):
                    keep.add(d)
    return keep, dropped


def build(start: int, end: int, cache_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    dates: set[str] = set()
    dropped: list[str] = []
    for year in range(start, end + 1):
        html = _get(HISTORICAL.format(year=year), cache_dir)
        if html is None:
            continue
        k, d = parse_historical(html, year)
        dates |= k
        dropped += [f"{year} {x}" for x in d]
    html = _get(CURRENT, cache_dir)
    if html is not None:
        k, d = parse_current(html, start, end)
        dates |= k
        dropped += d
    frame = pd.DataFrame(
        {"announcement_date": sorted(dt.datetime.strptime(x, "%Y%m%d").date() for x in dates)}
    )
    return frame, dropped


def main() -> None:
    ap = argparse.ArgumentParser(description="Scheduled FOMC announcement dates")
    ap.add_argument("--start", type=int, default=2010)
    ap.add_argument("--end", type=int, default=dt.date.today().year)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/raw/fomc_pages"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    frame, dropped = build(args.start, args.end, args.cache_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    per_year = Counter(d.year for d in frame["announcement_date"])
    print(f"{len(frame)} scheduled announcements -> {args.out}")
    for y in sorted(per_year):
        flag = "" if per_year[y] == 8 else "   <-- not 8, check"
        print(f"  {y}: {per_year[y]}{flag}")
    print(f"\nexcluded {len(dropped)} non-scheduled entries:")
    for x in dropped:
        print(f"   {x}")


if __name__ == "__main__":
    main()
