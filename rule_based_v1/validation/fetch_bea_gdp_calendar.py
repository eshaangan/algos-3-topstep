"""BEA GDP advance-estimate release dates, from BEA's own news archive.

Unlike ISM (deterministic: first business day of the month), GDP release dates
follow no rule you can derive, so they have to come from BEA. Each archive row
carries a <time datetime="..."> stamp with the actual release instant, which is
what an 8:30 ET event study needs -- guessing "late in the month after quarter
end" would put the window on the wrong day often enough to destroy the test.

Only ADVANCE estimates are kept. Second/third estimates revise numbers the market
has already seen, so they are not the uncertainty-resolution event HPWZ describe.

Usage:
    python3 rule_based_v1/validation/fetch_bea_gdp_calendar.py \
        --pages 40 --out data/processed/bea_gdp_advance_releases.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import requests

ARCHIVE = ("https://www.bea.gov/news/archive"
           "?field_related_product_target_id=451&created=All&title=&page={page}")
HEADERS = {"User-Agent": "Mozilla/5.0 (research; contact via repo owner)"}
ROW = re.compile(
    r'<td[^>]*views-field-title"><a[^>]*>(?P<title>[^<]+)</a>.*?'
    r'<time datetime="(?P<ts>[^"]+)"',
    re.S,
)
# BEA has used several namings: "GDP (Advance Estimate), 2nd Quarter 2026" and
# older "Gross Domestic Product, 4th quarter 2016 (advance estimate)".
ADVANCE = re.compile(r"advance\s+estimate", re.I)


def main() -> None:
    ap = argparse.ArgumentParser(description="BEA GDP advance release dates")
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--cache-dir", type=Path, default=Path("data/raw/bea_pages"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    rows, seen_pages = [], 0
    for page in range(args.pages):
        cached = args.cache_dir / f"page{page}.html"
        if cached.exists():
            html = cached.read_text(errors="replace")
        else:
            r = requests.get(ARCHIVE.format(page=page), headers=HEADERS, timeout=60)
            if r.status_code != 200:
                break
            html = r.text
            cached.write_text(html, errors="replace")
        found = list(ROW.finditer(html))
        if not found:
            break
        seen_pages += 1
        for m in found:
            title = m.group("title").strip()
            if not ADVANCE.search(title):
                continue
            ts = pd.Timestamp(m.group("ts"))
            rows.append({"release_date": ts.tz_convert("America/New_York").date(),
                         "release_time_et": ts.tz_convert("America/New_York").strftime("%H:%M"),
                         "title": title})

    frame = (pd.DataFrame(rows).drop_duplicates(subset="release_date")
             .sort_values("release_date").reset_index(drop=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    print(f"scanned {seen_pages} archive pages -> {len(frame)} advance estimates")
    if not frame.empty:
        print(f"span {frame['release_date'].min()} .. {frame['release_date'].max()}")
        print("release times seen:", frame["release_time_et"].value_counts().to_dict())
        per_year = frame["release_date"].map(lambda d: d.year).value_counts().sort_index()
        odd = {y: n for y, n in per_year.items() if n != 4}
        print("years without exactly 4 advance estimates:", odd or "none")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
