"""Fetch ECB Governing Council monetary policy DECISION dates from ecb.europa.eu.

Dates are derived from the ECB's own press-release URL slugs, never from model
memory. Each monetary policy decision is published as

    /press/govcdec/mopo/<year>/html/ecb.mp<YYMMDD>~<hash>.en.html

so the `mp` slug carries the decision date. This mirrors what
`fetch_fomc_calendar.py` does with the Fed's `monetary<YYYYMMDD>a` URLs, and it
exists for the same reason: the ledger has already been burned once by dates
that came from memory rather than from the issuing institution.

RELEASE TIME IS NOT CONSTANT, and this matters for any pre-announcement test:

    up to and including 2022-07-20 : decision published 13:45 CET
    from            2022-07-21     : decision published 14:15 CET

(ECB press release 2022-06-27 announced the change, effective 21 July 2022. The
press conference moved from 14:30 to 14:45 CET at the same time.)

Times are stored as EUROPE/BERLIN LOCAL, not as a fixed US hour. Frankfurt and
New York shift to and from daylight saving on different dates, so a hardcoded ET
hour is wrong for roughly three weeks a year -- twice a year, in March and
October/November, which is exactly when several ECB meetings fall.

Usage:
    python3 rule_based_v1/validation/fetch_ecb_calendar.py \
        --start 2020 --end 2026 --out data/processed/ecb_decisions.csv
"""
from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from pathlib import Path

import pandas as pd

INDEX = ("https://www.ecb.europa.eu/press/govcdec/mopo/{year}/html/"
         "index_include.en.html")
SLUG = re.compile(r"ecb\.mp(\d{6})")

# The publication-time change, as announced by the ECB on 2022-06-27.
TIME_CHANGE_DATE = pd.Timestamp("2022-07-21")
TIME_BEFORE = "13:45"
TIME_FROM = "14:15"


def fetch_year(year: int, timeout: int = 30) -> list[pd.Timestamp]:
    req = urllib.request.Request(
        INDEX.format(year=year),
        headers={"User-Agent": "Mozilla/5.0 (research calendar fetch)"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        html = fh.read().decode("utf-8", errors="replace")
    out = set()
    for yymmdd in SLUG.findall(html):
        try:
            d = pd.Timestamp(f"20{yymmdd[:2]}-{yymmdd[2:4]}-{yymmdd[4:]}")
        except ValueError:
            continue
        if d.year == year:
            out.add(d)
    return sorted(out)


def release_time_cet(d: pd.Timestamp) -> str:
    return TIME_BEFORE if d < TIME_CHANGE_DATE else TIME_FROM


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=2020)
    ap.add_argument("--end", type=int, default=2026)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    rows = []
    for year in range(a.start, a.end + 1):
        try:
            days = fetch_year(year)
        except Exception as exc:                      # noqa: BLE001
            print(f"  {year}: FETCH FAILED ({exc})", file=sys.stderr)
            continue
        print(f"  {year}: {len(days)} decisions")
        for d in days:
            rows.append({"decision_date": d.date().isoformat(),
                         "release_time_cet": release_time_cet(d),
                         "tz": "Europe/Berlin"})

    if not rows:
        raise SystemExit("no decisions fetched; refusing to write an empty calendar")

    df = pd.DataFrame(rows).sort_values("decision_date").reset_index(drop=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    per_year = df.groupby(df["decision_date"].str[:4]).size()
    print(f"\nwrote {a.out}  n={len(df)}")
    print(per_year.to_string())
    odd = per_year[(per_year != 8) & (per_year.index.astype(int) < 2026)]
    if len(odd):
        print(f"\nWARNING: the ECB holds 8 monetary policy meetings a year; "
              f"these years do not have 8 and should be checked by hand:\n{odd}")


if __name__ == "__main__":
    main()
