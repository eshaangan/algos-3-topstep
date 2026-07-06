"""Consolidate the many small L2 recorder parquet chunks into one file per day per
type, so the dataset is shareable (a few dozen files instead of ~44k) and faster to
load. Same schema, same rows — just merged and re-compressed with zstd.

    python data_collection/consolidate_l2.py                       # -> data/l2_consolidated/
    python data_collection/consolidate_l2.py --out data/share_l2   # custom dest

Output: <out>/{bbo,trade}_MNQ_YYYYMMDD.parquet  (one each per recorded day).
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAY_RE = re.compile(r"_(20\d{6})_")  # ..._YYYYMMDD_<ns>.parquet


def _group_by_day(src: Path, kind: str) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for f in src.glob(f"{kind}_*.parquet"):
        m = DAY_RE.search(f.name)
        if not m:
            continue
        day = m.group(1)
        # skip the malformed epoch-ns day tokens (real days are 20260613..20260706)
        if not ("20260101" <= day <= "20261231"):
            continue
        groups[day].append(f)
    return groups


def consolidate(src: Path, out: Path, kind: str) -> None:
    groups = _group_by_day(src, kind)
    out.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for day in sorted(groups):
        files = sorted(groups[day])
        frames = []
        for f in files:
            try:
                frames.append(pd.read_parquet(f))
            except Exception as e:  # skip a torn chunk rather than abort the day
                print(f"  ! skip {f.name}: {e}")
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        # de-dup on the recorder's monotonic receive clock if present
        if "recv_ns" in df.columns:
            df = df.drop_duplicates(subset="recv_ns").sort_values("recv_ns")
        dest = out / f"{kind}_MNQ_{day}.parquet"
        df.to_parquet(dest, compression="zstd", index=False)
        total_rows += len(df)
        print(f"  {kind} {day}: {len(files):>5} chunks → {len(df):>10,} rows  {dest.name}")
    print(f"  {kind}: {total_rows:,} total rows across {len(groups)} days")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data" / "l2_raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "l2_consolidated"))
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    print(f"Consolidating {src} → {out}")
    for kind in ("bbo", "trade", "book"):
        consolidate(src, out, kind)
    files = sorted(out.glob("*.parquet"))
    size = sum(f.stat().st_size for f in files) / 1e9
    print(f"\nDone: {len(files)} files, {size:.2f} GB in {out}")


if __name__ == "__main__":
    main()
