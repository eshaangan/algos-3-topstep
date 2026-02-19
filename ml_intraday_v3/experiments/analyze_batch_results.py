#!/usr/bin/env python3
"""Analyze batch experiment results and rank top configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _score_row(row: pd.Series) -> float:
    auc = float(row.get("median_test_auc", 0.0))
    signal = float(row.get("median_pct_signals_above_055", 0.0))
    gap = abs(float(row.get("mean_train_test_gap", 0.0)))
    stability = float(row.get("std_test_auc", 1.0))

    # Higher is better.
    return 0.40 * auc + 0.30 * signal + 0.20 * (1.0 - min(gap, 1.0)) + 0.10 * (1.0 - min(stability, 1.0))


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("result_*.json")):
        with open(path, "r") as f:
            data = json.load(f)
        if data.get("status") != "SUCCESS":
            continue
        summary = data.get("summary", {})
        rows.append(
            {
                "exp_id": data.get("exp_id"),
                "phase": data.get("config", {}).get("phase"),
                "path": str(path),
                **summary,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--batch", default="all")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-dir", default="ml_intraday_v3/experiments/results")
    args = parser.parse_args()

    df = load_results(Path(args.results_dir))
    if df.empty:
        raise SystemExit("No successful result_*.json files found")

    if args.batch != "all":
        df = df[df["phase"] == args.batch].copy()
        if df.empty:
            raise SystemExit(f"No rows found for batch={args.batch}")

    df["composite_score"] = df.apply(_score_row, axis=1)
    ranked = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.batch}_ranked"

    ranked.to_csv(out_dir / f"{stem}.csv", index=False)
    ranked.head(args.top_k).to_csv(out_dir / f"{stem}_top{args.top_k}.csv", index=False)

    summary = {
        "batch": args.batch,
        "n_rows": int(len(ranked)),
        "top_k": int(args.top_k),
        "best_exp_id": ranked.iloc[0]["exp_id"],
        "best_composite": float(ranked.iloc[0]["composite_score"]),
    }
    with open(out_dir / f"{stem}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
