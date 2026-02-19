#!/usr/bin/env python3
"""
Rank experiment results by performance metric.

Usage:
    python rank_models.py --results-dir results/ --metric sharpe_ratio --top-n 10
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import pandas as pd


def load_all_results(results_dir: Path) -> List[Dict]:
    """Load all result JSON files from directory."""
    results = []

    for batch_dir in results_dir.glob("batch*"):
        if not batch_dir.is_dir():
            continue

        for result_file in batch_dir.glob("*.json"):
            if result_file.name.endswith("_summary.json"):
                continue

            try:
                with open(result_file, "r") as f:
                    result = json.load(f)
                    results.append(result)
            except Exception as e:
                print(f"Warning: Failed to load {result_file}: {e}")

    return results


def extract_metric(result: Dict, metric_name: str) -> float:
    """Extract metric value from result dict."""
    if result.get("status") != "success":
        return float("-inf")

    metrics = result.get("metrics", {})

    if metric_name == "sharpe_ratio":
        # Calculate from test metrics
        test_pnl = metrics.get("test_pnl", 0)
        test_trades = metrics.get("test_trades", 0)
        if test_trades == 0:
            return float("-inf")

        # Estimate Sharpe (simplified)
        avg_pnl_per_trade = test_pnl / test_trades
        # Assume ~250 trading days, 20 trades/day
        annual_trades = 250 * 20
        annual_pnl = avg_pnl_per_trade * annual_trades
        # Rough vol estimate (improve later)
        sharpe = annual_pnl / (abs(annual_pnl) * 0.5 + 1)
        return sharpe

    elif metric_name == "test_auc":
        return metrics.get("test_auc", 0.0)

    elif metric_name == "test_pnl":
        return metrics.get("test_pnl", 0.0)

    elif metric_name == "win_rate":
        wins = metrics.get("test_wins", 0)
        losses = metrics.get("test_losses", 0)
        total = wins + losses
        if total == 0:
            return 0.0
        return wins / total

    elif metric_name == "max_drawdown":
        # Negative sign so lower DD ranks higher
        return -metrics.get("test_max_dd", float("inf"))

    else:
        return metrics.get(metric_name, float("-inf"))


def rank_models(
    results: List[Dict],
    metric: str,
    top_n: int = 10,
    min_test_trades: int = 10,
) -> pd.DataFrame:
    """Rank models by metric."""
    data = []

    for result in results:
        if result.get("status") != "success":
            continue

        metrics = result.get("metrics", {})

        # Filter by minimum trades
        if metrics.get("test_trades", 0) < min_test_trades:
            continue

        row = {
            "exp_id": result.get("exp_id"),
            "metric_value": extract_metric(result, metric),
            "test_auc": metrics.get("test_auc", 0.0),
            "test_pnl": metrics.get("test_pnl", 0.0),
            "test_trades": metrics.get("test_trades", 0),
            "win_rate": (
                metrics.get("test_wins", 0) /
                max(metrics.get("test_wins", 0) + metrics.get("test_losses", 0), 1)
            ),
            "test_max_dd": metrics.get("test_max_dd", 0.0),
            "labeling_method": result.get("labeling_method", "unknown"),
            "sample_weight": result.get("sample_weight", "unknown"),
            "cv_method": result.get("cv_method", "unknown"),
        }
        data.append(row)

    if not data:
        print("No valid results found!")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.sort_values("metric_value", ascending=False).head(top_n)

    return df


def main():
    parser = argparse.ArgumentParser(description="Rank experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Directory containing batch results",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sharpe_ratio",
        choices=["sharpe_ratio", "test_auc", "test_pnl", "win_rate", "max_drawdown"],
        help="Metric to rank by",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top models to return",
    )
    parser.add_argument(
        "--min-test-trades",
        type=int,
        default=10,
        help="Minimum test trades required",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: print to stdout)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        return 1

    print(f"Loading results from: {results_dir}")
    results = load_all_results(results_dir)
    print(f"Loaded {len(results)} results")

    print(f"\nRanking by: {args.metric}")
    ranked_df = rank_models(
        results,
        metric=args.metric,
        top_n=args.top_n,
        min_test_trades=args.min_test_trades,
    )

    if ranked_df.empty:
        print("No valid models found!")
        return 1

    print(f"\nTop {len(ranked_df)} models:")
    print("=" * 100)
    print(ranked_df.to_string(index=False))
    print()

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON with full configs
        top_exp_ids = ranked_df["exp_id"].tolist()
        top_results = [r for r in results if r.get("exp_id") in top_exp_ids]

        with open(output_path, "w") as f:
            json.dump(top_results, f, indent=2)

        print(f"✅ Saved top {len(top_results)} model configs to: {output_path}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
