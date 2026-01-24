"""
Compare backtest results between baseline and enhanced filtering.

Usage:
    python -m ml_intraday_v3.analysis.compare_backtests \
        --baseline runs/v3_2022_5m/bar_size=5m/backtests/baseline \
        --enhanced runs/v3_2022_5m/bar_size=5m/backtests/enhanced
"""

import argparse
import json
from pathlib import Path
import pandas as pd


def load_backtest_metrics(backtest_dir: Path) -> dict:
    """Load summary metrics from backtest directory."""
    summary_path = backtest_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")

    with open(summary_path) as f:
        return json.load(f)


def compare_metrics(baseline: dict, enhanced: dict) -> pd.DataFrame:
    """Compare key metrics between baseline and enhanced."""
    metrics = [
        "sharpe_ratio",
        "win_rate",
        "avg_win_usd",
        "avg_loss_usd",
        "total_pnl_usd",
        "max_drawdown_usd",
        "total_trades",
        "profit_factor",
    ]

    comparison = []
    for metric in metrics:
        baseline_val = baseline.get(metric, 0)
        enhanced_val = enhanced.get(metric, 0)

        if baseline_val != 0:
            pct_change = ((enhanced_val - baseline_val) / abs(baseline_val)) * 100
        else:
            pct_change = 0

        comparison.append({
            "Metric": metric,
            "Baseline": baseline_val,
            "Enhanced": enhanced_val,
            "Change": enhanced_val - baseline_val,
            "Change %": pct_change,
        })

    return pd.DataFrame(comparison)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--enhanced", type=Path, required=True)
    args = parser.parse_args()

    baseline_metrics = load_backtest_metrics(args.baseline)
    enhanced_metrics = load_backtest_metrics(args.enhanced)

    comparison = compare_metrics(baseline_metrics, enhanced_metrics)

    print("\n=== BACKTEST COMPARISON: Baseline vs Enhanced ===\n")
    print(comparison.to_string(index=False))

    # Highlight key improvements
    print("\n=== KEY INSIGHTS ===")

    sharpe_change = comparison[comparison['Metric'] == 'sharpe_ratio']['Change'].values[0]
    if sharpe_change > 0:
        print(f"✓ Sharpe improved by {sharpe_change:.2f}")
    else:
        print(f"✗ Sharpe degraded by {abs(sharpe_change):.2f}")

    trades_change_pct = comparison[comparison['Metric'] == 'total_trades']['Change %'].values[0]
    print(f"  Trade count changed by {trades_change_pct:.1f}%")

    win_rate_change = comparison[comparison['Metric'] == 'win_rate']['Change'].values[0]
    if win_rate_change > 0:
        print(f"✓ Win rate improved by {win_rate_change:.1f}%")
    else:
        print(f"  Win rate changed by {win_rate_change:.1f}%")

    pnl_change = comparison[comparison['Metric'] == 'total_pnl_usd']['Change'].values[0]
    print(f"  Total P&L changed by ${pnl_change:.2f}")

    dd_change = comparison[comparison['Metric'] == 'max_drawdown_usd']['Change'].values[0]
    if dd_change < 0:  # Negative change in drawdown is good
        print(f"✓ Max drawdown reduced by ${abs(dd_change):.2f}")
    else:
        print(f"  Max drawdown increased by ${dd_change:.2f}")


if __name__ == "__main__":
    main()
