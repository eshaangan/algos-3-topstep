#!/usr/bin/env python3
"""
Analyze Jan 2026 backtest results with config fixes applied.

Compares:
1. Baseline results (before fixes)
2. Fixed results (vol_regime 30, stop 5.1x, target 4.35x)

Focus: Win rate, profit factor, stop-hit rate, SHORT signals
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_trades(trades_csv: Path) -> dict:
    """Analyze trades from CSV."""
    df = pd.read_csv(trades_csv)

    if len(df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'stop_hit_rate': 0.0,
            'target_hit_rate': 0.0,
            'median_duration_min': 0.0,
            'total_pnl': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'long_pct': 0.0,
            'short_pct': 0.0,
        }

    total_trades = len(df)
    winners = df[df['pnl'] > 0]
    losers = df[df['pnl'] <= 0]

    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

    gross_profit = winners['pnl'].sum() if len(winners) > 0 else 0
    gross_loss = abs(losers['pnl'].sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (np.inf if gross_profit > 0 else 0)

    stop_hits = df[df['exit_reason'] == 'stop']
    target_hits = df[df['exit_reason'] == 'target']

    stop_hit_rate = len(stop_hits) / total_trades * 100 if total_trades > 0 else 0
    target_hit_rate = len(target_hits) / total_trades * 100 if total_trades > 0 else 0

    long_trades = df[df['direction'] == 'LONG']
    short_trades = df[df['direction'] == 'SHORT']

    long_pct = len(long_trades) / total_trades * 100 if total_trades > 0 else 0
    short_pct = len(short_trades) / total_trades * 100 if total_trades > 0 else 0

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor if profit_factor != np.inf else 999.0,
        'stop_hit_rate': stop_hit_rate,
        'target_hit_rate': target_hit_rate,
        'median_duration_min': df['duration_minutes'].median(),
        'total_pnl': df['pnl'].sum(),
        'avg_win': winners['pnl'].mean() if len(winners) > 0 else 0,
        'avg_loss': losers['pnl'].mean() if len(losers) > 0 else 0,
        'long_pct': long_pct,
        'short_pct': short_pct,
    }

def main():
    # Find most recent baseline results
    results_dir = Path("/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/backtest_results")

    # Get most recent databento validation results
    validation_dirs = sorted(results_dir.glob("databento_validation_*"))
    if not validation_dirs:
        print("No validation results found")
        return

    latest_dir = validation_dirs[-1]
    print(f"\n{'='*80}")
    print(f"ANALYZING: {latest_dir.name}")
    print(f"{'='*80}\n")

    # Analyze each scenario
    scenarios = ['baseline', 'high_confidence', 'aggressive']
    results = {}

    for scenario in scenarios:
        scenario_dir = latest_dir / scenario
        if not scenario_dir.exists():
            continue

        trades_files = list(scenario_dir.glob("trades_*.csv"))
        if not trades_files:
            continue

        trades_csv = trades_files[0]
        results[scenario] = analyze_trades(trades_csv)

    if not results:
        print("No results to analyze")
        return

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARATIVE RESULTS")
    print(f"{'='*80}\n")

    metrics = [
        ('total_trades', 'Total Trades', ''),
        ('win_rate', 'Win Rate', '%'),
        ('profit_factor', 'Profit Factor', ''),
        ('stop_hit_rate', 'Stop Hit Rate', '%'),
        ('target_hit_rate', 'Target Hit Rate', '%'),
        ('median_duration_min', 'Median Duration', 'min'),
        ('total_pnl', 'Total PnL', '$'),
        ('avg_win', 'Avg Win', '$'),
        ('avg_loss', 'Avg Loss', '$'),
        ('long_pct', 'LONG %', '%'),
        ('short_pct', 'SHORT %', '%'),
    ]

    # Header
    print(f"{'Metric':<25}", end='')
    for scenario in scenarios:
        if scenario in results:
            print(f"{scenario:>20}", end='')
    print()
    print('-' * 80)

    # Rows
    for key, label, unit in metrics:
        print(f"{label:<25}", end='')
        for scenario in scenarios:
            if scenario in results:
                value = results[scenario][key]
                if unit == '%':
                    print(f"{value:>19.1f}%", end='')
                elif unit == '$':
                    print(f"${value:>18.2f}", end='')
                elif unit == 'min':
                    print(f"{value:>19.1f}m", end='')
                elif isinstance(value, float):
                    print(f"{value:>20.2f}", end='')
                else:
                    print(f"{value:>20}", end='')
        print()

    # Check vol_regime NaN improvement
    print(f"\n{'='*80}")
    print("VOL_REGIME WARMUP CHECK")
    print(f"{'='*80}\n")

    # Load a features file to check NaN counts
    baseline_dir = latest_dir / 'baseline'
    features_files = list(baseline_dir.glob("features_*.parquet"))
    if features_files:
        features_df = pd.read_parquet(features_files[0])
        if 'vol_regime' in features_df.columns:
            vol_regime_nans = features_df['vol_regime'].isna().sum()
            total_rows = len(features_df)
            print(f"vol_regime NaN count: {vol_regime_nans}/{total_rows} ({vol_regime_nans/total_rows*100:.1f}%)")
            print(f"Expected improvement: 70 → ~49 NaNs (30% reduction)")
        else:
            print("vol_regime column not found in features")
    else:
        print("No features file found")

    # Key takeaways
    print(f"\n{'='*80}")
    print("KEY TAKEAWAYS")
    print(f"{'='*80}\n")

    baseline = results.get('baseline', {})

    print(f"1. Win Rate: {baseline.get('win_rate', 0):.1f}%")
    print(f"   - Target: >25% (original was 13.7%)")
    print(f"   - Status: {'✓ PASS' if baseline.get('win_rate', 0) > 25 else '✗ FAIL'}")

    print(f"\n2. Profit Factor: {baseline.get('profit_factor', 0):.2f}")
    print(f"   - Target: >0.8 (original was 0.19)")
    print(f"   - Status: {'✓ PASS' if baseline.get('profit_factor', 0) > 0.8 else '✗ FAIL'}")

    print(f"\n3. Stop Hit Rate: {baseline.get('stop_hit_rate', 0):.1f}%")
    print(f"   - Target: <70% (original was 86%)")
    print(f"   - Status: {'✓ PASS' if baseline.get('stop_hit_rate', 0) < 70 else '✗ FAIL'}")

    print(f"\n4. SHORT Signals: {baseline.get('short_pct', 0):.1f}%")
    print(f"   - Target: >10% (original was 0%)")
    print(f"   - Status: {'✓ PASS' if baseline.get('short_pct', 0) > 10 else '✗ FAIL'}")

    # Save results
    output_file = latest_dir / "analysis_summary.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()
