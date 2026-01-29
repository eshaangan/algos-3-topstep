#!/usr/bin/env python3
"""
Diagnostic Analysis: Model Performance Degradation (Jan 2026)

Investigates the critical issues:
1. Directional bias: 100% LONG, 0% SHORT signals
2. Win rate collapse: 58% → 13.7%
3. Stop-loss hit rate: 86% vs 14% targets
4. Feature quality: vol_regime 100% NaN

Author: Claude Code
Date: 2026-01-24
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import json

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
sns.set_style('whitegrid')

class ModelDegradationDiagnostic:
    """Diagnose model performance degradation on Jan 2026 data"""

    def __init__(self, backtest_root: Path, output_dir: Path):
        self.backtest_root = Path(backtest_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Load data
        self.signals_baseline = pd.read_csv(
            self.backtest_root / "baseline" / "signals_20260124_182119.csv"
        )
        self.trades_baseline = pd.read_csv(
            self.backtest_root / "baseline" / "trades_20260124_182119.csv"
        )

        # Convert timestamps
        self.signals_baseline['timestamp'] = pd.to_datetime(self.signals_baseline['timestamp'])
        self.trades_baseline['entry_time'] = pd.to_datetime(self.trades_baseline['entry_time'])
        self.trades_baseline['exit_time'] = pd.to_datetime(self.trades_baseline['exit_time'])

        print(f"✓ Loaded {len(self.signals_baseline)} signals")
        print(f"✓ Loaded {len(self.trades_baseline)} trades")

    def task1_analyze_directional_bias(self) -> Dict:
        """Task #1: Investigate 100% LONG, 0% SHORT bias"""
        print("\n" + "="*80)
        print("TASK #1: DIRECTIONAL BIAS ANALYSIS")
        print("="*80)

        results = {}

        # Direction distribution
        direction_counts = self.signals_baseline['direction'].value_counts()
        direction_pct = self.signals_baseline['direction'].value_counts(normalize=True) * 100

        print(f"\n📊 Signal Direction Distribution:")
        print(f"  LONG:  {direction_counts.get('LONG', 0):>3} signals ({direction_pct.get('LONG', 0):>5.1f}%)")
        print(f"  SHORT: {direction_counts.get('SHORT', 0):>3} signals ({direction_pct.get('SHORT', 0):>5.1f}%)")

        results['direction_counts'] = direction_counts.to_dict()
        results['direction_pct'] = direction_pct.to_dict()

        # Score distribution
        print(f"\n📈 Score Statistics:")
        print(f"  Min:    {self.signals_baseline['score'].min():.4f}")
        print(f"  25%:    {self.signals_baseline['score'].quantile(0.25):.4f}")
        print(f"  Median: {self.signals_baseline['score'].median():.4f}")
        print(f"  75%:    {self.signals_baseline['score'].quantile(0.75):.4f}")
        print(f"  Max:    {self.signals_baseline['score'].max():.4f}")
        print(f"  Mean:   {self.signals_baseline['score'].mean():.4f}")
        print(f"  Std:    {self.signals_baseline['score'].std():.4f}")

        results['score_stats'] = {
            'min': float(self.signals_baseline['score'].min()),
            'q25': float(self.signals_baseline['score'].quantile(0.25)),
            'median': float(self.signals_baseline['score'].median()),
            'q75': float(self.signals_baseline['score'].quantile(0.75)),
            'max': float(self.signals_baseline['score'].max()),
            'mean': float(self.signals_baseline['score'].mean()),
            'std': float(self.signals_baseline['score'].std()),
        }

        # Score by direction
        if 'SHORT' in self.signals_baseline['direction'].values:
            print(f"\n📊 Score by Direction:")
            for direction in ['LONG', 'SHORT']:
                scores = self.signals_baseline[self.signals_baseline['direction'] == direction]['score']
                print(f"  {direction}:")
                print(f"    Count: {len(scores)}")
                print(f"    Mean:  {scores.mean():.4f}")
                print(f"    Std:   {scores.std():.4f}")
        else:
            print(f"\n⚠️  WARNING: No SHORT signals to compare!")

        # Temporal analysis
        print(f"\n📅 Signal Distribution Over Time:")
        self.signals_baseline['date'] = self.signals_baseline['timestamp'].dt.date
        daily_signals = self.signals_baseline.groupby(['date', 'direction']).size().unstack(fill_value=0)
        print(daily_signals)

        results['daily_distribution'] = daily_signals.to_dict()

        # Plot score distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Histogram
        axes[0, 0].hist(self.signals_baseline['score'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(self.signals_baseline['score'].median(), color='red',
                          linestyle='--', label=f'Median: {self.signals_baseline["score"].median():.3f}')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Score Distribution (All Signals)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Box plot by direction
        direction_data = [
            self.signals_baseline[self.signals_baseline['direction'] == d]['score'].values
            for d in ['LONG', 'SHORT'] if d in self.signals_baseline['direction'].values
        ]
        direction_labels = [
            d for d in ['LONG', 'SHORT'] if d in self.signals_baseline['direction'].values
        ]
        axes[0, 1].boxplot(direction_data, labels=direction_labels)
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Score Distribution by Direction')
        axes[0, 1].grid(True, alpha=0.3)

        # Time series
        daily_avg_score = self.signals_baseline.groupby('date')['score'].mean()
        axes[1, 0].plot(daily_avg_score.index, daily_avg_score.values, marker='o')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Average Score')
        axes[1, 0].set_title('Average Daily Score Over Time')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Direction counts
        axes[1, 1].bar(direction_counts.index, direction_counts.values, edgecolor='black')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Signal Count by Direction')
        axes[1, 1].grid(True, alpha=0.3)
        for i, (idx, val) in enumerate(direction_counts.items()):
            axes[1, 1].text(i, val + 2, str(val), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plot_path = self.output_dir / 'task1_directional_bias.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Saved plot: {plot_path}")
        plt.close()

        return results

    def task3_analyze_stop_loss_hits(self) -> Dict:
        """Task #3: Analyze 86% stop-loss vs 14% target hit rate"""
        print("\n" + "="*80)
        print("TASK #3: STOP-LOSS HIT RATE ANALYSIS")
        print("="*80)

        results = {}

        # Exit reason distribution
        exit_counts = self.trades_baseline['exit_reason'].value_counts()
        exit_pct = self.trades_baseline['exit_reason'].value_counts(normalize=True) * 100

        print(f"\n📊 Exit Reason Distribution:")
        for reason in exit_counts.index:
            print(f"  {reason:20s}: {exit_counts[reason]:>3} trades ({exit_pct[reason]:>5.1f}%)")

        results['exit_counts'] = exit_counts.to_dict()
        results['exit_pct'] = exit_pct.to_dict()

        # Trade duration analysis
        self.trades_baseline['duration_minutes'] = (
            (self.trades_baseline['exit_time'] - self.trades_baseline['entry_time']).dt.total_seconds() / 60
        )

        print(f"\n⏱️  Trade Duration by Exit Reason:")
        for reason in exit_counts.index:
            durations = self.trades_baseline[self.trades_baseline['exit_reason'] == reason]['duration_minutes']
            print(f"  {reason:20s}:")
            print(f"    Count:  {len(durations)}")
            print(f"    Mean:   {durations.mean():.1f} min")
            print(f"    Median: {durations.median():.1f} min")
            print(f"    Std:    {durations.std():.1f} min")

        results['duration_stats'] = {}
        for reason in exit_counts.index:
            durations = self.trades_baseline[self.trades_baseline['exit_reason'] == reason]['duration_minutes']
            results['duration_stats'][reason] = {
                'count': int(len(durations)),
                'mean': float(durations.mean()),
                'median': float(durations.median()),
                'std': float(durations.std()),
            }

        # Win rate by exit reason
        print(f"\n💰 Win Rate by Exit Reason:")
        for reason in exit_counts.index:
            trades = self.trades_baseline[self.trades_baseline['exit_reason'] == reason]
            wins = (trades['pnl'] > 0).sum()
            total = len(trades)
            win_rate = wins / total * 100 if total > 0 else 0
            avg_pnl = trades['pnl'].mean()
            print(f"  {reason:20s}: {wins}/{total} = {win_rate:>5.1f}% WR, Avg PnL: ${avg_pnl:>7.2f}")

        results['win_rate_by_exit'] = {}
        for reason in exit_counts.index:
            trades = self.trades_baseline[self.trades_baseline['exit_reason'] == reason]
            wins = (trades['pnl'] > 0).sum()
            total = len(trades)
            results['win_rate_by_exit'][reason] = {
                'wins': int(wins),
                'total': int(total),
                'win_rate': float(wins / total * 100 if total > 0 else 0),
                'avg_pnl': float(trades['pnl'].mean()),
            }

        # Plot analysis
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Exit reason pie chart
        axes[0, 0].pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Exit Reason Distribution')

        # Duration box plot
        duration_data = [
            self.trades_baseline[self.trades_baseline['exit_reason'] == reason]['duration_minutes'].values
            for reason in exit_counts.index
        ]
        axes[0, 1].boxplot(duration_data, labels=exit_counts.index, showmeans=True)
        axes[0, 1].set_ylabel('Duration (minutes)')
        axes[0, 1].set_title('Trade Duration by Exit Reason')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # Win rate by exit reason
        win_rates = [results['win_rate_by_exit'][reason]['win_rate'] for reason in exit_counts.index]
        axes[1, 0].bar(exit_counts.index, win_rates, edgecolor='black')
        axes[1, 0].set_ylabel('Win Rate (%)')
        axes[1, 0].set_title('Win Rate by Exit Reason')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% benchmark')
        axes[1, 0].legend()
        for i, (reason, wr) in enumerate(zip(exit_counts.index, win_rates)):
            axes[1, 0].text(i, wr + 2, f'{wr:.1f}%', ha='center', va='bottom')

        # PnL distribution by exit reason
        for reason in exit_counts.index:
            pnls = self.trades_baseline[self.trades_baseline['exit_reason'] == reason]['pnl']
            axes[1, 1].hist(pnls, bins=20, alpha=0.6, label=reason, edgecolor='black')
        axes[1, 1].axvline(x=0, color='black', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('PnL ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('PnL Distribution by Exit Reason')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = self.output_dir / 'task3_stop_loss_analysis.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Saved plot: {plot_path}")
        plt.close()

        return results

    def task5_data_quality_check(self) -> Dict:
        """Task #5: Verify data quality of Jan 2026 data"""
        print("\n" + "="*80)
        print("TASK #5: DATA QUALITY VERIFICATION")
        print("="*80)

        results = {'issues': [], 'warnings': []}

        # Load bars data
        bars_path = Path("/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/runs/databento_backtest_20260124_182118/bar_size=5m/bars.parquet")

        if not bars_path.exists():
            print(f"⚠️  Bars file not found: {bars_path}")
            results['issues'].append(f"Bars file not found: {bars_path}")
            return results

        bars = pd.read_parquet(bars_path)
        bars = bars.reset_index()  # Move timestamp from index to column
        bars['timestamp'] = pd.to_datetime(bars['timestamp'])

        print(f"\n📊 Dataset Overview:")
        print(f"  Total bars: {len(bars)}")
        print(f"  Date range: {bars['timestamp'].min()} to {bars['timestamp'].max()}")
        print(f"  Columns: {list(bars.columns)}")

        results['total_bars'] = len(bars)
        results['date_range'] = {
            'start': str(bars['timestamp'].min()),
            'end': str(bars['timestamp'].max()),
        }

        # Check for duplicates
        duplicates = bars.duplicated(subset=['timestamp']).sum()
        print(f"\n🔍 Duplicate Check:")
        print(f"  Duplicate timestamps: {duplicates}")
        if duplicates > 0:
            results['issues'].append(f"Found {duplicates} duplicate timestamps")
        else:
            print(f"  ✓ No duplicates found")

        # Check for gaps
        bars_sorted = bars.sort_values('timestamp')
        time_diffs = bars_sorted['timestamp'].diff()
        expected_freq = pd.Timedelta('5min')
        gaps = time_diffs[time_diffs > expected_freq * 2]  # Allow for weekends/overnight

        print(f"\n⏰ Timestamp Gap Check:")
        print(f"  Expected frequency: 5 minutes")
        print(f"  Large gaps (>10 min): {len(gaps)}")
        if len(gaps) > 0:
            print(f"  Largest gap: {gaps.max()}")
            results['warnings'].append(f"Found {len(gaps)} large gaps in data")
        else:
            print(f"  ✓ No unusual gaps")

        # OHLC validation
        print(f"\n📈 OHLC Validation:")
        ohlc_violations = 0

        # High >= Low
        high_low_violations = (bars['high'] < bars['low']).sum()
        print(f"  High < Low violations: {high_low_violations}")
        if high_low_violations > 0:
            results['issues'].append(f"Found {high_low_violations} High < Low violations")
            ohlc_violations += high_low_violations

        # Close within OHLC range
        close_violations = ((bars['close'] > bars['high']) | (bars['close'] < bars['low'])).sum()
        print(f"  Close outside [Low, High]: {close_violations}")
        if close_violations > 0:
            results['issues'].append(f"Found {close_violations} Close outside range violations")
            ohlc_violations += close_violations

        # Open within OHLC range
        open_violations = ((bars['open'] > bars['high']) | (bars['open'] < bars['low'])).sum()
        print(f"  Open outside [Low, High]: {open_violations}")
        if open_violations > 0:
            results['issues'].append(f"Found {open_violations} Open outside range violations")
            ohlc_violations += open_violations

        if ohlc_violations == 0:
            print(f"  ✓ All OHLC relationships valid")

        # Volume check
        print(f"\n📊 Volume Check:")
        zero_volume = (bars['volume'] == 0).sum()
        print(f"  Zero volume bars: {zero_volume} ({zero_volume/len(bars)*100:.2f}%)")
        if zero_volume > 0:
            results['warnings'].append(f"Found {zero_volume} bars with zero volume")

        # Price statistics
        print(f"\n💵 Price Statistics:")
        print(f"  Close Min:    ${bars['close'].min():.2f}")
        print(f"  Close Max:    ${bars['close'].max():.2f}")
        print(f"  Close Mean:   ${bars['close'].mean():.2f}")
        print(f"  Close Std:    ${bars['close'].std():.2f}")

        results['price_stats'] = {
            'min': float(bars['close'].min()),
            'max': float(bars['close'].max()),
            'mean': float(bars['close'].mean()),
            'std': float(bars['close'].std()),
        }

        # Outlier detection (returns > 5 std devs)
        bars_sorted['returns'] = bars_sorted['close'].pct_change()
        outlier_threshold = 5 * bars_sorted['returns'].std()
        outliers = bars_sorted[abs(bars_sorted['returns']) > outlier_threshold]

        print(f"\n⚠️  Outlier Check (|return| > 5 std):")
        print(f"  Outlier bars: {len(outliers)}")
        if len(outliers) > 0:
            print(f"  Max return: {bars_sorted['returns'].max():.4f}")
            print(f"  Min return: {bars_sorted['returns'].min():.4f}")
            results['warnings'].append(f"Found {len(outliers)} statistical outliers")
        else:
            print(f"  ✓ No extreme outliers")

        # Summary
        print(f"\n" + "="*80)
        if len(results['issues']) == 0 and len(results['warnings']) == 0:
            print("✅ DATA QUALITY: EXCELLENT - No issues or warnings")
            results['quality'] = 'excellent'
        elif len(results['issues']) == 0:
            print(f"⚠️  DATA QUALITY: GOOD - {len(results['warnings'])} warnings (no critical issues)")
            results['quality'] = 'good'
        else:
            print(f"❌ DATA QUALITY: POOR - {len(results['issues'])} critical issues")
            results['quality'] = 'poor'
        print("="*80)

        return results

    def generate_summary_report(self, task1_results: Dict, task3_results: Dict, task5_results: Dict):
        """Generate comprehensive diagnostic report"""
        print("\n" + "="*80)
        print("COMPREHENSIVE DIAGNOSTIC SUMMARY")
        print("="*80)

        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'task1_directional_bias': task1_results,
            'task3_stop_loss_analysis': task3_results,
            'task5_data_quality': task5_results,
            'critical_findings': [],
            'recommendations': []
        }

        # Critical findings
        print("\n🚨 CRITICAL FINDINGS:\n")

        # Finding 1: Directional bias
        long_pct = task1_results['direction_pct'].get('LONG', 0)
        short_pct = task1_results['direction_pct'].get('SHORT', 0)
        if long_pct > 90 or short_pct > 90:
            finding = f"SEVERE DIRECTIONAL BIAS: {long_pct:.1f}% LONG, {short_pct:.1f}% SHORT"
            print(f"  1. {finding}")
            report['critical_findings'].append(finding)
            report['recommendations'].append(
                "Investigate model scoring logic - check if features/model favor one direction"
            )

        # Finding 2: Stop-loss dominated
        stop_loss_pct = task3_results['exit_pct'].get('stop_loss', 0)
        if stop_loss_pct > 75:
            finding = f"EXCESSIVE STOP-LOSS HITS: {stop_loss_pct:.1f}% of trades"
            print(f"  2. {finding}")
            report['critical_findings'].append(finding)
            report['recommendations'].append(
                "Consider widening stop-loss (test 1.5x ATR instead of 1x) or reassess entry quality"
            )

        # Finding 3: Poor win rate
        for reason, stats in task3_results['win_rate_by_exit'].items():
            if stats['win_rate'] < 20:
                finding = f"CATASTROPHIC WIN RATE on {reason}: {stats['win_rate']:.1f}%"
                print(f"  3. {finding}")
                report['critical_findings'].append(finding)

        # Recommendations
        print("\n💡 RECOMMENDATIONS:\n")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")

        # Add additional recommendations
        additional_recs = [
            "Retrain model on recent data (Q4 2024 - Jan 2026) to capture regime change",
            "Implement regime detection to avoid trading in unfavorable conditions",
            "Backtest with alternative stop/target multipliers: 1.5x/2.5x, 1.25x/3x",
            "Investigate feature drift - compare feature distributions Dec 2024 vs Jan 2026",
            "Consider ensemble model with separate LONG/SHORT classifiers"
        ]

        for i, rec in enumerate(additional_recs, len(report['recommendations']) + 1):
            print(f"  {i}. {rec}")
            report['recommendations'].append(rec)

        # Save report
        report_path = self.output_dir / 'diagnostic_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n💾 Saved comprehensive report: {report_path}")

        return report


def main():
    """Run diagnostic analysis"""
    print("="*80)
    print("MODEL DEGRADATION DIAGNOSTIC")
    print("Issue: 58% → 13.7% win rate drop on Jan 2026 out-of-sample data")
    print("="*80)

    backtest_root = Path("/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/backtest_results/databento_validation_20260124_182118")
    output_dir = Path("/Users/eshaanganguly/Documents/projects/algos 3 topstep/ml_intraday_v3/diagnostics")

    diagnostic = ModelDegradationDiagnostic(backtest_root, output_dir)

    # Run tasks
    task1_results = diagnostic.task1_analyze_directional_bias()
    task3_results = diagnostic.task3_analyze_stop_loss_hits()
    task5_results = diagnostic.task5_data_quality_check()

    # Generate summary
    report = diagnostic.generate_summary_report(task1_results, task3_results, task5_results)

    print("\n" + "="*80)
    print("✅ DIAGNOSTIC COMPLETE")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Generated plots:")
    print(f"   - task1_directional_bias.png")
    print(f"   - task3_stop_loss_analysis.png")
    print(f"📄 Report: diagnostic_report.json")
    print("="*80)


if __name__ == "__main__":
    main()
