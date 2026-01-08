"""
Analyze backtest results and determine if ready for live trading.

Usage:
    python analyze_backtest.py path/to/backtest_results.csv

Or if you have results in notebook, just copy the summary stats here:
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

def analyze_backtest_results(results_df=None, summary_stats=None):
    """
    Analyze backtest results and provide detailed feedback.

    Args:
        results_df: DataFrame with trade-by-trade results
        summary_stats: Dict with summary statistics if you don't have the full CSV
    """

    print("=" * 80)
    print("BACKTEST RESULTS ANALYSIS")
    print("=" * 80)
    print()

    if summary_stats:
        # User provided summary stats
        return analyze_summary(summary_stats)

    if results_df is None:
        print("📋 MANUAL INPUT MODE")
        print("=" * 80)
        print("\nPaste your backtest results here:")
        print()

        stats = {}
        stats['total_trades'] = int(input("Total Trades: "))
        stats['win_rate'] = float(input("Win Rate (%): "))
        stats['total_pnl'] = float(input("Total P&L ($): "))
        stats['profit_factor'] = float(input("Profit Factor: "))
        stats['max_drawdown'] = float(input("Max Drawdown ($): "))
        stats['sharpe_ratio'] = float(input("Sharpe Ratio: "))

        print()
        return analyze_summary(stats)

    # Analyze from full DataFrame
    return analyze_full_results(results_df)


def analyze_summary(stats):
    """Analyze from summary statistics."""

    print("\n" + "=" * 80)
    print("📊 YOUR RESULTS")
    print("=" * 80)
    print(f"Total Trades:    {stats['total_trades']:,}")
    print(f"Win Rate:        {stats['win_rate']:.1f}%")
    print(f"Total P&L:       ${stats['total_pnl']:,.2f}")
    print(f"Profit Factor:   {stats['profit_factor']:.2f}")
    print(f"Max Drawdown:    ${stats['max_drawdown']:,.2f}")
    print(f"Sharpe Ratio:    {stats['sharpe_ratio']:.2f}")

    # Scoring
    score = 0
    max_score = 0
    issues = []
    strengths = []

    print("\n" + "=" * 80)
    print("🎯 DETAILED ANALYSIS")
    print("=" * 80)

    # 1. Profitability
    print("\n1️⃣  PROFITABILITY")
    max_score += 20
    if stats['total_pnl'] > 10000:
        print(f"   ✅ Excellent P&L: ${stats['total_pnl']:,.2f}")
        score += 20
        strengths.append("Strong profitability")
    elif stats['total_pnl'] > 5000:
        print(f"   ✅ Good P&L: ${stats['total_pnl']:,.2f}")
        score += 15
        strengths.append("Profitable")
    elif stats['total_pnl'] > 0:
        print(f"   ⚠️  Marginal P&L: ${stats['total_pnl']:,.2f}")
        score += 8
        issues.append("Low profitability - might not cover slippage/fees in live")
    else:
        print(f"   ❌ Negative P&L: ${stats['total_pnl']:,.2f}")
        issues.append("CRITICAL: Strategy is losing money")

    # 2. Win Rate
    print("\n2️⃣  WIN RATE")
    max_score += 15
    if 50 <= stats['win_rate'] <= 65:
        print(f"   ✅ Optimal range: {stats['win_rate']:.1f}%")
        score += 15
        strengths.append("Balanced win rate")
    elif 45 <= stats['win_rate'] < 50 or 65 < stats['win_rate'] <= 70:
        print(f"   ✅ Good: {stats['win_rate']:.1f}%")
        score += 12
    elif 40 <= stats['win_rate'] < 45 or 70 < stats['win_rate'] <= 75:
        print(f"   ⚠️  Acceptable: {stats['win_rate']:.1f}%")
        score += 8
        if stats['win_rate'] < 45:
            issues.append("Win rate a bit low - need larger winners")
        else:
            issues.append("Win rate high - might be curve-fitted")
    else:
        print(f"   ❌ Outside normal range: {stats['win_rate']:.1f}%")
        score += 3
        if stats['win_rate'] < 40:
            issues.append("CRITICAL: Win rate too low")
        else:
            issues.append("CRITICAL: Win rate suspiciously high (>75%)")

    # 3. Profit Factor
    print("\n3️⃣  PROFIT FACTOR")
    max_score += 20
    if stats['profit_factor'] >= 2.0:
        print(f"   ✅ Excellent: {stats['profit_factor']:.2f}")
        score += 20
        strengths.append("Strong profit factor")
    elif stats['profit_factor'] >= 1.5:
        print(f"   ✅ Good: {stats['profit_factor']:.2f}")
        score += 15
        strengths.append("Good profit factor")
    elif stats['profit_factor'] >= 1.2:
        print(f"   ⚠️  Marginal: {stats['profit_factor']:.2f}")
        score += 8
        issues.append("Profit factor a bit low")
    else:
        print(f"   ❌ Too low: {stats['profit_factor']:.2f}")
        issues.append("CRITICAL: Profit factor < 1.2 (barely profitable)")

    # 4. Sharpe Ratio
    print("\n4️⃣  SHARPE RATIO")
    max_score += 15
    if stats['sharpe_ratio'] >= 2.0:
        print(f"   ✅ Excellent: {stats['sharpe_ratio']:.2f}")
        score += 15
        strengths.append("Excellent risk-adjusted returns")
    elif stats['sharpe_ratio'] >= 1.0:
        print(f"   ✅ Good: {stats['sharpe_ratio']:.2f}")
        score += 12
        strengths.append("Good risk-adjusted returns")
    elif stats['sharpe_ratio'] >= 0.5:
        print(f"   ⚠️  Acceptable: {stats['sharpe_ratio']:.2f}")
        score += 6
    else:
        print(f"   ❌ Too low: {stats['sharpe_ratio']:.2f}")
        issues.append("Poor risk-adjusted returns")

    # 5. Max Drawdown (CRITICAL for Topstep)
    print("\n5️⃣  MAX DRAWDOWN (Topstep Limit: $2,500)")
    max_score += 30
    if stats['max_drawdown'] < 1000:
        print(f"   ✅ Excellent: ${stats['max_drawdown']:,.2f} (40% of limit)")
        score += 30
        strengths.append("Excellent drawdown control")
    elif stats['max_drawdown'] < 1500:
        print(f"   ✅ Good: ${stats['max_drawdown']:,.2f} (60% of limit)")
        score += 25
        strengths.append("Good drawdown control")
    elif stats['max_drawdown'] < 2000:
        print(f"   ⚠️  Acceptable: ${stats['max_drawdown']:,.2f} (80% of limit)")
        score += 15
        issues.append("Drawdown getting close to limit")
    elif stats['max_drawdown'] < 2500:
        print(f"   ⚠️  DANGER: ${stats['max_drawdown']:,.2f} (>80% of limit)")
        score += 5
        issues.append("WARNING: Drawdown very close to Topstep limit")
    else:
        print(f"   ❌ VIOLATED: ${stats['max_drawdown']:,.2f} (>100% of limit)")
        issues.append("CRITICAL: Would have been kicked out of Topstep!")

    # 6. Sample Size
    print("\n6️⃣  SAMPLE SIZE")
    max_score += 10
    if stats['total_trades'] >= 500:
        print(f"   ✅ Excellent: {stats['total_trades']:,} trades")
        score += 10
        strengths.append("High statistical significance")
    elif stats['total_trades'] >= 200:
        print(f"   ✅ Good: {stats['total_trades']:,} trades")
        score += 8
        strengths.append("Good sample size")
    elif stats['total_trades'] >= 100:
        print(f"   ⚠️  Acceptable: {stats['total_trades']:,} trades")
        score += 5
        issues.append("Sample size a bit small")
    else:
        print(f"   ❌ Too small: {stats['total_trades']:,} trades")
        issues.append("CRITICAL: Not enough trades for statistical significance")

    # Final Grade
    percentage = (score / max_score) * 100

    print("\n" + "=" * 80)
    print("📈 OVERALL GRADE")
    print("=" * 80)
    print(f"\nScore: {score}/{max_score} ({percentage:.1f}%)")
    print()

    if percentage >= 85:
        grade = "A (EXCELLENT)"
        emoji = "🟢"
        verdict = "READY FOR LIVE TRADING"
    elif percentage >= 70:
        grade = "B (GOOD)"
        emoji = "🟢"
        verdict = "READY FOR LIVE TRADING with minor caution"
    elif percentage >= 55:
        grade = "C (ACCEPTABLE)"
        emoji = "🟡"
        verdict = "PROCEED WITH CAUTION - Start small"
    elif percentage >= 40:
        grade = "D (MARGINAL)"
        emoji = "🟡"
        verdict = "NOT RECOMMENDED - Needs improvement"
    else:
        grade = "F (FAILING)"
        emoji = "🔴"
        verdict = "DO NOT TRADE LIVE - Major issues"

    print(f"{emoji} Grade: {grade}")
    print(f"{emoji} Verdict: {verdict}")

    # Strengths
    if strengths:
        print("\n✅ STRENGTHS:")
        for strength in strengths:
            print(f"   • {strength}")

    # Issues
    if issues:
        print("\n⚠️  AREAS OF CONCERN:")
        for issue in issues:
            print(f"   • {issue}")

    # Recommendations
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    if percentage >= 70:
        print("\n✅ This strategy looks solid for live trading!")
        print("\nNext steps:")
        print("   1. Review trade-by-trade results for any anomalies")
        print("   2. Check equity curve for smooth growth")
        print("   3. Verify no single day violates Topstep rules")
        print("   4. Start with paper trading to confirm")
        print("   5. Monitor closely in first week of live trading")
    elif percentage >= 55:
        print("\n⚠️  Strategy has potential but needs monitoring:")
        print("\nNext steps:")
        print("   1. Start with 1 contract only")
        print("   2. Paper trade for at least 1-2 weeks first")
        print("   3. Watch drawdown closely")
        print("   4. Consider tightening stop losses")
        print("   5. Re-evaluate after 50 live trades")
    else:
        print("\n❌ Strategy needs improvement before live trading:")
        print("\nSuggested improvements:")
        print("   1. Retrain model with different hyperparameters")
        print("   2. Add/remove features to improve predictions")
        print("   3. Adjust stop/target multipliers")
        print("   4. Review feature engineering")
        print("   5. Check for data leakage or overfitting")

    print("\n" + "=" * 80)

    return score, percentage, verdict


def analyze_full_results(df):
    """Analyze from full trade-by-trade results."""

    print(f"\n📊 Analyzing {len(df):,} trades...")

    # Calculate metrics
    total_pnl = df['pnl'].sum()
    win_rate = (df['pnl'] > 0).mean() * 100

    gross_profit = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    cumulative_pnl = df['pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()

    returns = df['pnl']
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0

    stats = {
        'total_trades': len(df),
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe
    }

    return analyze_summary(stats)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Load from CSV
        csv_path = sys.argv[1]
        print(f"Loading results from: {csv_path}")
        df = pd.read_csv(csv_path)
        analyze_full_results(df)
    else:
        # Manual input
        analyze_backtest_results()
