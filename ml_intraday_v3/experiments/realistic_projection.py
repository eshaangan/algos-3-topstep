#!/usr/bin/env python3
"""
More Realistic Jan 2026 Projection with Full Improvements

Accounts for:
1. Better entry timing (2-3 ticks improvement)
2. Dynamic stops (better R:R ratio)
3. Tiered position sizing (bigger on high confidence)
4. Ensemble/feature improvements (+5-10% win rate)
"""

# Jan 2026 baseline
BASELINE = {
    'trades': 152,
    'days': 18,
    'win_rate': 0.355,
    'total_pnl': -884.73,
    'avg_win': 41.16,  # From actual data
    'avg_loss': -19.98,  # From actual data
}


def calculate_improved_metrics(
    trades_per_day: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    days: int = 18
) -> dict:
    """Calculate P&L with given parameters."""

    total_trades = trades_per_day * days
    total_wins = total_trades * win_rate
    total_losses = total_trades * (1 - win_rate)

    total_pnl = (total_wins * avg_win) + (total_losses * avg_loss)
    avg_trade = total_pnl / total_trades if total_trades > 0 else 0
    daily_pnl = total_pnl / days

    days_to_3000 = 3000 / daily_pnl if daily_pnl > 0 else float('inf')

    return {
        'total_trades': total_trades,
        'trades_per_day': trades_per_day,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'total_pnl': total_pnl,
        'avg_trade': avg_trade,
        'daily_pnl': daily_pnl,
        'days_to_3000': days_to_3000,
        'improvement': total_pnl - BASELINE['total_pnl']
    }


print("=" * 80)
print("REALISTIC JAN 2026 PROJECTION WITH ALL IMPROVEMENTS")
print("=" * 80)

print("\n📊 BASELINE (Actual Jan 2026)")
print("-" * 80)
baseline_calc = calculate_improved_metrics(
    BASELINE['trades'] / BASELINE['days'],
    BASELINE['win_rate'],
    BASELINE['avg_win'],
    BASELINE['avg_loss'],
    BASELINE['days']
)
print(f"Trades/Day:       {baseline_calc['trades_per_day']:.1f}")
print(f"Win Rate:         {baseline_calc['win_rate']:.1%}")
print(f"Avg Win:          ${baseline_calc['avg_win']:.2f}")
print(f"Avg Loss:         ${baseline_calc['avg_loss']:.2f}")
print(f"Total P&L:        ${baseline_calc['total_pnl']:.2f}")
print(f"Daily P&L:        ${baseline_calc['daily_pnl']:.2f}")

# SCENARIO 1: Conservative (minimal improvements)
print("\n" + "=" * 80)
print("SCENARIO 1: CONSERVATIVE")
print("=" * 80)
print("Improvements:")
print("  - Confidence filter 0.55 → reduce trades to 6/day, +10% win rate")
print("  - Entry timing → +$5 avg win, -$3 avg loss")
print("  - Dynamic stops → Better R:R")
print("-" * 80)

conservative = calculate_improved_metrics(
    trades_per_day=6.0,
    win_rate=0.355 + 0.10,  # 45.5%
    avg_win=BASELINE['avg_win'] + 5,  # $46.16
    avg_loss=BASELINE['avg_loss'] + 3,  # -$16.98
    days=18
)

print(f"Trades/Day:       {conservative['trades_per_day']:.1f}")
print(f"Win Rate:         {conservative['win_rate']:.1%}")
print(f"Avg Win:          ${conservative['avg_win']:.2f}")
print(f"Avg Loss:         ${conservative['avg_loss']:.2f}")
print(f"Total P&L:        ${conservative['total_pnl']:.2f} ({conservative['improvement']:+.2f})")
print(f"Daily P&L:        ${conservative['daily_pnl']:.2f}")
print(f"Days to $3,000:   {conservative['days_to_3000']:.1f} days")

# SCENARIO 2: Realistic (expected improvements)
print("\n" + "=" * 80)
print("SCENARIO 2: REALISTIC (Recommended)")
print("=" * 80)
print("Improvements:")
print("  - Confidence filter 0.55 → 7 quality trades/day, +15% win rate")
print("  - Entry timing → +$8 avg win, -$5 avg loss")
print("  - Dynamic stops → Better R:R by 20%")
print("  - Tiered sizing → Bigger positions on high confidence")
print("-" * 80)

realistic = calculate_improved_metrics(
    trades_per_day=7.0,
    win_rate=0.355 + 0.15,  # 50.5%
    avg_win=BASELINE['avg_win'] + 8,  # $49.16
    avg_loss=BASELINE['avg_loss'] + 5,  # -$14.98
    days=18
)

print(f"Trades/Day:       {realistic['trades_per_day']:.1f}")
print(f"Win Rate:         {realistic['win_rate']:.1%}")
print(f"Avg Win:          ${realistic['avg_win']:.2f}")
print(f"Avg Loss:         ${realistic['avg_loss']:.2f}")
print(f"Total P&L:        ${realistic['total_pnl']:.2f} ({realistic['improvement']:+.2f})")
print(f"Daily P&L:        ${realistic['daily_pnl']:.2f}")
print(f"Days to $3,000:   {realistic['days_to_3000']:.1f} days")

if realistic['days_to_3000'] <= 20:
    print("✅ PASSES COMBINE TIMELINE")
elif realistic['days_to_3000'] <= 30:
    print("⚠️ ACCEPTABLE BUT SLOWER")
else:
    print("❌ TOO SLOW")

# SCENARIO 3: Optimistic (best case)
print("\n" + "=" * 80)
print("SCENARIO 3: OPTIMISTIC")
print("=" * 80)
print("Improvements:")
print("  - All realistic improvements PLUS")
print("  - Model ensemble → +18% win rate")
print("  - Feature improvements → +$12 avg win")
print("  - Perfect execution → -$8 avg loss")
print("-" * 80)

optimistic = calculate_improved_metrics(
    trades_per_day=7.5,
    win_rate=0.355 + 0.18,  # 53.5%
    avg_win=BASELINE['avg_win'] + 12,  # $53.16
    avg_loss=BASELINE['avg_loss'] + 8,  # -$11.98
    days=18
)

print(f"Trades/Day:       {optimistic['trades_per_day']:.1f}")
print(f"Win Rate:         {optimistic['win_rate']:.1%}")
print(f"Avg Win:          ${optimistic['avg_win']:.2f}")
print(f"Avg Loss:         ${optimistic['avg_loss']:.2f}")
print(f"Total P&L:        ${optimistic['total_pnl']:.2f} ({optimistic['improvement']:+.2f})")
print(f"Daily P&L:        ${optimistic['daily_pnl']:.2f}")
print(f"Days to $3,000:   {optimistic['days_to_3000']:.1f} days")

if optimistic['days_to_3000'] <= 20:
    print("✅ PASSES COMBINE TIMELINE")

# Summary table
print("\n" + "=" * 80)
print("SUMMARY COMPARISON")
print("=" * 80)
print(f"{'Scenario':<15} {'Trades/Day':<12} {'Win Rate':<10} {'Daily P&L':<12} {'Days to $3k':<12}")
print("-" * 80)
print(f"{'Baseline':<15} {baseline_calc['trades_per_day']:<12.1f} {baseline_calc['win_rate']:<10.1%} ${baseline_calc['daily_pnl']:<11.2f} {'NEVER':<12}")
print(f"{'Conservative':<15} {conservative['trades_per_day']:<12.1f} {conservative['win_rate']:<10.1%} ${conservative['daily_pnl']:<11.2f} {conservative['days_to_3000']:<12.1f}")
print(f"{'Realistic':<15} {realistic['trades_per_day']:<12.1f} {realistic['win_rate']:<10.1%} ${realistic['daily_pnl']:<11.2f} {realistic['days_to_3000']:<12.1f}")
print(f"{'Optimistic':<15} {optimistic['trades_per_day']:<12.1f} {optimistic['win_rate']:<10.1%} ${optimistic['daily_pnl']:<11.2f} {optimistic['days_to_3000']:<12.1f}")

print("\n" + "=" * 80)
print("🎯 RECOMMENDED PATH: REALISTIC SCENARIO")
print("=" * 80)
print(f"\nWith threshold=0.55 + all improvements:")
print(f"  • Jan 2026 would be: ${realistic['total_pnl']:,.2f} (vs -$884.73 actual)")
print(f"  • Daily P&L: ${realistic['daily_pnl']:.2f}")
print(f"  • Pass combine in: {realistic['days_to_3000']:.0f} days")
print(f"\nThis requires implementing ALL improvements:")
print(f"  1. Confidence filter (0.55)")
print(f"  2. Entry timing optimization")
print(f"  3. Dynamic stops")
print(f"  4. Tiered position sizing")
print(f"  5. Signal quality improvements")
print("\n⚠️  These are PROJECTIONS. Must validate with backtest!")
print("=" * 80)
