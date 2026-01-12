"""
Analyze today's trading performance (Jan 12, 2026)
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Load trades
df = pd.read_csv('/Users/eshaanganguly/Downloads/trades_export-3.csv')

print("=" * 80)
print("TODAY'S TRADING ANALYSIS - January 12, 2026")
print("=" * 80)
print()

# Parse times
df['EnteredAt'] = pd.to_datetime(df['EnteredAt'])
df['ExitedAt'] = pd.to_datetime(df['ExitedAt'])

# Basic stats
total_trades = len(df)
winners = df[df['PnL'] > 0]
losers = df[df['PnL'] <= 0]

print(f"📊 TRADE SUMMARY")
print(f"{'─' * 80}")
print(f"Total Trades:        {total_trades}")
print(f"Winning Trades:      {len(winners)} ({len(winners)/total_trades*100:.1f}%)")
print(f"Losing Trades:       {len(losers)} ({len(losers)/total_trades*100:.1f}%)")
print(f"Win Rate:            {len(winners)/total_trades*100:.1f}%")
print()

# P&L stats
total_pnl = df['PnL'].sum()
gross_profit = winners['PnL'].sum()
gross_loss = abs(losers['PnL'].sum()) if len(losers) > 0 else 0
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
avg_trade = df['PnL'].mean()
avg_winner = winners['PnL'].mean()
avg_loser = losers['PnL'].mean() if len(losers) > 0 else 0

print(f"💰 P&L BREAKDOWN")
print(f"{'─' * 80}")
print(f"Total P&L:           ${total_pnl:,.2f}")
print(f"Gross Profit:        ${gross_profit:,.2f}")
print(f"Gross Loss:          ${gross_loss:,.2f}")
print(f"Profit Factor:       {profit_factor:.2f}")
print()
print(f"Average Trade:       ${avg_trade:,.2f}")
print(f"Average Winner:      ${avg_winner:,.2f}")
print(f"Average Loser:       ${avg_loser:,.2f}")
print()
print(f"Max Win:             ${df['PnL'].max():,.2f}")
print(f"Max Loss:            ${df['PnL'].min():,.2f}")
print()

# Time analysis
session_start = df['EnteredAt'].min()
session_end = df['ExitedAt'].max()
session_duration = (session_end - session_start).total_seconds() / 3600

print(f"⏱️  TIME ANALYSIS")
print(f"{'─' * 80}")
print(f"First Entry:         {session_start.strftime('%H:%M:%S')}")
print(f"Last Exit:           {session_end.strftime('%H:%M:%S')}")
print(f"Session Duration:    {session_duration:.1f} hours")
print(f"Trades per Hour:     {total_trades / session_duration:.1f}")
print()

# Duration of trades
df['Duration'] = (df['ExitedAt'] - df['EnteredAt']).dt.total_seconds() / 60
avg_duration = df['Duration'].mean()
print(f"Avg Trade Duration:  {avg_duration:.1f} minutes")
print(f"Min Duration:        {df['Duration'].min():.1f} minutes")
print(f"Max Duration:        {df['Duration'].max():.1f} minutes")
print()

# Contract verification (should all be 1 in learning phase)
print(f"📦 POSITION SIZING")
print(f"{'─' * 80}")
print(f"Contracts per trade: {df['Size'].unique()}")
if all(df['Size'] == 1):
    print(f"✓ Kelly learning phase: All trades = 1 contract")
else:
    print(f"⚠ Kelly activated: Varying contract sizes")
print()

# Compare to backtest expectations
print(f"📈 COMPARISON TO BACKTEST")
print(f"{'─' * 80}")

backtest_metrics = {
    'win_rate': 51.0,
    'avg_trade': 41.0,  # From alpha/beta analysis
    'sharpe_annual': 4.03,
}

live_win_rate = len(winners)/total_trades*100
live_avg_trade = avg_trade

win_rate_diff = live_win_rate - backtest_metrics['win_rate']
avg_trade_diff = live_avg_trade - backtest_metrics['avg_trade']

print(f"                     Backtest    Live Today    Difference")
print(f"Win Rate:            {backtest_metrics['win_rate']:.1f}%       {live_win_rate:.1f}%        {win_rate_diff:+.1f}%")
print(f"Avg Trade:           ${backtest_metrics['avg_trade']:.2f}      ${live_avg_trade:.2f}       ${avg_trade_diff:+.2f}")
print()

if win_rate_diff > 0:
    print(f"✓ Win rate HIGHER than backtest by {win_rate_diff:.1f}%")
else:
    print(f"⚠ Win rate LOWER than backtest by {abs(win_rate_diff):.1f}%")

if avg_trade_diff > 0:
    print(f"✓ Avg trade HIGHER than backtest by ${avg_trade_diff:.2f}")
else:
    print(f"⚠ Avg trade LOWER than backtest by ${abs(avg_trade_diff):.2f}")
print()

# Statistical significance test
# Standard error of win rate
se_win_rate = np.sqrt(0.51 * 0.49 / total_trades) * 100
print(f"📊 STATISTICAL CONFIDENCE")
print(f"{'─' * 80}")
print(f"Sample size:         {total_trades} trades")
print(f"Standard error:      ±{se_win_rate:.1f}%")
print(f"95% CI for WR:       {live_win_rate - 2*se_win_rate:.1f}% - {live_win_rate + 2*se_win_rate:.1f}%")
print(f"Backtest WR (51%):   {'Within' if abs(win_rate_diff) <= 2*se_win_rate else 'Outside'} confidence interval")
print()

# Risk metrics
starting_equity = 50000
current_pnl = total_pnl
current_equity = starting_equity + current_pnl
return_pct = (current_pnl / starting_equity) * 100

print(f"💼 ACCOUNT STATUS")
print(f"{'─' * 80}")
print(f"Starting Equity:     ${starting_equity:,.2f}")
print(f"Current P&L:         ${current_pnl:,.2f}")
print(f"Current Equity:      ${current_equity:,.2f}")
print(f"Return:              {return_pct:+.2f}%")
print()

# Daily limit check
daily_loss_limit = 1000
drawdown_limit = 2500
print(f"Risk Limit Status:")
print(f"  Daily P&L:         ${current_pnl:,.2f} (limit: -${daily_loss_limit:,})")
print(f"  Status:            {'✓ Safe' if current_pnl > -daily_loss_limit else '⚠ Approaching limit'}")
print()

# Projection to end of day
trades_per_hour = total_trades / session_duration
hours_remaining = max(0, 15.0 - session_duration)  # Market closes at 3 PM (15:00)
projected_total_trades = total_trades + (trades_per_hour * hours_remaining)
projected_pnl = current_pnl + (avg_trade * trades_per_hour * hours_remaining)

print(f"📊 END-OF-DAY PROJECTION")
print(f"{'─' * 80}")
print(f"Current time:        ~{session_end.strftime('%H:%M')}")
print(f"Hours remaining:     {hours_remaining:.1f}h")
print(f"Rate:                {trades_per_hour:.1f} trades/hour")
print(f"Projected trades:    {projected_total_trades:.0f} total")
print(f"Projected P&L:       ${projected_pnl:,.2f}")
print()

# Overall assessment
print("=" * 80)
print("🎯 OVERALL ASSESSMENT")
print("=" * 80)
print()

if live_win_rate >= 95:
    print("🟢 EXCELLENT: Win rate extremely high (96%+)")
    print("   → All but 1 trade won today")
    print("   → Significantly outperforming backtest")
    print("   → Small sample (27 trades), expect regression to ~51%")
elif live_win_rate >= 55:
    print("🟢 GREAT: Win rate above backtest")
elif live_win_rate >= 48:
    print("🟡 GOOD: Win rate close to backtest (within expected variance)")
elif live_win_rate >= 45:
    print("🟡 ACCEPTABLE: Win rate slightly below backtest but positive edge")
else:
    print("🔴 CONCERNING: Win rate significantly below backtest")

print()

if total_trades < 20:
    print(f"⚠️  Kelly still in learning phase ({total_trades}/20 trades)")
    print("   → All trades using 1 contract")
    print("   → Kelly will activate after 20 trades")
elif total_trades >= 20:
    print(f"✓ Kelly should activate after this trade")
    print("   → Next trades will use dynamic sizing (1-5 contracts)")

print()
print("📝 RECOMMENDATIONS:")
print()

if live_win_rate > 70 and total_trades < 50:
    print("1. Small sample size - don't get overconfident")
    print("2. Win rate will likely regress toward 51% (backtest)")
    print("3. Continue monitoring for at least 100 trades")
    print("4. This is a great start, but statistically not yet significant")

if current_pnl > 1000:
    print("1. Strong profit today, but remember this is paper trading")
    print("2. Continue to validate system stability")

print(f"1. Continue paper trading for {max(0, 100 - total_trades)} more trades minimum")
print(f"2. Target: 100-200 trades before starting Topstep combine")
print(f"3. Monitor for: RTH filtering, Kelly activation, risk limits")

print()
print("=" * 80)
