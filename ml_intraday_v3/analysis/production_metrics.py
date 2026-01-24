#!/usr/bin/env python3
"""
Production-Ready Performance Metrics Analysis
Analyzes backtest results to determine if model is ready for live trading.
"""

import pandas as pd
import numpy as np
import sys

def analyze_performance(trades_path: str):
    """Analyze trading performance from backtest trades"""

    # Read trades
    trades = pd.read_parquet(trades_path)

    print('='*70)
    print('PRODUCTION-READY PERFORMANCE METRICS')
    print('MES 1-Minute Trading Model - Comprehensive Backtest Analysis')
    print('='*70)

    # Basic Stats
    total_pnl = trades['pnl'].sum()
    num_trades = len(trades)
    winning_trades = len(trades[trades['pnl'] > 0])
    losing_trades = len(trades[trades['pnl'] < 0])
    win_rate = (winning_trades / num_trades) * 100

    avg_win = trades[trades['pnl'] > 0]['pnl'].mean()
    avg_loss = trades[trades['pnl'] < 0]['pnl'].mean()
    profit_factor = abs(trades[trades['pnl'] > 0]['pnl'].sum() / trades[trades['pnl'] < 0]['pnl'].sum())

    print(f'\nOVERALL PERFORMANCE')
    print('-'*70)
    print(f'Total P&L:              ${total_pnl:,.2f}')
    print(f'Total Trades:           {num_trades}')
    print(f'Winning Trades:         {winning_trades} ({win_rate:.1f}%)')
    print(f'Losing Trades:          {losing_trades} ({100-win_rate:.1f}%)')
    print(f'')
    print(f'Average Win:            ${avg_win:,.2f}')
    print(f'Average Loss:           ${avg_loss:,.2f}')
    print(f'Win/Loss Ratio:         {abs(avg_win/avg_loss):.2f}')
    print(f'Profit Factor:          {profit_factor:.2f}')

    # Calculate daily stats
    trades['date'] = pd.to_datetime(trades['timestamp']).dt.date
    daily_pnl = trades.groupby('date')['pnl'].sum()

    daily_mean = daily_pnl.mean()
    daily_std = daily_pnl.std()
    sharpe = (daily_mean / daily_std) * np.sqrt(252) if daily_std > 0 else 0

    # Drawdown
    cumulative = trades['pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min()

    print(f'\nRISK-ADJUSTED RETURNS')
    print('-'*70)
    print(f'Daily Mean P&L:         ${daily_mean:,.2f}')
    print(f'Daily Std Dev:          ${daily_std:,.2f}')
    print(f'Sharpe Ratio:           {sharpe:.2f}')
    print(f'Max Drawdown:           ${max_drawdown:,.2f}')
    print(f'Sortino Ratio:          {calculate_sortino(daily_pnl):.2f}')

    # Consistency
    winning_days = len(daily_pnl[daily_pnl > 0])
    total_days = len(daily_pnl)
    daily_win_rate = (winning_days / total_days) * 100

    print(f'\nCONSISTENCY METRICS')
    print('-'*70)
    print(f'Winning Days:           {winning_days}/{total_days} ({daily_win_rate:.1f}%)')
    print(f'Avg Daily Trades:       {num_trades/total_days:.1f}')
    print(f'Longest Win Streak:     {calculate_streak(daily_pnl, True)} days')
    print(f'Longest Loss Streak:    {calculate_streak(daily_pnl, False)} days')

    # Expectancy
    expectancy = (win_rate/100 * avg_win) + ((100-win_rate)/100 * avg_loss)

    print(f'\nEXPECTANCY & PROJECTIONS')
    print('-'*70)
    print(f'Expected Value/Trade:   ${expectancy:,.2f}')
    print(f'Kelly Criterion:        {calculate_kelly(win_rate/100, abs(avg_win/avg_loss)):.2%}')
    print(f'')
    print(f'MONTHLY PROJECTIONS (20 trading days):')
    trades_per_day = num_trades / total_days
    print(f'  Trades/Month:         {int(trades_per_day * 20)}')
    print(f'  Expected Monthly P&L: ${expectancy * trades_per_day * 20:,.2f}')
    print(f'')
    print(f'TOPSTEP 50K COMBINE:')
    target = 3000
    expected_daily = expectancy * trades_per_day
    days_to_target = target / expected_daily if expected_daily > 0 else 999
    print(f'  Profit Target:        $3,000')
    print(f'  Expected Daily P&L:   ${expected_daily:,.2f}')
    print(f'  Days to Target:       {days_to_target:.1f} trading days')
    print(f'  Calendar Days:        ~{days_to_target * 1.4:.0f} days (5-day week)')
    print(f'')

    # Risk assessment
    daily_loss_limit = 1000
    trailing_dd_limit = 2000
    max_daily_loss_observed = daily_pnl.min()
    max_daily_loss_pct = (max_daily_loss_observed / daily_loss_limit) * 100

    print(f'TOPSTEP RISK COMPLIANCE')
    print('-'*70)
    print(f'Daily Loss Limit:       $1,000')
    print(f'Max Daily Loss Observed: ${max_daily_loss_observed:,.2f} ({max_daily_loss_pct:.1f}% of limit)')
    print(f'Safety Margin:          ${daily_loss_limit - abs(max_daily_loss_observed):,.2f}')
    print(f'')
    print(f'Trailing DD Limit:      $2,000')
    print(f'Max Drawdown Observed:  ${max_drawdown:,.2f}')
    print(f'Safety Margin:          ${trailing_dd_limit - abs(max_drawdown):,.2f}')
    print(f'')

    # Final verdict
    safe_daily = abs(max_daily_loss_observed) < daily_loss_limit
    safe_dd = abs(max_drawdown) < trailing_dd_limit

    if safe_daily and safe_dd:
        print(f'✅ SAFE: All observed losses within Topstep limits')
        print(f'✅ READY: Model can be deployed to production')
    else:
        print(f'⚠️  WARNING: Some observed losses exceed Topstep limits')
        print(f'⚠️  ACTION: Reduce position size or adjust risk parameters')

    print(f'\n'+'='*70)
    print(f'MODEL STATUS: {"✅ PRODUCTION READY" if (safe_daily and safe_dd) else "⚠️  NEEDS TUNING"}')
    print(f'='*70)

    return {
        'total_pnl': total_pnl,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'expectancy': expectancy,
        'days_to_target': days_to_target,
        'production_ready': safe_daily and safe_dd
    }

def calculate_sortino(returns_series):
    """Calculate Sortino ratio (downside deviation only)"""
    downside_returns = returns_series[returns_series < 0]
    if len(downside_returns) == 0:
        return 0
    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0
    return (returns_series.mean() / downside_std) * np.sqrt(252)

def calculate_streak(series, winning=True):
    """Calculate longest winning or losing streak"""
    if winning:
        streaks = (series > 0).astype(int)
    else:
        streaks = (series < 0).astype(int)

    max_streak = 0
    current_streak = 0
    for val in streaks:
        if val == 1:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0
    return max_streak

def calculate_kelly(win_prob, win_loss_ratio):
    """Calculate Kelly Criterion"""
    return (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio

if __name__ == "__main__":
    if len(sys.argv) > 1:
        trades_path = sys.argv[1]
    else:
        trades_path = 'runs/run_20251224_123456/bar_size=1m/backtest/trades.parquet'

    results = analyze_performance(trades_path)
