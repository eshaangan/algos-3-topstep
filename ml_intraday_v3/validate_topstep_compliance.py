import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime

def analyze_compliance(file_path):
    print(f"\nAnalyzing {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)
    
    if df.empty:
        print("  No trades found.")
        return

    # Ensure datetime
    if 'exit_time' in df.columns:
        df['exit_time'] = pd.to_datetime(df['exit_time'])
        df = df.sort_values('exit_time')
    
    # Metrics
    if 'pnl_usd' in df.columns:
        pnl_col = 'pnl_usd'
    elif 'pnl' in df.columns:
        pnl_col = 'pnl'
    else:
        print("  No PnL column found.")
        return

    total_trades = len(df)
    total_pnl = df[pnl_col].sum()
    win_rate = (df[pnl_col] > 0).mean() * 100
    avg_trade = df[pnl_col].mean()
    
    gross_profit = df[df[pnl_col] > 0][pnl_col].sum()
    gross_loss = abs(df[df[pnl_col] < 0][pnl_col].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Topstep Compliance
    # Daily Loss Limit
    df['date'] = df['exit_time'].dt.date
    daily_pnl = df.groupby('date')[pnl_col].sum()
    
    min_daily_pnl = daily_pnl.min()
    max_daily_pnl = daily_pnl.max()
    days_below_limit = daily_pnl[daily_pnl < -1000]
    
    # Drawdown
    df['cumulative_pnl'] = df[pnl_col].cumsum()
    df['high_water_mark'] = df['cumulative_pnl'].cummax()
    df['drawdown'] = df['cumulative_pnl'] - df['high_water_mark']
    max_drawdown = df['drawdown'].min()

    # Sharpe / Sortino (assuming daily returns for simplicity, or per trade)
    # Using per-trade for now as an approximation, but strictly should be time-based.
    # Let's use daily PnL for Sharpe/Sortino to be more standard
    daily_std = daily_pnl.std()
    avg_daily_pnl = daily_pnl.mean()
    
    # Annualized assuming 252 days
    sharpe = (avg_daily_pnl / daily_std) * np.sqrt(252) if daily_std > 0 else 0
    
    downside_std = daily_pnl[daily_pnl < 0].std()
    sortino = (avg_daily_pnl / downside_std) * np.sqrt(252) if downside_std > 0 else 0

    print(f"  Total Trades: {total_trades}")
    print(f"  Total PnL: ${total_pnl:,.2f}")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Avg Trade: ${avg_trade:.2f}")
    print("-" * 20)
    print(f"  Min Daily PnL: ${min_daily_pnl:,.2f} (Limit: -$1,000)")
    if not days_below_limit.empty:
        print(f"  ! VIOLATIONS: {len(days_below_limit)} days below -$1,000")
        print(days_below_limit)
    else:
        print("  ✓ Daily Loss Limit Passed")
        
    print(f"  Max Drawdown: ${max_drawdown:,.2f} (Limit: -$2,000)")
    if max_drawdown < -2000:
        print(f"  ! VIOLATION: Max Drawdown exceeds -$2,000")
    else:
        print("  ✓ Max Drawdown Passed")
        
    print("-" * 20)
    print(f"  Sharpe Ratio (Daily): {sharpe:.2f}")
    print(f"  Sortino Ratio (Daily): {sortino:.2f}")

    # Check for lucky streaks (e.g. max single trade vs total pnl)
    max_win = df[pnl_col].max()
    print(f"  Max Single Win: ${max_win:,.2f}")
    if max_win > total_pnl * 0.5 and total_pnl > 0:
        print("  ! WARNING: One trade accounts for >50% of profits")

if __name__ == "__main__":
    # Find recent logs
    # Adjust pattern as needed to match the running test
    logs = sorted(glob.glob("logs/trades_20260125_19*.csv"))
    
    if not logs:
        print("No logs found.")
    else:
        for log in logs:
            analyze_compliance(log)
