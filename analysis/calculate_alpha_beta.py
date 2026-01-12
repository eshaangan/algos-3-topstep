"""
Calculate Alpha and Beta for ML Intraday V3 Strategy

Alpha: Excess return above what beta exposure would predict
Beta: Correlation of strategy returns with benchmark (MES buy-and-hold)

For a market-neutral intraday strategy, we expect:
- Beta ≈ 0 (uncorrelated with market direction)
- Alpha = total strategy return (since beta ≈ 0)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_trades(run_dir: Path):
    """Load walk-forward trades from run directory."""
    wf_dir = run_dir / "walkforward" / "bar_size=5m"

    if not wf_dir.exists():
        raise FileNotFoundError(f"Walkforward directory not found: {wf_dir}")

    all_trades = []
    for window_dir in sorted(wf_dir.glob("window_*")):
        trades_path = window_dir / "trades.parquet"
        if trades_path.exists():
            df = pd.read_parquet(trades_path)
            if 'executed' in df.columns:
                df = df[df['executed'] == True].copy()
            all_trades.append(df)

    if not all_trades:
        raise ValueError("No trades found!")

    trades_df = pd.concat(all_trades, ignore_index=True)

    # Standardize columns
    if 'entry_ts' not in trades_df.columns and 't0' in trades_df.columns:
        trades_df['entry_ts'] = pd.to_datetime(trades_df['t0'])
    else:
        trades_df['entry_ts'] = pd.to_datetime(trades_df['entry_ts'])

    if 'pnl_usd' not in trades_df.columns and 'pnl_net' in trades_df.columns:
        trades_df['pnl_usd'] = trades_df['pnl_net']

    trades_df['date'] = trades_df['entry_ts'].dt.date
    trades_df = trades_df.sort_values('entry_ts').reset_index(drop=True)

    return trades_df


def calculate_benchmark_returns(trades_df, data_dir: Path = None):
    """
    Calculate MES buy-and-hold benchmark returns for the same period.
    Uses actual MES OHLCV data to compute daily close-to-close returns.
    """
    # Get date range
    start_date = trades_df['date'].min()
    end_date = trades_df['date'].max()

    # Load actual MES data
    mes_csv_path = Path("data/raw/MES_2010_2025_OHLCV_1m.csv")

    if mes_csv_path.exists():
        print(f"Loading MES data from: {mes_csv_path}")
        mes_data = pd.read_csv(mes_csv_path)

        # Parse timestamp and create date column
        if 'timestamp' in mes_data.columns:
            mes_data['timestamp'] = pd.to_datetime(mes_data['timestamp'])
            mes_data['date'] = mes_data['timestamp'].dt.date
        elif 'ts_event' in mes_data.columns:
            mes_data['ts_event'] = pd.to_datetime(mes_data['ts_event'])
            mes_data['date'] = mes_data['ts_event'].dt.date
        else:
            raise ValueError("No timestamp column found in MES data")

        # Filter to strategy date range
        mes_data = mes_data[
            (mes_data['date'] >= start_date) &
            (mes_data['date'] <= end_date)
        ]

        print(f"  MES data range: {mes_data['date'].min()} to {mes_data['date'].max()}")
        print(f"  Total bars: {len(mes_data):,}")

        # Calculate daily close-to-close returns
        daily_close = mes_data.groupby('date')['close'].last().sort_index()
        benchmark_returns = daily_close.pct_change().dropna() * 100  # Percentage

        print(f"  Daily returns calculated: {len(benchmark_returns)} days")
        print(f"  MES mean daily return: {benchmark_returns.mean():.3f}%")
        print(f"  MES daily volatility: {benchmark_returns.std():.3f}%")

        return benchmark_returns

    else:
        raise FileNotFoundError(f"MES data not found at: {mes_csv_path}")


def calculate_alpha_beta(strategy_returns, benchmark_returns, risk_free_rate=0.0):
    """
    Calculate alpha and beta using linear regression.

    Beta = Cov(strategy, benchmark) / Var(benchmark)
    Alpha = Mean(strategy) - Beta * Mean(benchmark) - Risk_free_rate

    Returns:
        dict with beta, alpha (annualized), correlation, etc.
    """
    # Align dates
    common_dates = strategy_returns.index.intersection(benchmark_returns.index)
    if len(common_dates) == 0:
        raise ValueError("No common dates between strategy and benchmark!")

    strat = strategy_returns.loc[common_dates]
    bench = benchmark_returns.loc[common_dates]

    # Remove NaNs
    mask = ~(strat.isna() | bench.isna())
    strat = strat[mask]
    bench = bench[mask]

    if len(strat) < 10:
        raise ValueError(f"Insufficient data: only {len(strat)} days")

    # Calculate beta
    cov = np.cov(strat, bench)[0, 1]
    var_bench = np.var(bench, ddof=1)

    if var_bench < 1e-10:
        # Benchmark has no variance (e.g., all zeros for intraday)
        beta = 0.0
    else:
        beta = cov / var_bench

    # Calculate alpha (excess return)
    mean_strategy = strat.mean()
    mean_benchmark = bench.mean()
    alpha_daily = mean_strategy - beta * mean_benchmark - risk_free_rate / 252

    # Annualize (assuming 252 trading days)
    alpha_annual = alpha_daily * 252

    # Correlation
    if var_bench < 1e-10:
        correlation = 0.0
    else:
        correlation = np.corrcoef(strat, bench)[0, 1]

    # R-squared
    r_squared = correlation ** 2 if not np.isnan(correlation) else 0.0

    # Information Ratio (alpha / tracking error)
    tracking_error = np.std(strat - beta * bench, ddof=1)
    if tracking_error > 0:
        information_ratio = alpha_daily / tracking_error * np.sqrt(252)
    else:
        information_ratio = np.inf

    return {
        'beta': beta,
        'alpha_daily': alpha_daily,
        'alpha_annual': alpha_annual,
        'correlation': correlation,
        'r_squared': r_squared,
        'tracking_error': tracking_error,
        'information_ratio': information_ratio,
        'n_days': len(strat),
    }


def main():
    print("=" * 80)
    print("ALPHA & BETA CALCULATION - ML INTRADAY V3")
    print("=" * 80)
    print()

    # Load trades
    run_dir = Path("runs/v3_2022_5m")
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return

    print(f"Loading trades from: {run_dir}")
    trades_df = load_trades(run_dir)
    print(f"Total trades: {len(trades_df)}")
    print(f"Date range: {trades_df['date'].min()} to {trades_df['date'].max()}")
    print()

    # Calculate daily strategy returns
    daily_pnl = trades_df.groupby('date')['pnl_usd'].sum()
    starting_capital = 50_000  # Topstep 50k account
    daily_returns = (daily_pnl / starting_capital) * 100  # Percentage

    print(f"Daily Returns Summary:")
    print(f"  Mean: {daily_returns.mean():.3f}%")
    print(f"  Std: {daily_returns.std():.3f}%")
    print(f"  Sharpe (daily): {daily_returns.mean() / daily_returns.std():.3f}")
    print(f"  Sharpe (annual): {daily_returns.mean() / daily_returns.std() * np.sqrt(252):.3f}")
    print()

    # Calculate benchmark returns
    data_dir = Path("data")
    benchmark_returns = calculate_benchmark_returns(trades_df, data_dir)
    print(f"Benchmark returns loaded: {len(benchmark_returns)} days")
    print()

    # Calculate alpha & beta
    results = calculate_alpha_beta(daily_returns, benchmark_returns)

    print("=" * 80)
    print("ALPHA & BETA RESULTS")
    print("=" * 80)
    print()
    print(f"Beta: {results['beta']:.4f}")
    print(f"  Interpretation: Strategy moves {results['beta']:.2f}x for every 1% move in MES")
    if abs(results['beta']) < 0.1:
        print(f"  → Market-neutral (beta ≈ 0) ✓")
    print()

    print(f"Alpha (daily): {results['alpha_daily']:.4f}%")
    print(f"Alpha (annual): {results['alpha_annual']:.2f}%")
    print(f"  Interpretation: Excess return above benchmark (skill component)")
    print()

    print(f"Correlation with benchmark: {results['correlation']:.4f}")
    print(f"R-squared: {results['r_squared']:.4f}")
    print(f"  → {results['r_squared']*100:.1f}% of variance explained by benchmark")
    print()

    print(f"Tracking Error (daily): {results['tracking_error']:.3f}%")
    print(f"Information Ratio: {results['information_ratio']:.3f}")
    print(f"  → Alpha per unit of tracking error (higher = better)")
    print()

    print(f"Sample size: {results['n_days']} trading days")
    print()

    # Visualization
    print("Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Cumulative returns
    ax1 = axes[0, 0]
    cum_strategy = (1 + daily_returns / 100).cumprod()
    cum_benchmark = (1 + benchmark_returns / 100).cumprod()
    ax1.plot(cum_strategy.index, cum_strategy.values, label='Strategy', linewidth=2)
    ax1.plot(cum_benchmark.index, cum_benchmark.values, label='Benchmark (MES)', linewidth=2, alpha=0.7)
    ax1.set_title('Cumulative Returns')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Strategy vs Benchmark scatter
    ax2 = axes[0, 1]
    common_dates = daily_returns.index.intersection(benchmark_returns.index)
    if len(common_dates) > 0 and benchmark_returns.loc[common_dates].std() > 1e-6:
        x = benchmark_returns.loc[common_dates]
        y = daily_returns.loc[common_dates]
        ax2.scatter(x, y, alpha=0.5, s=20)

        # Regression line (only if benchmark has variance)
        try:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax2.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Beta={results["beta"]:.3f}')
        except:
            pass

        ax2.set_xlabel('Benchmark Return (%)')
        ax2.set_ylabel('Strategy Return (%)')
        ax2.set_title(f'Strategy vs Benchmark (β={results["beta"]:.3f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        # No benchmark variance - show text instead
        ax2.text(0.5, 0.5, f'Market-Neutral Strategy\nBeta = {results["beta"]:.4f}\n(No benchmark correlation)',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Strategy vs Benchmark')

    # Plot 3: Rolling alpha (30-day)
    ax3 = axes[1, 0]
    rolling_alpha = daily_returns.rolling(30).mean() * 252  # Annualized
    ax3.plot(rolling_alpha.index, rolling_alpha.values, linewidth=1.5)
    ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax3.set_title('Rolling Alpha (30-day, annualized)')
    ax3.set_ylabel('Alpha (%)')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Return distribution
    ax4 = axes[1, 1]
    ax4.hist(daily_returns.dropna(), bins=30, alpha=0.7, edgecolor='black')
    ax4.axvline(daily_returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={daily_returns.mean():.2f}%')
    ax4.set_xlabel('Daily Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Return Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('analysis/alpha_beta_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: analysis/alpha_beta_analysis.png")
    print()

    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv('analysis/alpha_beta_results.csv', index=False)
    print("✓ Saved: analysis/alpha_beta_results.csv")
    print()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    if abs(results['beta']) < 0.1:
        print("✓ Strategy is MARKET-NEUTRAL (beta ≈ 0)")
        print(f"✓ Alpha = {results['alpha_annual']:.2f}% per year (pure skill)")
    else:
        print(f"Strategy has beta exposure: {results['beta']:.2f}")
        print(f"Alpha (excess return): {results['alpha_annual']:.2f}% per year")

    print()
    print(f"Information Ratio: {results['information_ratio']:.2f}")
    if results['information_ratio'] > 1.0:
        print("  → Excellent risk-adjusted performance ✓")
    elif results['information_ratio'] > 0.5:
        print("  → Good risk-adjusted performance ✓")
    else:
        print("  → Moderate risk-adjusted performance")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
