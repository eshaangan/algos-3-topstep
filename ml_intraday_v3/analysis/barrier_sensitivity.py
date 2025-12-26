#!/usr/bin/env python3
"""
Barrier Sensitivity Analysis - Find optimal PT/SL for combine goals

Simulates different barrier combinations to find sweet spot between:
- Trade frequency (need 20-50 trades/month for combine)
- Profitability (need positive expectancy)
- Win rate (need 55-70% to be sustainable)

Goal: Pass $50k Topstep combine in 30 days ($3k profit target)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import math
from typing import Dict, List, Tuple
import argparse


def load_events_and_bars(run_dir: Path, bar_size: str = "1m") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load events and bars data"""
    events_path = run_dir / f"bar_size={bar_size}" / "events.parquet"
    bars_path = run_dir / f"bar_size={bar_size}" / "bars.parquet"

    events = pd.read_parquet(events_path)
    bars = pd.read_parquet(bars_path)

    return events, bars


def simulate_barrier(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
    horizon_bars: int,
    cost_per_trade: float = 6.25,  # $1.25 commission + ~$5 slippage
) -> Dict:
    """
    Simulate what would happen with different barrier settings

    For each event:
    1. Calculate new PT/SL based on sigma and multipliers
    2. Walk forward from entry to find first touch or vertical
    3. Compute return and label
    """
    results = []

    # Ensure bars index is datetime
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars = bars.set_index('timestamp')

    bars_indexed = bars.sort_index()

    for idx, event in events.iterrows():
        if idx % 10000 == 0:
            print(f"Processing event {idx}/{len(events)}...")

        t0 = event['t0']
        entry_price = event['entry_price']
        sigma = event['sigma']
        side = event.get('side', 1)  # Default to long if not specified

        # Calculate barriers
        if side == 1:  # Long
            pt_price = entry_price + (pt_mult * sigma)
            sl_price = entry_price - (sl_mult * sigma)
        else:  # Short
            pt_price = entry_price - (pt_mult * sigma)
            sl_price = entry_price + (sl_mult * sigma)

        # Find bars from entry to horizon
        t1 = t0 + pd.Timedelta(minutes=horizon_bars)
        future_bars = bars_indexed.loc[t0:t1]

        if len(future_bars) < 2:
            continue

        # Skip entry bar, start from next bar
        future_bars = future_bars.iloc[1:]

        # Check each bar for PT/SL touch (OHLC path ordering)
        hit_pt = False
        hit_sl = False
        exit_price = None
        exit_reason = None
        bars_held = 0

        for bar_idx, (bar_time, bar) in enumerate(future_bars.iterrows()):
            bars_held = bar_idx + 1

            # OHLC path ordering (open -> high -> low -> close)
            prices_to_check = [
                ('open', bar['open']),
                ('high', bar['high']),
                ('low', bar['low']),
                ('close', bar['close']),
            ]

            for price_label, price in prices_to_check:
                if side == 1:  # Long
                    if price >= pt_price and not hit_pt:
                        hit_pt = True
                        exit_price = pt_price
                        exit_reason = 'target_first'
                        break
                    elif price <= sl_price and not hit_sl:
                        hit_sl = True
                        exit_price = sl_price
                        exit_reason = 'stop_first'
                        break
                else:  # Short
                    if price <= pt_price and not hit_pt:
                        hit_pt = True
                        exit_price = pt_price
                        exit_reason = 'target_first'
                        break
                    elif price >= sl_price and not hit_sl:
                        hit_sl = True
                        exit_price = sl_price
                        exit_reason = 'stop_first'
                        break

            if hit_pt or hit_sl:
                break

        # Vertical barrier (time exit)
        if not hit_pt and not hit_sl:
            exit_price = future_bars.iloc[-1]['close'] if len(future_bars) > 0 else entry_price
            exit_reason = 'vertical'
            bars_held = len(future_bars)

        # Calculate return
        if side == 1:
            ret_gross = exit_price - entry_price
        else:
            ret_gross = entry_price - exit_price

        ret_net = ret_gross - cost_per_trade

        # Label
        if exit_reason == 'target_first':
            label = 1
        elif exit_reason == 'stop_first':
            label = -1
        else:
            label = 0

        results.append({
            'event_id': event.get('event_id', idx),
            't0': t0,
            'sigma': sigma,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'bars_held': bars_held,
            'ret_gross': ret_gross,
            'ret_net': ret_net,
            'label': label,
            'pt_mult': pt_mult,
            'sl_mult': sl_mult,
            'horizon': horizon_bars,
        })

    return pd.DataFrame(results)


def analyze_barrier_config(
    sim_results: pd.DataFrame,
    years_of_data: float,
    combine_days: int = 30,
    combine_target: float = 3000.0,
) -> Dict:
    """Analyze a barrier configuration for combine feasibility"""

    total_events = len(sim_results)

    # Filter to only positive net returns (would actually trade these)
    profitable = sim_results[sim_results['ret_net'] > 0]
    unprofitable = sim_results[sim_results['ret_net'] <= 0]

    # Stats
    win_rate = (sim_results['label'] == 1).sum() / len(sim_results) if len(sim_results) > 0 else 0
    avg_win = sim_results[sim_results['ret_net'] > 0]['ret_net'].mean() if len(profitable) > 0 else 0
    avg_loss = sim_results[sim_results['ret_net'] < 0]['ret_net'].mean() if len(unprofitable) > 0 else 0
    avg_ret_net = sim_results['ret_net'].mean()

    # Trade frequency
    trades_per_year = total_events / years_of_data
    trades_per_month = trades_per_year / 12
    trades_in_combine = trades_per_month * (combine_days / 30)

    # Expected P&L
    total_pnl = sim_results['ret_net'].sum()
    pnl_per_trade = total_pnl / total_events if total_events > 0 else 0
    expected_combine_pnl = trades_in_combine * pnl_per_trade

    # Probability of hitting target
    if trades_in_combine > 0:
        # Simplified: assume normal distribution of returns
        std_ret = sim_results['ret_net'].std()
        combine_std = std_ret * np.sqrt(trades_in_combine)
        z_score = (combine_target - expected_combine_pnl) / combine_std if combine_std > 0 else 0
        prob_hit_target = 1 - 0.5 * (1 + math.erf(z_score / math.sqrt(2)))
    else:
        prob_hit_target = 0.0

    return {
        'total_events': total_events,
        'trades_per_year': trades_per_year,
        'trades_per_month': trades_per_month,
        'trades_in_30d': trades_in_combine,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_ret_per_trade': pnl_per_trade,
        'total_pnl_15y': total_pnl,
        'expected_30d_pnl': expected_combine_pnl,
        'prob_hit_3k_target': prob_hit_target,
        'avg_bars_held': sim_results['bars_held'].mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Barrier sensitivity analysis")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/clean_data_20251225_040511"),
        help="Run directory with events data"
    )
    parser.add_argument(
        "--bar-size",
        type=str,
        default="1m",
        help="Bar size to analyze"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/barrier_sensitivity_results.csv"),
        help="Output CSV path"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("BARRIER SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Run: {args.run_dir}")
    print(f"Bar size: {args.bar_size}")
    print()

    # Load data
    print("Loading events and bars...")
    events, bars = load_events_and_bars(args.run_dir, args.bar_size)
    print(f"  Events: {len(events):,}")
    print(f"  Bars: {len(bars):,}")
    print()

    # Calculate years of data
    bars_with_time = bars.copy()
    if 'timestamp' in bars_with_time.columns:
        bars_with_time = bars_with_time.set_index('timestamp')

    date_range = bars_with_time.index.max() - bars_with_time.index.min()
    years_of_data = date_range.days / 365.25
    print(f"Data span: {years_of_data:.1f} years")
    print()

    # Barrier combinations to test
    configurations = [
        # Current (baseline)
        {"pt": 2.9, "sl": 3.4, "horizon": 12, "name": "Current (2.9/3.4/12)"},
        {"pt": 2.9, "sl": 3.4, "horizon": 24, "name": "Current (2.9/3.4/24)"},

        # Aggressive (high frequency)
        {"pt": 0.5, "sl": 0.6, "horizon": 5, "name": "Ultra-Aggressive (0.5/0.6/5)"},
        {"pt": 0.8, "sl": 1.0, "horizon": 8, "name": "Very Aggressive (0.8/1.0/8)"},
        {"pt": 1.0, "sl": 1.2, "horizon": 10, "name": "Aggressive (1.0/1.2/10)"},

        # Moderate (balance)
        {"pt": 1.5, "sl": 1.8, "horizon": 12, "name": "Moderate (1.5/1.8/12)"},
        {"pt": 2.0, "sl": 2.5, "horizon": 15, "name": "Conservative (2.0/2.5/15)"},
    ]

    all_results = []

    for config in configurations:
        print(f"\n{'=' * 80}")
        print(f"Testing: {config['name']}")
        print(f"  PT={config['pt']}σ, SL={config['sl']}σ, Horizon={config['horizon']} bars")
        print(f"{'=' * 80}")

        # Simulate
        sim_results = simulate_barrier(
            events=events,
            bars=bars,
            pt_mult=config['pt'],
            sl_mult=config['sl'],
            horizon_bars=config['horizon'],
        )

        # Analyze
        metrics = analyze_barrier_config(sim_results, years_of_data)

        # Print summary
        print(f"\n📊 RESULTS:")
        print(f"  Total events: {metrics['total_events']:,}")
        print(f"  Trades/year: {metrics['trades_per_year']:.1f}")
        print(f"  Trades/month: {metrics['trades_per_month']:.1f}")
        print(f"  Expected trades in 30 days: {metrics['trades_in_30d']:.1f}")
        print(f"\n💰 PROFITABILITY:")
        print(f"  Win rate: {metrics['win_rate']:.1%}")
        print(f"  Avg win: ${metrics['avg_win']:.2f}")
        print(f"  Avg loss: ${metrics['avg_loss']:.2f}")
        print(f"  Avg return/trade: ${metrics['avg_ret_per_trade']:.2f}")
        print(f"  Total P&L (15y): ${metrics['total_pnl_15y']:,.2f}")
        print(f"\n🎯 COMBINE FEASIBILITY:")
        print(f"  Expected 30-day P&L: ${metrics['expected_30d_pnl']:.2f}")
        print(f"  Probability of hitting $3k: {metrics['prob_hit_3k_target']:.1%}")
        print(f"  Avg bars held: {metrics['avg_bars_held']:.1f}")

        # Grade it
        if metrics['trades_in_30d'] >= 20 and metrics['expected_30d_pnl'] >= 500:
            grade = "✅ VIABLE"
        elif metrics['trades_in_30d'] >= 10 and metrics['expected_30d_pnl'] >= 200:
            grade = "⚠️  MARGINAL"
        else:
            grade = "❌ NOT VIABLE"

        print(f"\n  ASSESSMENT: {grade}")

        # Store
        result_row = {
            'config_name': config['name'],
            'pt_mult': config['pt'],
            'sl_mult': config['sl'],
            'horizon': config['horizon'],
            **metrics,
            'assessment': grade,
        }
        all_results.append(result_row)

    # Save results
    results_df = pd.DataFrame(all_results)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(args.output, index=False)
    print(f"\n✅ Results saved to: {args.output}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"\n{'Configuration':<35} {'Trades/30d':<12} {'$/Trade':<12} {'30d P&L':<12} {'Assessment'}")
    print("-" * 90)

    for _, row in results_df.iterrows():
        print(f"{row['config_name']:<35} {row['trades_in_30d']:>11.1f} "
              f"${row['avg_ret_per_trade']:>10.2f} ${row['expected_30d_pnl']:>10.2f}  "
              f"{row['assessment']}")

    # Find best config
    viable = results_df[results_df['expected_30d_pnl'] > 0].copy()
    if len(viable) > 0:
        viable['score'] = viable['expected_30d_pnl'] * viable['prob_hit_3k_target']
        best = viable.loc[viable['score'].idxmax()]

        print("\n" + "=" * 80)
        print("🏆 RECOMMENDED CONFIGURATION")
        print("=" * 80)
        print(f"  {best['config_name']}")
        print(f"  Expected 30-day P&L: ${best['expected_30d_pnl']:.2f}")
        print(f"  Probability of $3k target: {best['prob_hit_3k_target']:.1%}")
        print(f"  Trades in 30 days: {best['trades_in_30d']:.1f}")
        print(f"  Win rate: {best['win_rate']:.1%}")

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
