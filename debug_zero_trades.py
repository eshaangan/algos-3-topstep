"""
Debug why backtest is producing 0 trades.
Run this to see what's blocking trades.
"""

import pandas as pd
import numpy as np
from models.train import load_bars, train_models
from backtesting.backtest import run_backtest
from core.simple_config import TRAINING_CONFIG

print("="*60)
print("DEBUGGING ZERO TRADES IN BACKTEST")
print("="*60)

# Load data
bars = load_bars("data/processed/mes_bars.h5")
experiment_bars = bars.tail(50000).reset_index(drop=True)
print(f"\nData: {len(experiment_bars)} bars")

# Train models (this is what the notebook does)
print("\nTraining models...")
print(f"Config: stop={TRAINING_CONFIG.stop_loss_ticks}, target={TRAINING_CONFIG.target_multiplier}x, hold={TRAINING_CONFIG.max_hold_bars}")

results = train_models(experiment_bars)

print("\n" + "="*60)
print("POLICY TUNING RESULTS")
print("="*60)
policy = results.get('policy', {})
for key, value in policy.items():
    print(f"{key:30s}: {value}")

print("\n" + "="*60)
print("BACKTEST METRICS BY SPLIT")
print("="*60)
backtest_metrics = results.get('backtest_metrics', {})
for split_name in ['training', 'validation', 'test']:
    split_result = backtest_metrics.get(split_name, {})
    summary = split_result.get('summary', {})
    trades = summary.get('trades', 0)
    win_rate = summary.get('win_rate', 0)
    pnl = summary.get('net_pnl', 0)
    print(f"{split_name:12s}: {trades:4d} trades, WR={win_rate:.2%}, P&L=${pnl:>8.2f}")

# Now run the backtest with the tuned policy
print("\n" + "="*60)
print("RUNNING FULL BACKTEST WITH TUNED POLICY")
print("="*60)

backtest = run_backtest(
    experiment_bars,
    results['long_model'],
    results['short_model'],
    results['feature_cols'],
    min_probability_long=policy.get('min_probability_long'),
    min_probability_short=policy.get('min_probability_short'),
    enable_long=policy.get('enable_long'),
    enable_short=policy.get('enable_short'),
    blocked_hours=policy.get('blocked_hours'),
    allowed_hours=policy.get('allowed_hours'),
    exclude_lunch=policy.get('exclude_lunch'),
    require_trend_long=policy.get('require_trend_long'),
    require_trend_short=policy.get('require_trend_short'),
    min_atr_ticks=policy.get('min_atr_ticks'),
    max_atr_ticks=policy.get('max_atr_ticks'),
    stop_loss_ticks=TRAINING_CONFIG.stop_loss_ticks,
    target_multiplier=TRAINING_CONFIG.target_multiplier,
    max_hold_bars=TRAINING_CONFIG.max_hold_bars,
    slippage_ticks=1,
    commission_per_contract=2.35,
)

print(f"Trades: {backtest['summary']['trades']}")
print(f"Win Rate: {backtest['summary']['win_rate']:.2%}")
print(f"Net P&L: ${backtest['summary']['net_pnl']:.2f}")

# If 0 trades, let's diagnose why
if backtest['summary']['trades'] == 0:
    print("\n" + "="*60)
    print("⚠️  ZERO TRADES - DIAGNOSING ISSUE")
    print("="*60)

    # Check if both directions are disabled
    if not policy.get('enable_long') and not policy.get('enable_short'):
        print("\n❌ ISSUE: Both enable_long and enable_short are FALSE")
        print("   The policy tuning couldn't find a profitable strategy.")
        print("   This usually means:")
        print("   - The model doesn't have predictive power on this data")
        print("   - The validation window had no profitable configuration")
        print("   - The tuning criteria are too strict")
        print("\n   SOLUTIONS:")
        print("   1. Lower the probability threshold (try 0.5 instead of 0.65)")
        print("   2. Remove some gating filters")
        print("   3. Use more training data")
        print("   4. Check if the model actually learned something useful")

    # Check probability thresholds
    min_prob_long = policy.get('min_probability_long', 0.65)
    min_prob_short = policy.get('min_probability_short', 0.65)

    if min_prob_long > 0.9 or min_prob_short > 0.9:
        print(f"\n❌ ISSUE: Probability thresholds are very high")
        print(f"   Long: {min_prob_long:.2f}, Short: {min_prob_short:.2f}")
        print("   Very few predictions will exceed these thresholds")

    # Check if hours are blocked
    blocked = policy.get('blocked_hours', [])
    allowed = policy.get('allowed_hours', [])
    if blocked and len(blocked) > 4:
        print(f"\n⚠️  Many hours blocked: {blocked}")
    if allowed and len(allowed) < 2:
        print(f"\n⚠️  Very few hours allowed: {allowed}")

    # Check ATR filters
    min_atr = policy.get('min_atr_ticks')
    max_atr = policy.get('max_atr_ticks')
    if min_atr and max_atr and (max_atr - min_atr) < 5:
        print(f"\n⚠️  Very tight ATR filter: {min_atr}-{max_atr} ticks")

    # Let's check what the model probabilities actually look like
    print("\n" + "="*60)
    print("CHECKING MODEL PREDICTIONS")
    print("="*60)

    from backtesting.backtest import compute_probabilities
    prob_df = compute_probabilities(
        experiment_bars,
        results['long_model'],
        results['short_model'],
        results['feature_cols']
    )

    long_probs = prob_df['long_prob'].dropna()
    short_probs = prob_df['short_prob'].dropna()

    print(f"\nLong probabilities:")
    print(f"  Min:  {long_probs.min():.3f}")
    print(f"  25%:  {long_probs.quantile(0.25):.3f}")
    print(f"  50%:  {long_probs.quantile(0.50):.3f}")
    print(f"  75%:  {long_probs.quantile(0.75):.3f}")
    print(f"  Max:  {long_probs.max():.3f}")
    print(f"  > {min_prob_long:.2f}: {(long_probs > min_prob_long).sum()} bars ({(long_probs > min_prob_long).mean()*100:.1f}%)")

    print(f"\nShort probabilities:")
    print(f"  Min:  {short_probs.min():.3f}")
    print(f"  25%:  {short_probs.quantile(0.25):.3f}")
    print(f"  50%:  {short_probs.quantile(0.50):.3f}")
    print(f"  75%:  {short_probs.quantile(0.75):.3f}")
    print(f"  Max:  {short_probs.max():.3f}")
    print(f"  > {min_prob_short:.2f}: {(short_probs > min_prob_short).sum()} bars ({(short_probs > min_prob_short).mean()*100:.1f}%)")

    # Try a relaxed backtest
    print("\n" + "="*60)
    print("TRYING RELAXED POLICY (0.5 threshold, no gates)")
    print("="*60)

    relaxed = run_backtest(
        experiment_bars,
        results['long_model'],
        results['short_model'],
        results['feature_cols'],
        min_probability_long=0.5,
        min_probability_short=0.5,
        enable_long=True,
        enable_short=False,
        blocked_hours=None,
        allowed_hours=None,
        exclude_lunch=False,
        require_trend_long=False,
        require_trend_short=False,
        min_atr_ticks=None,
        max_atr_ticks=None,
        stop_loss_ticks=TRAINING_CONFIG.stop_loss_ticks,
        target_multiplier=TRAINING_CONFIG.target_multiplier,
        max_hold_bars=TRAINING_CONFIG.max_hold_bars,
        slippage_ticks=1,
        commission_per_contract=2.35,
    )

    print(f"Relaxed backtest trades: {relaxed['summary']['trades']}")
    print(f"Win Rate: {relaxed['summary']['win_rate']:.2%}")
    print(f"Net P&L: ${relaxed['summary']['net_pnl']:.2f}")

    if relaxed['summary']['trades'] > 0:
        print("\n✅ Relaxed policy generates trades!")
        print("   The issue is with the tuned policy being too restrictive.")
        print("\n   RECOMMENDATION:")
        print("   - Use the relaxed policy (0.5 threshold)")
        print("   - Or manually adjust the tuned policy")
        print("   - The tuning algorithm is trying to avoid losses but ended up too conservative")
    else:
        print("\n❌ Even relaxed policy produces 0 trades!")
        print("   This suggests a deeper issue:")
        print("   - Model might not be making predictions")
        print("   - Features might have NaN values")
        print("   - Session times might not match data")

else:
    print(f"\n✅ Backtest generated {backtest['summary']['trades']} trades")
    print("   No issue detected")
