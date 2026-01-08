"""
HMM Test Results Visualization (2022-2025)

Run this in your notebook to visualize the HMM regime detection results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# Load data
print("Loading HMM test results...")
run_dir = "runs/v3_data_20260106_225113/bar_size=1m"

bars = pd.read_parquet(f"{run_dir}/bars.parquet")
regimes = pd.read_parquet(f"{run_dir}/hmm_regimes.parquet")

print(f"Loaded {len(bars):,} bars and {len(regimes):,} regime assignments")
print(f"Date range: {bars.index[0]} to {bars.index[-1]}")
print("")

# ============================================================================
# 1. PRICE WITH REGIME COLORING
# ============================================================================
print("=" * 80)
print("1. PRICE WITH REGIME COLORING")
print("=" * 80)

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

# Plot 1: Price with regime background
ax = axes[0]
ax.plot(bars.index, bars['close'], 'k-', linewidth=0.5, alpha=0.8, label='Close Price')

# Get regime spans for background coloring
regime_changes = regimes['hmm_state'].diff().fillna(1) != 0
change_indices = np.where(regime_changes)[0]

for i in range(len(change_indices) - 1):
    start_idx = change_indices[i]
    end_idx = change_indices[i + 1]
    regime = regimes['hmm_state'].iloc[start_idx]

    if pd.notna(regime):
        color = 'green' if regime == 1 else 'red'
        start_time = regimes.index[start_idx]
        end_time = regimes.index[end_idx]
        ax.axvspan(start_time, end_time, alpha=0.15, color=color)

ax.set_ylabel('Price', fontsize=12)
ax.set_title('ES/MES Price with HMM Regime Detection (Green=Regime 1, Red=Regime 0)', fontsize=14, fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Plot 2: Regime probabilities (if available)
ax = axes[1]
if 'prob_state_0' in regimes.columns and 'prob_state_1' in regimes.columns:
    ax.plot(regimes.index, regimes['prob_state_0'], 'r-', alpha=0.7, linewidth=0.8, label='P(Regime 0)')
    ax.plot(regimes.index, regimes['prob_state_1'], 'g-', alpha=0.7, linewidth=0.8, label='P(Regime 1)')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title('Regime Probabilities Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1)
else:
    # Plot discrete regime states
    ax.scatter(regimes.index, regimes['hmm_state'], c=regimes['hmm_state'],
               cmap='RdYlGn', s=0.5, alpha=0.5)
    ax.set_ylabel('Regime', fontsize=12)
    ax.set_title('Regime States (0 or 1)', fontsize=14, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Regime 0', 'Regime 1'])

ax.grid(True, alpha=0.3)

# Plot 3: Returns colored by regime
ax = axes[2]
returns = bars['close'].pct_change()
regime_0_mask = regimes['hmm_state'] == 0
regime_1_mask = regimes['hmm_state'] == 1

ax.scatter(regimes[regime_0_mask].index, returns[regimes[regime_0_mask].index],
           c='red', s=1, alpha=0.3, label='Regime 0 Returns')
ax.scatter(regimes[regime_1_mask].index, returns[regimes[regime_1_mask].index],
           c='green', s=1, alpha=0.3, label='Regime 1 Returns')
ax.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
ax.set_ylabel('Returns', fontsize=12)
ax.set_title('Returns by Regime', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Plot 4: Rolling volatility
ax = axes[3]
rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
ax.plot(rolling_vol.index, rolling_vol, 'b-', linewidth=0.8, alpha=0.8)
ax.set_ylabel('Annualized Vol (%)', fontsize=12)
ax.set_title('Rolling 20-bar Volatility', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=12)
ax.grid(True, alpha=0.3)

# Format x-axis
for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))

plt.tight_layout()
plt.savefig(f'{run_dir}/hmm_visualization.png', dpi=150, bbox_inches='tight')
print("Saved visualization to: hmm_visualization.png")
plt.show()

# ============================================================================
# 2. REGIME STATISTICS
# ============================================================================
print("")
print("=" * 80)
print("2. REGIME STATISTICS")
print("=" * 80)

valid_regimes = regimes['hmm_state'].dropna()
regime_counts = valid_regimes.value_counts().sort_index()

print(f"\nTotal bars with regime assignments: {len(valid_regimes):,}")
print(f"\nRegime Distribution:")
for regime, count in regime_counts.items():
    pct = count / len(valid_regimes) * 100
    print(f"  Regime {int(regime)}: {count:,} bars ({pct:.1f}%)")

# Returns by regime
returns_with_regime = pd.DataFrame({
    'return': returns,
    'regime': regimes['hmm_state']
}).dropna()

print(f"\n{'='*40}")
print("RETURNS BY REGIME")
print(f"{'='*40}")

for regime in sorted(returns_with_regime['regime'].unique()):
    regime_returns = returns_with_regime[returns_with_regime['regime'] == regime]['return']
    print(f"\nRegime {int(regime)}:")
    print(f"  Mean return:  {regime_returns.mean()*100:.4f}% per bar")
    print(f"  Std return:   {regime_returns.std()*100:.4f}% per bar")
    print(f"  Median:       {regime_returns.median()*100:.4f}% per bar")
    print(f"  Skewness:     {regime_returns.skew():.4f}")
    print(f"  Kurtosis:     {regime_returns.kurtosis():.4f}")
    print(f"  Sharpe (ann): {regime_returns.mean() / regime_returns.std() * np.sqrt(252*390):.4f}")

# ============================================================================
# 3. REGIME TRANSITIONS
# ============================================================================
print(f"\n{'='*80}")
print("3. REGIME TRANSITIONS")
print(f"{'='*80}")

regime_changes = valid_regimes.diff().fillna(0) != 0
n_transitions = regime_changes.sum()
print(f"\nTotal regime transitions: {n_transitions:,}")

# Transition matrix
from collections import Counter
transitions = []
for i in range(len(valid_regimes) - 1):
    transitions.append((valid_regimes.iloc[i], valid_regimes.iloc[i+1]))

transition_counts = Counter(transitions)
print(f"\nTransition Matrix (counts):")
print(f"  (0 → 0): {transition_counts.get((0, 0), 0):,}")
print(f"  (0 → 1): {transition_counts.get((0, 1), 0):,}")
print(f"  (1 → 0): {transition_counts.get((1, 0), 0):,}")
print(f"  (1 → 1): {transition_counts.get((1, 1), 0):,}")

# Average regime duration
regime_spans = []
current_regime = valid_regimes.iloc[0]
span_start = 0

for i in range(1, len(valid_regimes)):
    if valid_regimes.iloc[i] != current_regime:
        span_length = i - span_start
        regime_spans.append((current_regime, span_length))
        current_regime = valid_regimes.iloc[i]
        span_start = i

# Last span
regime_spans.append((current_regime, len(valid_regimes) - span_start))

regime_0_durations = [s[1] for s in regime_spans if s[0] == 0]
regime_1_durations = [s[1] for s in regime_spans if s[0] == 1]

print(f"\nAverage Regime Duration:")
print(f"  Regime 0: {np.mean(regime_0_durations):.1f} bars (median: {np.median(regime_0_durations):.0f})")
print(f"  Regime 1: {np.mean(regime_1_durations):.1f} bars (median: {np.median(regime_1_durations):.0f})")

# ============================================================================
# 4. REGIME INTERPRETATION
# ============================================================================
print(f"\n{'='*80}")
print("4. REGIME INTERPRETATION")
print(f"{'='*80}")

regime_0_return = returns_with_regime[returns_with_regime['regime'] == 0]['return'].mean()
regime_1_return = returns_with_regime[returns_with_regime['regime'] == 1]['return'].mean()
regime_0_vol = returns_with_regime[returns_with_regime['regime'] == 0]['return'].std()
regime_1_vol = returns_with_regime[returns_with_regime['regime'] == 1]['return'].std()

print(f"\nRegime Characteristics:")
if regime_1_return > regime_0_return:
    print(f"  🟢 Regime 1: BULLISH (higher returns: {regime_1_return*100:.4f}% vs {regime_0_return*100:.4f}%)")
    print(f"  🔴 Regime 0: BEARISH")
else:
    print(f"  🟢 Regime 0: BULLISH (higher returns: {regime_0_return*100:.4f}% vs {regime_1_return*100:.4f}%)")
    print(f"  🔴 Regime 1: BEARISH")

if regime_0_vol > regime_1_vol:
    print(f"\n  Regime 0: HIGH VOLATILITY ({regime_0_vol*100:.4f}%)")
    print(f"  Regime 1: LOW VOLATILITY ({regime_1_vol*100:.4f}%)")
else:
    print(f"\n  Regime 1: HIGH VOLATILITY ({regime_1_vol*100:.4f}%)")
    print(f"  Regime 0: LOW VOLATILITY ({regime_0_vol*100:.4f}%)")

# ============================================================================
# 5. RECOMMENDATION
# ============================================================================
print(f"\n{'='*80}")
print("5. VALIDATION CHECKLIST")
print(f"{'='*80}")

print("\n✓ Review the visualization above and check:")
print("  1. Do the regime colors match market conditions (bull/bear)?")
print("  2. Are regime transitions reasonable (not too frequent)?")
print("  3. Do the regime statistics make sense?")
print("  4. Is there clear separation between regime return distributions?")
print("\nIf YES to all → Proceed with full 2010-2025 run")
print("If NO → May need to adjust HMM parameters or reconsider approach")
print("")
