#!/usr/bin/env python3
"""Simple results analyzer for local experiments."""

import json
from pathlib import Path
import pandas as pd

results_dir = Path(__file__).parent.parent / 'results'
results = []

print("\n" + "="*100)
print("GRID SEARCH RESULTS")
print("="*100 + "\n")

# Load all result files
for json_file in results_dir.glob('*.json'):
    try:
        with open(json_file, 'r') as f:
            result = json.load(f)
        
        if result['status'] != 'SUCCESS':
            continue
        
        summary = result['summary']
        config = result['config']
        
        results.append({
            'exp_id': result['exp_id'],
            'model': config['model_name'],
            'features': config['feature_set_name'],
            'window': config['training_window_months'],
            'calib': config['calibration'] or 'none',
            'test_auc': summary['median_test_auc'],
            'gap': summary['mean_train_test_gap'],
            'sig_055': summary['median_pct_signals_above_055'],
            'trades_day': summary['median_est_trades_per_day']
        })
    except Exception as e:
        print(f"⚠️  Error loading {json_file.name}: {e}")

if not results:
    print("No results found in results/")
    exit(1)

# Create DataFrame and sort
df = pd.DataFrame(results)
df = df.sort_values('test_auc', ascending=False)

# Print top configurations
print(f"{'Rank':<6} {'Model':<15} {'Features':<10} {'Win':<5} {'Calib':<10} {'AUC':<8} {'Gap':<8} {'Sig>0.55':<10}")
print("-" * 100)

for i, row in df.head(10).iterrows():
    rank = df.index.get_loc(i) + 1
    print(f"{rank:<6} {row['model']:<15} {row['features']:<10} {row['window']:<5}mo {row['calib']:<10} "
          f"{row['test_auc']:<8.3f} {row['gap']:<8.3f} {row['sig_055']:<10.1%}")

print("\n" + "="*100)
print(f"BEST CONFIGURATION: {df.iloc[0]['exp_id']}")
print("="*100)
print(f"  Model: {df.iloc[0]['model']}")
print(f"  Features: {df.iloc[0]['features']}")
print(f"  Window: {df.iloc[0]['window']} months")
print(f"  Calibration: {df.iloc[0]['calib']}")
print(f"  Test AUC: {df.iloc[0]['test_auc']:.3f}")
print(f"  Train-Test Gap: {df.iloc[0]['gap']:.3f}")
print(f"  Signals > 0.55: {df.iloc[0]['sig_055']:.1%}")
print(f"  Est Trades/Day: {df.iloc[0]['trades_day']:.1f}")
print("="*100 + "\n")

# Save summary
summary_file = results_dir / 'summary.csv'
df.to_csv(summary_file, index=False)
print(f"📊 Full results saved to: {summary_file}\n")
