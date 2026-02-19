"""
Comprehensive analysis of ALL completed experiment batches.

Analyzes Batches 1, 3, and 4 to find the best overall models and insights.
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def load_batch_results(batch_name, results_dir="ml_intraday_v3/experiments/results"):
    """Load all results from a batch."""
    batch_dir = Path(results_dir) / batch_name
    results = []

    for result_file in sorted(batch_dir.glob("result_*.json")):
        with open(result_file) as f:
            data = json.load(f)
            config = data.get('config', {})
            summary = data.get('summary', {})

            result = {
                'batch': batch_name,
                'exp_id': data['exp_id'],
                'status': data.get('status', 'UNKNOWN'),

                # Config
                'architecture': config.get('architecture', config.get('labeling_method', 'unknown')),
                'labeling_method': config.get('labeling_method', 'unknown'),
                'sample_weight': config.get('sample_weight', 'unknown'),
                'cv_method': config.get('cv_method', 'unknown'),
                'calibration': config.get('calibration', 'none'),
                'feature_set': config.get('feature_set_name', 'unknown'),

                # Performance metrics
                'median_test_auc': summary.get('median_test_auc', 0),
                'median_train_auc': summary.get('median_train_auc', 0),
                'std_test_auc': summary.get('std_test_auc', 0),
                'median_pr_auc': summary.get('median_pr_auc', 0),
                'median_brier': summary.get('median_brier', 1.0),

                # Signal metrics
                'pct_signals_055': summary.get('median_pct_signals_above_055', 0),
                'pct_signals_060': summary.get('median_pct_signals_above_060', 0),
                'est_trades_per_day': summary.get('median_est_trades_per_day', 0),

                # Quality metrics
                'train_test_gap': summary.get('mean_train_test_gap', 0),
                'n_successful_folds': summary.get('n_successful_folds', 0),
            }

            results.append(result)

    return pd.DataFrame(results)


def analyze_all_batches():
    """Comprehensive analysis across all batches."""

    print("=" * 80)
    print("COMPREHENSIVE CROSS-BATCH ANALYSIS")
    print("=" * 80)
    print()

    # Load all batches
    print("Loading results...")
    batch1 = load_batch_results("batch1")
    batch3 = load_batch_results("batch3")
    batch4 = load_batch_results("batch4")

    print(f"  Batch 1: {len(batch1)} experiments")
    print(f"  Batch 3: {len(batch3)} experiments")
    print(f"  Batch 4: {len(batch4)} experiments")
    print(f"  Total: {len(batch1) + len(batch3) + len(batch4)} experiments")
    print()

    # Combine all
    all_results = pd.concat([batch1, batch3, batch4], ignore_index=True)

    # Overall statistics
    print("=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print()

    print(f"Mean test AUC: {all_results['median_test_auc'].mean():.3f}")
    print(f"Best test AUC: {all_results['median_test_auc'].max():.3f}")
    print(f"Worst test AUC: {all_results['median_test_auc'].min():.3f}")
    print()

    print(f"Mean signal rate >0.55: {100*all_results['pct_signals_055'].mean():.1f}%")
    print(f"Best signal rate >0.55: {100*all_results['pct_signals_055'].max():.1f}%")
    print()

    # Models with AUC > 0.50
    good_auc = all_results[all_results['median_test_auc'] > 0.50]
    excellent_auc = all_results[all_results['median_test_auc'] > 0.52]
    print(f"Models with AUC > 0.50: {len(good_auc)}/{len(all_results)} ({100*len(good_auc)/len(all_results):.1f}%)")
    print(f"Models with AUC > 0.52: {len(excellent_auc)}/{len(all_results)} ({100*len(excellent_auc)/len(all_results):.1f}%)")
    print()

    # Batch comparison
    print("=" * 80)
    print("BATCH COMPARISON")
    print("=" * 80)
    print()

    batch_stats = all_results.groupby('batch').agg({
        'median_test_auc': ['mean', 'max', 'std'],
        'pct_signals_055': ['mean', 'max', 'std'],
        'train_test_gap': 'mean'
    }).round(3)

    print(batch_stats)
    print()

    # Best by technique
    print("=" * 80)
    print("BEST RESULTS BY TECHNIQUE")
    print("=" * 80)
    print()

    # Labeling method
    print("BY LABELING METHOD:")
    labeling_stats = all_results.groupby('labeling_method').agg({
        'median_test_auc': ['mean', 'max', 'count'],
        'pct_signals_055': 'mean'
    }).round(3)
    labeling_stats.columns = ['_'.join(col).strip() for col in labeling_stats.columns.values]
    print(labeling_stats.sort_values('median_test_auc_max', ascending=False))
    print()

    # CV method
    print("BY CV METHOD:")
    cv_stats = all_results[all_results['cv_method'] != 'unknown'].groupby('cv_method').agg({
        'median_test_auc': ['mean', 'max', 'count'],
        'train_test_gap': 'mean'
    }).round(3)
    cv_stats.columns = ['_'.join(col).strip() for col in cv_stats.columns.values]
    print(cv_stats)
    print()

    # Sample weighting
    print("BY SAMPLE WEIGHT:")
    weight_stats = all_results.groupby('sample_weight').agg({
        'median_test_auc': ['mean', 'max', 'count'],
        'pct_signals_055': 'mean'
    }).round(3)
    weight_stats.columns = ['_'.join(col).strip() for col in weight_stats.columns.values]
    print(weight_stats.sort_values('median_test_auc_max', ascending=False))
    print()

    # Calibration
    print("BY CALIBRATION:")
    cal_stats = all_results.groupby('calibration').agg({
        'median_test_auc': ['mean', 'max', 'count'],
        'median_brier': 'mean'
    }).round(3)
    cal_stats.columns = ['_'.join(col).strip() for col in cal_stats.columns.values]
    print(cal_stats.sort_values('median_test_auc_max', ascending=False))
    print()

    # Top 20 models overall
    print("=" * 80)
    print("TOP 20 MODELS OVERALL")
    print("=" * 80)
    print()

    # Ranking metric: 60% AUC + 40% signal quality
    all_results['score'] = all_results['median_test_auc'] * 0.6 + all_results['pct_signals_055'] * 0.4
    top20 = all_results.nlargest(20, 'score')

    for idx, row in top20.iterrows():
        print(f"\n{row['exp_id']} (Batch: {row['batch']})")
        print(f"  Score: {row['score']:.3f}")
        print(f"  AUC: {row['median_test_auc']:.3f}")
        print(f"  Signals >0.55: {100*row['pct_signals_055']:.1f}%")
        print(f"  Trades/day: {row['est_trades_per_day']:.1f}")
        print(f"  Labeling: {row['labeling_method']}")
        print(f"  CV: {row['cv_method']}")
        print(f"  Sample weight: {row['sample_weight']}")
        print(f"  Calibration: {row['calibration']}")
    print()

    # Key insights
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()

    # 1. Best overall model
    best = top20.iloc[0]
    print(f"1. BEST OVERALL MODEL: {best['exp_id']}")
    print(f"   - AUC: {best['median_test_auc']:.3f}")
    print(f"   - Signals: {100*best['pct_signals_055']:.1f}%")
    print(f"   - Approach: {best['labeling_method']} + {best['cv_method']} + {best['calibration']}")
    print()

    # 2. Best labeling
    best_labeling = labeling_stats.sort_values('median_test_auc_max', ascending=False).index[0]
    print(f"2. BEST LABELING METHOD: {best_labeling}")
    print(f"   - Max AUC: {labeling_stats.loc[best_labeling, 'median_test_auc_max']:.3f}")
    print()

    # 3. CPCV vs KFold
    if 'cpcv' in cv_stats.index and 'kfold' in cv_stats.index:
        cpcv_auc = cv_stats.loc['cpcv', 'median_test_auc_mean']
        kfold_auc = cv_stats.loc['kfold', 'median_test_auc_mean']
        print(f"3. CPCV vs KFold:")
        print(f"   - CPCV: {cpcv_auc:.3f} AUC")
        print(f"   - KFold: {kfold_auc:.3f} AUC")
        if cpcv_auc > kfold_auc:
            print(f"   - ✅ CPCV is better by {cpcv_auc - kfold_auc:.3f}")
        else:
            print(f"   - ⚠️  KFold is better by {kfold_auc - cpcv_auc:.3f}")
    print()

    # 4. Calibration impact
    if 'isotonic' in cal_stats.index and 'none' in cal_stats.index:
        iso_brier = cal_stats.loc['isotonic', 'median_brier_mean']
        none_brier = cal_stats.loc['none', 'median_brier_mean']
        print(f"4. CALIBRATION IMPACT:")
        print(f"   - Isotonic Brier: {iso_brier:.3f}")
        print(f"   - No calibration Brier: {none_brier:.3f}")
        if iso_brier < none_brier:
            print(f"   - ✅ Calibration improves Brier score")
    print()

    # 5. Signal quality tradeoff
    high_auc = all_results[all_results['median_test_auc'] > 0.52]
    if len(high_auc) > 0:
        print(f"5. SIGNAL QUALITY FOR HIGH-AUC MODELS:")
        print(f"   - Models with AUC > 0.52: {len(high_auc)}")
        print(f"   - Mean signal rate: {100*high_auc['pct_signals_055'].mean():.1f}%")
        print(f"   - Mean trades/day: {high_auc['est_trades_per_day'].mean():.1f}")
    print()

    # Save results
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    print()

    # Save top 20
    top20_file = "ml_intraday_v3/experiments/results/top_20_models.csv"
    top20.to_csv(top20_file, index=False)
    print(f"✅ Top 20 models saved to: {top20_file}")

    # Save full results
    all_file = "ml_intraday_v3/experiments/results/all_experiments.csv"
    all_results.to_csv(all_file, index=False)
    print(f"✅ All {len(all_results)} experiments saved to: {all_file}")

    # Save summary
    summary = {
        'total_experiments': len(all_results),
        'best_auc': float(all_results['median_test_auc'].max()),
        'mean_auc': float(all_results['median_test_auc'].mean()),
        'models_above_050_auc': int(len(good_auc)),
        'models_above_052_auc': int(len(excellent_auc)),
        'best_model': {
            'exp_id': best['exp_id'],
            'batch': best['batch'],
            'auc': float(best['median_test_auc']),
            'signals_055': float(best['pct_signals_055']),
            'labeling': best['labeling_method'],
            'cv': best['cv_method'],
            'calibration': best['calibration'],
        },
        'technique_rankings': {
            'best_labeling': best_labeling,
            'best_cv': cv_stats.sort_values('median_test_auc_max', ascending=False).index[0] if len(cv_stats) > 0 else 'unknown',
        }
    }

    summary_file = "ml_intraday_v3/experiments/results/overall_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to: {summary_file}")
    print()

    return all_results, top20


if __name__ == '__main__':
    all_results, top20 = analyze_all_batches()
