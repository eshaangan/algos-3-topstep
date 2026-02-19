"""
Analyze Batch 3 meta-labeling results.

Batch 3 tests a two-stage meta-labeling architecture:
- Primary model: High recall (70%+), predicts direction
- Secondary model: High precision, filters false positives
- Final signal: primary direction AND secondary "should trade"

Key questions:
1. Does meta-labeling improve signal quality vs single-stage?
2. What's the trade-off between recall (coverage) and precision (quality)?
3. Does CPCV improve generalization vs standard KFold?
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np

def analyze_metalabeling_results():
    """Analyze Batch 3 meta-labeling results."""

    results_dir = Path("ml_intraday_v3/experiments/results/batch3")
    results = []

    # Load all results
    for result_file in sorted(results_dir.glob("result_*.json")):
        with open(result_file) as f:
            data = json.load(f)
            config = data.get('config', {})
            summary = data.get('summary', {})

            results.append({
                'exp_id': data['exp_id'],
                'architecture': config.get('architecture', 'unknown'),
                'labeling_method': config.get('labeling_method', 'unknown'),
                'sample_weight': config.get('sample_weight', 'unknown'),
                'cv_method': config.get('cv_method', 'unknown'),
                'calibration': config.get('calibration', 'none'),

                # Meta-labeling specific
                'primary_recall_threshold': config.get('primary_recall_threshold', 0),
                'final_threshold': config.get('final_threshold', 0),

                # CV metrics
                'median_train_auc': summary.get('median_train_auc', 0),
                'median_test_auc': summary.get('median_test_auc', 0),
                'std_test_auc': summary.get('std_test_auc', 0),
                'median_pr_auc': summary.get('median_pr_auc', 0),
                'median_brier': summary.get('median_brier', 0),

                # Signal metrics
                'pct_signals_055': summary.get('median_pct_signals_above_055', 0),
                'pct_signals_060': summary.get('median_pct_signals_above_060', 0),
                'est_trades_per_day': summary.get('median_est_trades_per_day', 0),
                'mean_test_prob': summary.get('mean_test_prob', 0),

                # Model quality
                'train_test_gap': summary.get('mean_train_test_gap', 0),
                'n_successful_folds': summary.get('n_successful_folds', 0),
            })

    df = pd.DataFrame(results)

    print("=" * 80)
    print("BATCH 3 META-LABELING ANALYSIS")
    print("=" * 80)
    print()

    print("BASELINE COMPARISON:")
    print("  Batch 1 (single-stage trend_scanning):")
    print("    - Best models: 54% signals >0.55")
    print("    - Test AUC: ~0.51")
    print()
    print("  Batch 4 (forward test top 10):")
    print("    - Signals: 82.5% >0.55")
    print("    - Test AUC: 0.503 (barely better than random)")
    print()
    print("  Batch 3 (meta-labeling, two-stage):")
    print("    - Goal: Improve precision via secondary filtering")
    print()
    print("-" * 80)
    print()

    # Overview
    print(f"Total experiments: {len(df)}")
    print()

    # Architecture breakdown
    print("ARCHITECTURE BREAKDOWN:")
    arch_counts = df['architecture'].value_counts()
    for arch, count in arch_counts.items():
        print(f"  {arch}: {count} experiments")
    print()

    # CV method comparison
    print("CV METHOD COMPARISON:")
    cv_comparison = df.groupby('cv_method').agg({
        'median_test_auc': ['mean', 'std'],
        'pct_signals_055': ['mean', 'std'],
        'train_test_gap': ['mean', 'std']
    }).round(3)
    print(cv_comparison)
    print()

    # Calibration comparison
    print("CALIBRATION COMPARISON:")
    cal_comparison = df.groupby('calibration').agg({
        'median_test_auc': ['mean', 'std'],
        'median_brier': ['mean', 'std'],
        'pct_signals_055': 'mean'
    }).round(3)
    print(cal_comparison)
    print()

    # Meta-labeling analysis
    if 'meta_labeling' in df['architecture'].values:
        meta_df = df[df['architecture'] == 'meta_labeling']
        print("META-LABELING SPECIFIC ANALYSIS:")
        print(f"  Total meta-labeling experiments: {len(meta_df)}")
        print(f"  Mean test AUC: {meta_df['median_test_auc'].mean():.3f}")
        print(f"  Mean signals >0.55: {100*meta_df['pct_signals_055'].mean():.1f}%")
        print(f"  Mean signals >0.60: {100*meta_df['pct_signals_060'].mean():.1f}%")
        print(f"  Mean trades/day: {meta_df['est_trades_per_day'].mean():.1f}")
        print()

        # Threshold analysis
        print("  PRIMARY RECALL THRESHOLD IMPACT:")
        threshold_analysis = meta_df.groupby('primary_recall_threshold').agg({
            'median_test_auc': 'mean',
            'pct_signals_055': 'mean',
            'est_trades_per_day': 'mean'
        }).round(3)
        print(threshold_analysis)
        print()

        # Final threshold analysis
        print("  FINAL THRESHOLD IMPACT:")
        final_threshold_analysis = meta_df.groupby('final_threshold').agg({
            'median_test_auc': 'mean',
            'pct_signals_055': 'mean',
            'est_trades_per_day': 'mean'
        }).round(3)
        print(final_threshold_analysis)
        print()

    # Signal quality distribution
    print("SIGNAL QUALITY DISTRIBUTION:")
    print(f"  Mean % signals >0.55: {100*df['pct_signals_055'].mean():.1f}%")
    print(f"  Median % signals >0.55: {100*df['pct_signals_055'].median():.1f}%")
    print(f"  Std % signals >0.55: {100*df['pct_signals_055'].std():.1f}%")
    print(f"  Range: {100*df['pct_signals_055'].min():.1f}% - {100*df['pct_signals_055'].max():.1f}%")
    print()

    # AUC distribution
    print("AUC DISTRIBUTION:")
    print(f"  Mean test AUC: {df['median_test_auc'].mean():.3f}")
    print(f"  Median test AUC: {df['median_test_auc'].median():.3f}")
    print(f"  Std test AUC: {df['median_test_auc'].std():.3f}")
    print(f"  Range: {df['median_test_auc'].min():.3f} - {df['median_test_auc'].max():.3f}")

    good_auc = df[df['median_test_auc'] > 0.5]
    excellent_auc = df[df['median_test_auc'] > 0.55]
    print(f"  Models with AUC > 0.50: {len(good_auc)}/{len(df)} ({100*len(good_auc)/len(df):.1f}%)")
    print(f"  Models with AUC > 0.55: {len(excellent_auc)}/{len(df)} ({100*len(excellent_auc)/len(df):.1f}%)")
    print()

    # Signal diversity check
    print("SIGNAL DIVERSITY CHECK:")
    print(f"  Unique signal rates >0.55: {df['pct_signals_055'].nunique()}")
    print(f"  Unique trades/day: {df['est_trades_per_day'].nunique()}")
    if df['pct_signals_055'].nunique() > 50:
        print("  ✅ HIGHLY DIVERSE signals (vs Batch 4's 3 unique values)")
    elif df['pct_signals_055'].nunique() > 10:
        print("  ✅ MODERATE diversity")
    else:
        print("  ⚠️  LOW diversity")
    print()

    # Ranking metric: combine AUC, signal quality, and diversity
    df['score'] = df['median_test_auc'] * 0.6 + df['pct_signals_055'] * 0.4

    # Top 10 models
    print("TOP 10 MODELS (by 60% AUC + 40% signal quality):")
    print("-" * 80)
    top10 = df.nlargest(10, 'score')
    for idx, row in top10.iterrows():
        print(f"\n{row['exp_id']}:")
        print(f"  Score: {row['score']:.3f}")
        print(f"  Test AUC: {row['median_test_auc']:.3f}")
        print(f"  Signals >0.55: {100*row['pct_signals_055']:.1f}%")
        print(f"  Signals >0.60: {100*row['pct_signals_060']:.1f}%")
        print(f"  Est trades/day: {row['est_trades_per_day']:.1f}")
        print(f"  Train-test gap: {row['train_test_gap']:.3f}")
        print(f"  Architecture: {row['architecture']}")
        print(f"  CV method: {row['cv_method']}")
        print(f"  Calibration: {row['calibration']}")
        if row['architecture'] == 'meta_labeling':
            print(f"  Primary recall threshold: {row['primary_recall_threshold']:.2f}")
            print(f"  Final threshold: {row['final_threshold']:.2f}")
    print()

    # Bottom 10
    print("BOTTOM 10 MODELS:")
    print("-" * 80)
    bottom10 = df.nsmallest(10, 'score')
    for idx, row in bottom10.iterrows():
        print(f"\n{row['exp_id']}:")
        print(f"  Score: {row['score']:.3f}")
        print(f"  Test AUC: {row['median_test_auc']:.3f}")
        print(f"  Signals >0.55: {100*row['pct_signals_055']:.1f}%")
    print()

    # Summary statistics table (top 20)
    print("TOP 20 RESULTS TABLE:")
    print("-" * 80)
    display_cols = ['exp_id', 'score', 'median_test_auc', 'pct_signals_055',
                   'est_trades_per_day', 'cv_method', 'calibration']
    print(df.nlargest(20, 'score')[display_cols].to_string(index=False))
    print()

    # Final verdict
    print("=" * 80)
    print("FINAL VERDICT:")
    print("=" * 80)

    # Compare to baselines
    batch1_best_auc = 0.512
    batch1_best_signals = 0.54
    batch4_mean_auc = 0.503
    batch4_mean_signals = 0.825

    better_auc = df['median_test_auc'].max() > batch4_mean_auc
    better_signals = df['pct_signals_055'].max() > batch4_mean_signals
    diverse_signals = df['pct_signals_055'].nunique() > 10

    print(f"Best Batch 3 AUC: {df['median_test_auc'].max():.3f}")
    print(f"  vs Batch 4 mean: {batch4_mean_auc:.3f}")
    if better_auc:
        print(f"  ✅ BETTER (+{df['median_test_auc'].max() - batch4_mean_auc:.3f})")
    else:
        print(f"  ❌ WORSE ({df['median_test_auc'].max() - batch4_mean_auc:.3f})")
    print()

    print(f"Best Batch 3 signal rate: {100*df['pct_signals_055'].max():.1f}%")
    print(f"  vs Batch 4 mean: {100*batch4_mean_signals:.1f}%")
    if better_signals:
        print(f"  ✅ BETTER")
    else:
        print(f"  ⚠️  LOWER (but may be acceptable for higher precision)")
    print()

    print(f"Signal diversity: {df['pct_signals_055'].nunique()} unique values")
    if diverse_signals:
        print("  ✅ HIGHLY DIVERSE (vs Batch 4's 3)")
    print()

    # Key insights
    print("KEY INSIGHTS:")
    if 'cpcv' in df['cv_method'].values:
        cpcv_df = df[df['cv_method'] == 'cpcv']
        kfold_df = df[df['cv_method'] == 'kfold']
        if len(cpcv_df) > 0 and len(kfold_df) > 0:
            cpcv_auc = cpcv_df['median_test_auc'].mean()
            kfold_auc = kfold_df['median_test_auc'].mean()
            if cpcv_auc > kfold_auc:
                print(f"  ✅ CPCV improves AUC: {cpcv_auc:.3f} vs {kfold_auc:.3f} (KFold)")
            else:
                print(f"  ⚠️  CPCV doesn't improve AUC: {cpcv_auc:.3f} vs {kfold_auc:.3f} (KFold)")

    if 'isotonic' in df['calibration'].values:
        iso_df = df[df['calibration'] == 'isotonic']
        none_df = df[df['calibration'] == 'none']
        if len(iso_df) > 0 and len(none_df) > 0:
            iso_brier = iso_df['median_brier'].mean()
            none_brier = none_df['median_brier'].mean()
            if iso_brier < none_brier:
                print(f"  ✅ Isotonic calibration improves Brier: {iso_brier:.3f} vs {none_brier:.3f} (none)")
            else:
                print(f"  ⚠️  Isotonic doesn't improve Brier: {iso_brier:.3f} vs {none_brier:.3f} (none)")

    if 'meta_labeling' in df['architecture'].values:
        meta_df = df[df['architecture'] == 'meta_labeling']
        single_df = df[df['architecture'] == 'single_model']
        if len(meta_df) > 0 and len(single_df) > 0:
            meta_signals = meta_df['pct_signals_055'].mean()
            single_signals = single_df['pct_signals_055'].mean()
            meta_auc = meta_df['median_test_auc'].mean()
            single_auc = single_df['median_test_auc'].mean()
            print(f"  Meta-labeling vs single-model:")
            print(f"    - AUC: {meta_auc:.3f} vs {single_auc:.3f}")
            print(f"    - Signals >0.55: {100*meta_signals:.1f}% vs {100*single_signals:.1f}%")
            if meta_auc > single_auc or meta_signals < single_signals:
                print(f"    ✅ Meta-labeling may improve precision (fewer but better signals)")
    print()

    # Save summary
    summary = {
        'total_experiments': len(df),
        'mean_test_auc': float(df['median_test_auc'].mean()),
        'median_test_auc': float(df['median_test_auc'].median()),
        'best_test_auc': float(df['median_test_auc'].max()),
        'mean_pct_signals_055': float(df['pct_signals_055'].mean()),
        'best_pct_signals_055': float(df['pct_signals_055'].max()),
        'mean_est_trades_per_day': float(df['est_trades_per_day'].mean()),
        'signal_diversity_score': int(df['pct_signals_055'].nunique()),
        'models_auc_above_050': int(len(good_auc)),
        'models_auc_above_055': int(len(excellent_auc)),
        'baseline_comparison': {
            'batch4_mean_auc': batch4_mean_auc,
            'batch4_mean_signals': batch4_mean_signals,
            'better_than_batch4_auc': bool(better_auc),
            'better_than_batch4_signals': bool(better_signals),
        },
        'top_model': {
            'exp_id': top10.iloc[0]['exp_id'],
            'score': float(top10.iloc[0]['score']),
            'test_auc': float(top10.iloc[0]['median_test_auc']),
            'pct_signals_055': float(top10.iloc[0]['pct_signals_055']),
            'est_trades_per_day': float(top10.iloc[0]['est_trades_per_day']),
            'architecture': top10.iloc[0]['architecture'],
            'cv_method': top10.iloc[0]['cv_method'],
        }
    }

    with open('ml_intraday_v3/experiments/results/batch3_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to: ml_intraday_v3/experiments/results/batch3_summary.json")
    print()

    return df

if __name__ == '__main__':
    df = analyze_metalabeling_results()
