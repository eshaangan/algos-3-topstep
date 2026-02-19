"""
Phase 1: Model Selection from Completed Experiments

Implements the selection criteria from final_model_selection_plan.md:
- Filter to models with AUC > 0.52 (top ~25%)
- Rank by Final Score = 0.40*AUC + 0.30*Signal_Quality + 0.20*Calibration + 0.10*Stability
- Select top 10 candidates
- Check for diversity (different approaches)

Minimum Requirements:
- ✅ AUC > 0.52 (meaningful edge)
- ✅ Signal rate >0.15 (at least 15% actionable)
- ✅ Trades/day > 2 (sufficient opportunities)
- ✅ 5/5 successful CV folds (no failures)
- ✅ Train-test gap < 0.40 (not overfitting)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_all_experiments(results_dir: str = "ml_intraday_v3/experiments/results") -> pd.DataFrame:
    """Load all experiment results from all batches."""
    
    results_dir = Path(results_dir)
    all_results = []
    
    # Load from all batch directories
    for batch_dir in sorted(results_dir.glob("batch*")):
        if not batch_dir.is_dir():
            continue
            
        batch_name = batch_dir.name
        print(f"Loading {batch_name}...")
        
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
                
                all_results.append(result)
    
    df = pd.DataFrame(all_results)
    print(f"\nTotal experiments loaded: {len(df)}")
    return df


def apply_minimum_requirements(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to models meeting minimum requirements."""
    
    print("\n" + "=" * 80)
    print("APPLYING MINIMUM REQUIREMENTS")
    print("=" * 80)
    
    initial_count = len(df)
    
    # Only keep SUCCESS status
    df = df[df['status'] == 'SUCCESS'].copy()
    print(f"Status = SUCCESS: {len(df)}/{initial_count} models ({100*len(df)/initial_count:.1f}%)")
    
    # Requirement 1: AUC > 0.52
    req1 = df['median_test_auc'] > 0.52
    print(f"AUC > 0.52: {req1.sum()}/{len(df)} models ({100*req1.sum()/len(df):.1f}%)")
    
    # Requirement 2: Signal rate > 0.15
    req2 = df['pct_signals_055'] > 0.15
    print(f"Signal rate > 15%: {req2.sum()}/{len(df)} models ({100*req2.sum()/len(df):.1f}%)")
    
    # Requirement 3: Trades/day > 2
    req3 = df['est_trades_per_day'] > 2
    print(f"Trades/day > 2: {req3.sum()}/{len(df)} models ({100*req3.sum()/len(df):.1f}%)")
    
    # Requirement 4: 5/5 successful folds
    req4 = df['n_successful_folds'] >= 5
    print(f"5/5 successful folds: {req4.sum()}/{len(df)} models ({100*req4.sum()/len(df):.1f}%)")
    
    # Requirement 5: Train-test gap < 0.40
    req5 = df['train_test_gap'] < 0.40
    print(f"Train-test gap < 0.40: {req5.sum()}/{len(df)} models ({100*req5.sum()/len(df):.1f}%)")
    
    # Apply all requirements
    all_reqs = req1 & req2 & req3 & req4 & req5
    df_filtered = df[all_reqs].copy()
    
    print(f"\n✅ Models meeting ALL requirements: {len(df_filtered)}/{initial_count} ({100*len(df_filtered)/initial_count:.1f}%)")
    
    return df_filtered


def calculate_composite_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate final composite score for ranking."""
    
    print("\n" + "=" * 80)
    print("CALCULATING COMPOSITE SCORES")
    print("=" * 80)
    
    # Component 1: AUC (40%)
    df['score_auc'] = df['median_test_auc']
    
    # Component 2: Signal Quality (30%) - % predictions >0.55 confidence
    df['score_signal_quality'] = df['pct_signals_055']
    
    # Component 3: Calibration (20%) - Inverse Brier score (lower is better)
    # Normalize to 0-1 range where 1 is best
    df['score_calibration'] = 1.0 - df['median_brier']
    
    # Component 4: Stability (10%) - Inverse of std(test_auc)
    # Avoid division by zero
    df['score_stability'] = 1.0 / (df['std_test_auc'] + 0.001)
    # Normalize stability to 0-1 range
    max_stability = df['score_stability'].max()
    df['score_stability'] = df['score_stability'] / max_stability
    
    # Final composite score
    df['final_score'] = (
        0.40 * df['score_auc'] +
        0.30 * df['score_signal_quality'] +
        0.20 * df['score_calibration'] +
        0.10 * df['score_stability']
    )
    
    print("\nScore Components (normalized 0-1):")
    print(f"  AUC (40%): mean={df['score_auc'].mean():.3f}, range=[{df['score_auc'].min():.3f}, {df['score_auc'].max():.3f}]")
    print(f"  Signal Quality (30%): mean={df['score_signal_quality'].mean():.3f}, range=[{df['score_signal_quality'].min():.3f}, {df['score_signal_quality'].max():.3f}]")
    print(f"  Calibration (20%): mean={df['score_calibration'].mean():.3f}, range=[{df['score_calibration'].min():.3f}, {df['score_calibration'].max():.3f}]")
    print(f"  Stability (10%): mean={df['score_stability'].mean():.3f}, range=[{df['score_stability'].min():.3f}, {df['score_stability'].max():.3f}]")
    print(f"\nFinal Score: mean={df['final_score'].mean():.3f}, range=[{df['final_score'].min():.3f}, {df['final_score'].max():.3f}]")
    
    return df


def check_diversity(candidates: pd.DataFrame) -> Dict[str, any]:
    """Check diversity of selected candidates."""
    
    diversity = {
        'labeling_methods': candidates['labeling_method'].value_counts().to_dict(),
        'cv_methods': candidates['cv_method'].value_counts().to_dict(),
        'sample_weights': candidates['sample_weight'].value_counts().to_dict(),
        'calibrations': candidates['calibration'].value_counts().to_dict(),
        'feature_sets': candidates['feature_set'].value_counts().to_dict(),
    }
    
    return diversity


def select_top_models(df: pd.DataFrame, n_models: int = 10) -> Tuple[pd.DataFrame, Dict]:
    """Select top N models with diversity check."""
    
    print("\n" + "=" * 80)
    print(f"SELECTING TOP {n_models} MODELS")
    print("=" * 80)
    
    # Sort by final score
    df_sorted = df.sort_values('final_score', ascending=False)
    
    # Select top N
    top_models = df_sorted.head(n_models).copy()
    
    print(f"\nTop {n_models} models selected:")
    for idx, row in top_models.iterrows():
        print(f"\n{row['exp_id']} (Score: {row['final_score']:.4f})")
        print(f"  AUC: {row['median_test_auc']:.3f}")
        print(f"  Signals >0.55: {100*row['pct_signals_055']:.1f}%")
        print(f"  Brier: {row['median_brier']:.3f}")
        print(f"  Std(AUC): {row['std_test_auc']:.3f}")
        print(f"  Trades/day: {row['est_trades_per_day']:.1f}")
        print(f"  Config: {row['labeling_method']} + {row['cv_method']} + {row['sample_weight']} + {row['calibration']}")
    
    # Check diversity
    print("\n" + "=" * 80)
    print("DIVERSITY CHECK")
    print("=" * 80)
    
    diversity = check_diversity(top_models)
    
    print("\nLabeling Methods:")
    for method, count in diversity['labeling_methods'].items():
        print(f"  {method}: {count}/{n_models} models")
    
    print("\nCV Methods:")
    for method, count in diversity['cv_methods'].items():
        print(f"  {method}: {count}/{n_models} models")
    
    print("\nSample Weights:")
    for weight, count in diversity['sample_weights'].items():
        print(f"  {weight}: {count}/{n_models} models")
    
    print("\nCalibration:")
    for cal, count in diversity['calibrations'].items():
        print(f"  {cal}: {count}/{n_models} models")
    
    print("\nFeature Sets:")
    for fset, count in diversity['feature_sets'].items():
        print(f"  {fset}: {count}/{n_models} models")
    
    # Diversity score: entropy-based
    total_entropy = 0
    for category, counts in diversity.items():
        probs = np.array(list(counts.values())) / n_models
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        total_entropy += entropy
    
    # Normalize by maximum possible entropy (log2(n_models) per category)
    max_entropy = 5 * np.log2(n_models)  # 5 categories
    diversity_score = total_entropy / max_entropy
    
    print(f"\n✅ Diversity Score: {diversity_score:.3f} (1.0 = maximum diversity)")
    
    if diversity_score < 0.5:
        print("⚠️  WARNING: Low diversity - models are too similar!")
        print("   Consider expanding selection or adjusting criteria.")
    elif diversity_score > 0.8:
        print("✅ EXCELLENT: High diversity - models use different approaches!")
    else:
        print("✅ GOOD: Moderate diversity - reasonable approach variation.")
    
    return top_models, diversity


def save_results(top_models: pd.DataFrame, diversity: Dict, all_qualified: pd.DataFrame, output_dir: str):
    """Save selection results."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save top 10 models (full details)
    top_models_file = output_dir / "top_10_selected_models.csv"
    top_models.to_csv(top_models_file, index=False)
    print(f"\n✅ Top 10 models saved to: {top_models_file}")
    
    # Save all qualified models (backup candidates)
    all_qualified_file = output_dir / "all_qualified_models.csv"
    all_qualified.to_csv(all_qualified_file, index=False)
    print(f"✅ All {len(all_qualified)} qualified models saved to: {all_qualified_file}")
    
    # Save selection summary
    summary = {
        'selection_date': pd.Timestamp.now().isoformat(),
        'total_experiments_analyzed': len(all_qualified),
        'n_selected': len(top_models),
        'selection_criteria': {
            'min_auc': 0.52,
            'min_signal_rate': 0.15,
            'min_trades_per_day': 2,
            'min_successful_folds': 5,
            'max_train_test_gap': 0.40,
        },
        'scoring_weights': {
            'auc': 0.40,
            'signal_quality': 0.30,
            'calibration': 0.20,
            'stability': 0.10,
        },
        'top_model': {
            'exp_id': top_models.iloc[0]['exp_id'],
            'batch': top_models.iloc[0]['batch'],
            'final_score': float(top_models.iloc[0]['final_score']),
            'auc': float(top_models.iloc[0]['median_test_auc']),
            'signal_rate': float(top_models.iloc[0]['pct_signals_055']),
            'brier': float(top_models.iloc[0]['median_brier']),
            'config': {
                'labeling': top_models.iloc[0]['labeling_method'],
                'cv': top_models.iloc[0]['cv_method'],
                'sample_weight': top_models.iloc[0]['sample_weight'],
                'calibration': top_models.iloc[0]['calibration'],
                'features': top_models.iloc[0]['feature_set'],
            }
        },
        'diversity': diversity,
        'score_statistics': {
            'mean': float(top_models['final_score'].mean()),
            'std': float(top_models['final_score'].std()),
            'min': float(top_models['final_score'].min()),
            'max': float(top_models['final_score'].max()),
        }
    }
    
    summary_file = output_dir / "selection_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Selection summary saved to: {summary_file}")
    
    # Save list of experiment IDs for Phase 2 validation
    exp_ids = top_models['exp_id'].tolist()
    exp_ids_file = output_dir / "top_10_exp_ids.json"
    with open(exp_ids_file, 'w') as f:
        json.dump(exp_ids, f, indent=2)
    print(f"✅ Experiment IDs saved to: {exp_ids_file}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Select top models for validation")
    parser.add_argument('--results-dir', type=str, default='ml_intraday_v3/experiments/results',
                        help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='ml_intraday_v3/experiments/results',
                        help='Directory to save selection results')
    parser.add_argument('--n-models', type=int, default=10,
                        help='Number of top models to select')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("PHASE 1: MODEL SELECTION")
    print("=" * 80)
    print(f"\nResults directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of models to select: {args.n_models}")
    
    # Load all experiments
    all_experiments = load_all_experiments(args.results_dir)
    
    # Apply minimum requirements
    qualified_models = apply_minimum_requirements(all_experiments)
    
    if len(qualified_models) == 0:
        print("\n❌ ERROR: No models meet minimum requirements!")
        print("   Consider relaxing criteria or running more experiments.")
        return
    
    if len(qualified_models) < args.n_models:
        print(f"\n⚠️  WARNING: Only {len(qualified_models)} models qualify (requested {args.n_models})")
        print(f"   Will select all {len(qualified_models)} qualified models.")
        args.n_models = len(qualified_models)
    
    # Calculate composite scores
    qualified_models = calculate_composite_score(qualified_models)
    
    # Select top N models
    top_models, diversity = select_top_models(qualified_models, args.n_models)
    
    # Save results
    save_results(top_models, diversity, qualified_models, args.output_dir)
    
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review selected models in: top_10_selected_models.csv")
    print("2. Prepare true OOS data (Feb 11 - Mar 31, 2026)")
    print("3. Run Phase 2 validation: python validate_true_oos.py")


if __name__ == '__main__':
    main()
