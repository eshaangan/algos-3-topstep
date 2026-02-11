"""
Analyze and rank grid search results.

Downloads results from GCS, aggregates metrics, selects top configurations.
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_gcs_results(gcs_prefix: str, local_dir: Path):
    """Download all result JSON files from GCS."""
    local_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = ['gsutil', '-m', 'cp', f'{gcs_prefix}*.json', str(local_dir)]
    
    logger.info(f"Downloading results from {gcs_prefix} to {local_dir}")
    subprocess.run(cmd, check=True)


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""
    results = []
    
    for json_file in results_dir.glob('*.json'):
        with open(json_file, 'r') as f:
            result = json.load(f)
        
        if result['status'] != 'SUCCESS':
            logger.warning(f"Skipping failed experiment: {result['exp_id']}")
            continue
        
        # Flatten for DataFrame
        row = {
            'exp_id': result['exp_id'],
            'phase': result['config']['phase'],
            'model_name': result['config']['model_name'],
            'feature_set_name': result['config']['feature_set_name'],
            'training_window_months': result['config']['training_window_months'],
            'pt': result['config']['labeling']['pt'],
            'sl': result['config']['labeling']['sl'],
            'hz': result['config']['labeling']['hz'],
            'sample_weight': result['config']['sample_weight'],
            'calibration': result['config']['calibration'],
            'runtime_seconds': result['runtime_seconds'],
            **result['summary']  # Unpack summary metrics
        }
        
        results.append(row)
    
    df = pd.DataFrame(results)
    logger.info(f"Loaded {len(df)} successful experiments")
    
    return df


def rank_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank experiments by composite score.
    
    Ranking criteria (in order):
    1. median_test_auc (higher is better)
    2. median_pct_signals_above_055 (higher is better)
    3. mean_train_test_gap (lower is better, penalize overfitting)
    4. std_test_auc (lower is better, prefer stability)
    """
    # Filter: only keep configs with test AUC > 0.51
    df = df[df['median_test_auc'] > 0.51].copy()
    
    if len(df) == 0:
        logger.warning("No experiments with median_test_auc > 0.51!")
        return df
    
    # Create composite score
    # Normalize metrics to [0, 1] range
    df['auc_score'] = (df['median_test_auc'] - df['median_test_auc'].min()) / (df['median_test_auc'].max() - df['median_test_auc'].min() + 1e-6)
    df['signal_score'] = (df['median_pct_signals_above_055'] - df['median_pct_signals_above_055'].min()) / (df['median_pct_signals_above_055'].max() - df['median_pct_signals_above_055'].min() + 1e-6)
    df['gap_score'] = 1 - (df['mean_train_test_gap'] - df['mean_train_test_gap'].min()) / (df['mean_train_test_gap'].max() - df['mean_train_test_gap'].min() + 1e-6)
    df['stability_score'] = 1 - (df['std_test_auc'] - df['std_test_auc'].min()) / (df['std_test_auc'].max() - df['std_test_auc'].min() + 1e-6)
    
    # Weighted composite: AUC 40%, Signals 30%, Gap 20%, Stability 10%
    df['composite_score'] = (
        0.40 * df['auc_score'] +
        0.30 * df['signal_score'] +
        0.20 * df['gap_score'] +
        0.10 * df['stability_score']
    )
    
    # Sort by composite score
    df = df.sort_values('composite_score', ascending=False)
    
    return df


def print_summary(df: pd.DataFrame, top_k: int = 20):
    """Print summary table of top results."""
    print(f"\n{'='*120}")
    print(f"Top {top_k} Configurations")
    print(f"{'='*120}")
    print(f"{'Rank':<6} {'Exp ID':<20} {'Test AUC':<10} {'Sig>0.55':<10} {'Gap':<8} {'Model':<18} {'Features':<12} {'Window':<8}")
    print(f"{'-'*120}")
    
    for i, row in df.head(top_k).iterrows():
        rank = df.index.get_loc(i) + 1
        print(f"{rank:<6} {row['exp_id']:<20} {row['median_test_auc']:<10.3f} {row['median_pct_signals_above_055']:<10.1%} "
              f"{row['mean_train_test_gap']:<8.3f} {row['model_name']:<18} {row['feature_set_name']:<12} {row['training_window_months']:<8}mo")
    
    print(f"{'='*120}\n")


def save_top_configs(df: pd.DataFrame, output_path: Path, top_k: int = 20):
    """Save top K configurations to JSON for next phase."""
    top_configs = []
    
    for i, row in df.head(top_k).iterrows():
        # Reconstruct config dict
        config = {
            'exp_id': row['exp_id'],
            'phase': row['phase'],
            'model_name': row['model_name'],
            'model_params': {},  # Would need to store this in results
            'feature_set_name': row['feature_set_name'],
            'feature_set': None,  # Would need to store this in results
            'training_window_months': row['training_window_months'],
            'labeling': {'pt': row['pt'], 'sl': row['sl'], 'hz': row['hz']},
            'sample_weight': row['sample_weight'],
            'calibration': row['calibration'],
            'metrics': {
                'median_test_auc': row['median_test_auc'],
                'median_pct_signals_above_055': row['median_pct_signals_above_055'],
                'mean_train_test_gap': row['mean_train_test_gap'],
                'composite_score': row['composite_score']
            }
        }
        
        top_configs.append(config)
    
    with open(output_path, 'w') as f:
        json.dump(top_configs, f, indent=2)
    
    logger.info(f"Saved top {top_k} configs to {output_path}")


def quick_stats(gcs_prefix: str, total_experiments: int):
    """Print quick stats without downloading all files."""
    cmd = ['gsutil', 'ls', f'{gcs_prefix}*.json']
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    completed_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
    completed = len(completed_files)
    
    print(f"\nQuick Stats:")
    print(f"  Completed: {completed}/{total_experiments} ({100*completed/total_experiments:.1f}%)")
    print(f"  Remaining: {total_experiments - completed}")
    
    # Estimate ETA based on first 10% completion
    if completed >= total_experiments * 0.1:
        print(f"  Status: Early results available")
    elif completed >= total_experiments * 0.5:
        print(f"  Status: Halfway complete")
    elif completed >= total_experiments * 0.9:
        print(f"  Status: Nearly complete")
    else:
        print(f"  Status: In progress")
    print()


def main():
    parser = argparse.ArgumentParser(description="Analyze grid search results")
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3], help='Phase number')
    parser.add_argument('--top-k', type=int, default=20, help='Number of top configs to select')
    parser.add_argument('--bucket', type=str, default='gs://trading-algo-3', help='GCS bucket')
    parser.add_argument('--quick-stats', action='store_true', help='Just show quick stats, don\'t download')
    parser.add_argument('--total-experiments', type=int, help='Total experiments (for quick-stats)')
    
    args = parser.parse_args()
    
    gcs_prefix = f"{args.bucket}/experiment-results/phase{args.phase}/"
    
    # Quick stats mode
    if args.quick_stats:
        if not args.total_experiments:
            logger.error("--total-experiments required for --quick-stats")
            return
        quick_stats(gcs_prefix, args.total_experiments)
        return
    
    # Full analysis mode
    results_dir = Path(f'results/phase{args.phase}')
    
    # Download results
    download_gcs_results(gcs_prefix, results_dir)
    
    # Load and rank
    df = load_results(results_dir)
    
    if len(df) == 0:
        logger.error("No successful experiments found!")
        return
    
    df_ranked = rank_results(df)
    
    # Print summary
    print_summary(df_ranked, args.top_k)
    
    # Save top configs for next phase
    output_file = Path(f'results/phase{args.phase}_top{args.top_k}.json')
    save_top_configs(df_ranked, output_file, args.top_k)
    
    # Save full results to CSV
    csv_file = Path(f'results/phase{args.phase}_all_results.csv')
    df_ranked.to_csv(csv_file, index=False)
    logger.info(f"Saved full results to {csv_file}")


if __name__ == '__main__':
    main()
