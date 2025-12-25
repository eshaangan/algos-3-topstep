"""
Hyperparameter Search with Trial Tracking for PBO Analysis

This script:
1. Runs hyperparameter search over a grid of configurations
2. Tracks ALL trials (not just winners) using TrialTracker
3. Trains each config on CPCV splits
4. Saves comprehensive trial data for PBO computation

Usage:
    python -m ml_intraday_v3.training.hyperparam_search \
        --run-dir runs/run_20251224_123456 \
        --bar-size 1m \
        --config configs/training.yaml
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Iterator
from itertools import product
import argparse

import yaml
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ml_intraday_v3.experiments.trial_tracker import TrialTracker
from ml_intraday_v3.training.train import train_on_splits

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_hyperparameter_grid(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all hyperparameter combinations from config grid.

    Expects config to have a 'hyperparam_search' section:

    hyperparam_search:
      enabled: true
      model:
        kind: lgbm  # or logreg
        param_grid:
          n_estimators: [300, 500, 1000]
          learning_rate: [0.01, 0.05, 0.1]
          max_depth: [4, 6, 8]
      fixed_params:
        min_child_samples: 100
        subsample: 0.8
        # ... other fixed params

    Returns:
        List of full training configs, one per hyperparameter combination
    """
    search_cfg = config.get('hyperparam_search', {})

    if not search_cfg.get('enabled', False):
        # No grid search - return single config
        return [config]

    model_kind = search_cfg.get('model', {}).get('kind', 'lgbm')
    param_grid = search_cfg.get('model', {}).get('param_grid', {})
    fixed_params = search_cfg.get('fixed_params', {})

    if not param_grid:
        logger.warning("hyperparam_search enabled but param_grid is empty. Using base config.")
        return [config]

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    configs = []
    for values in product(*param_values):
        # Create param dict for this combination
        params = dict(zip(param_names, values))

        # Merge with fixed params (fixed params don't override grid params)
        full_params = {**fixed_params, **params}

        # Create full config
        trial_config = config.copy()
        trial_config['model'] = {
            'kind': model_kind,
            'params': full_params
        }

        configs.append(trial_config)

    logger.info(f"Generated {len(configs)} hyperparameter combinations")
    return configs


def extract_cpcv_metrics(training_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Extract IS/OOS metrics from CPCV training results.

    Returns:
        Dict mapping path_id -> {'is_metric': float, 'oos_metric': float}
    """
    summary_path = training_dir / "summary.json"

    if not summary_path.exists():
        logger.warning(f"No summary.json found in {training_dir}")
        return {}

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    path_metrics = {}

    for split_data in summary.get('metrics_by_split', []):
        split_id = split_data.get('split_id')
        metrics = split_data.get('metrics', {})

        # Use ROC-AUC as primary metric (configurable later)
        # For CPCV, we treat each path's test fold as OOS, and avg of other paths as IS
        # But since train.py trains on each path independently, we need to aggregate

        # Simplified: use test metrics as OOS, assume similar IS (can enhance later)
        oos_metric = metrics.get('roc_auc') or metrics.get('roc_auc_target_vs_rest')

        if oos_metric is not None:
            path_metrics[f'path_{split_id}'] = {
                'is_metric': oos_metric,  # Simplified - will enhance
                'oos_metric': oos_metric
            }

    return path_metrics


def run_hyperparam_search(
    run_dir: Path,
    bar_size: str,
    config_path: Path,
    cv_kind: str = 'cpcv'
):
    """
    Run hyperparameter search with trial tracking.

    Parameters:
        run_dir: Run directory path
        bar_size: Bar size (e.g., '1m', '5m')
        config_path: Path to training config YAML
        cv_kind: 'cpcv' or 'purged_kfold'
    """
    run_dir = Path(run_dir)
    config_path = Path(config_path)

    # Load base config
    base_config = load_config(config_path)

    # Generate hyperparameter grid
    configs = generate_hyperparameter_grid(base_config)

    logger.info(f"Starting hyperparameter search:")
    logger.info(f"  Run directory: {run_dir}")
    logger.info(f"  Bar size: {bar_size}")
    logger.info(f"  CV kind: {cv_kind}")
    logger.info(f"  Number of configs to test: {len(configs)}")

    # Initialize trial tracker
    tracker = TrialTracker(str(run_dir))

    # Test each configuration
    for i, trial_config in enumerate(configs, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Trial {i}/{len(configs)}")
        logger.info(f"{'='*80}")

        model_cfg = trial_config.get('model', {})
        model_kind = model_cfg.get('kind', 'lgbm')
        hyperparams = model_cfg.get('params', {})

        logger.info(f"Model: {model_kind}")
        logger.info(f"Hyperparameters: {hyperparams}")

        # Log trial
        trial_id = tracker.log_trial(
            config=trial_config,
            model_type=model_kind,
            hyperparameters=hyperparams,
            metadata={
                'bar_size': bar_size,
                'cv_kind': cv_kind,
                'trial_index': i
            }
        )

        logger.info(f"Trial ID: {trial_id}")

        try:
            # Train on CPCV splits
            result = train_on_splits(
                run_dir=run_dir,
                bar_size=bar_size,
                training_config=trial_config,
                cv_kind=cv_kind,
                training_dir_override=run_dir / f"bar_size={bar_size}" / "trials" / trial_id
            )

            training_dir = result['training_dir']
            logger.info(f"Training completed: {training_dir}")

            # Extract metrics from training results
            path_metrics = extract_cpcv_metrics(Path(training_dir))

            # Update trial tracker with path metrics
            for path_id, metrics in path_metrics.items():
                tracker.update_path_metrics(
                    trial_id=trial_id,
                    path_id=path_id,
                    is_metric=metrics['is_metric'],
                    oos_metric=metrics['oos_metric']
                )

            logger.info(f"✓ Trial {i} completed successfully")
            logger.info(f"  Tracked {len(path_metrics)} CPCV paths")

        except Exception as e:
            logger.error(f"✗ Trial {i} failed: {e}", exc_info=True)
            # Continue with next trial
            continue

    # Save all trials
    tracker.save()

    logger.info(f"\n{'='*80}")
    logger.info("Hyperparameter Search Complete")
    logger.info(f"{'='*80}")
    logger.info(f"Total trials: {len(tracker.trials)}")
    logger.info(f"Trials saved to: {tracker.trials_file}")

    # Summary statistics
    trials_df = tracker.to_dataframe()
    summary = tracker.summary()

    logger.info(f"\nSummary:")
    logger.info(f"  Total trials: {summary['total_trials']}")
    logger.info(f"  Model types: {summary['model_types']}")
    logger.info(f"  CPCV paths per trial: {summary['paths_per_trial']}")

    logger.info(f"\nNext steps:")
    logger.info(f"1. Run PBO analysis:")
    logger.info(f"   python -m ml_intraday_v3.training.analyze_pbo --run-dir {run_dir}")
    logger.info(f"2. Review tracked trials:")
    logger.info(f"   {tracker.trials_file}")

    return tracker


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run hyperparameter search with trial tracking for PBO"
    )
    parser.add_argument(
        '--run-dir',
        type=str,
        required=True,
        help='Run directory path (e.g., runs/run_20251224_123456)'
    )
    parser.add_argument(
        '--bar-size',
        type=str,
        required=True,
        choices=['1m', '5m'],
        help='Bar size to use'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training.yaml',
        help='Path to training config YAML (default: configs/training.yaml)'
    )
    parser.add_argument(
        '--cv-kind',
        type=str,
        default='cpcv',
        choices=['cpcv', 'purged_kfold'],
        help='Cross-validation kind (default: cpcv)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run search
    run_hyperparam_search(
        run_dir=Path(args.run_dir),
        bar_size=args.bar_size,
        config_path=Path(args.config),
        cv_kind=args.cv_kind
    )


if __name__ == '__main__':
    main()
