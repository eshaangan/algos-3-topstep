#!/usr/bin/env python3
"""
Test single experiment locally before GCP launch.

Usage:
    python test_single_experiment.py --config batch1_configs/batch1_exp_00001.json

This validates:
1. Config loading
2. Data loading
3. Feature generation
4. Labeling (triple_barrier OR trend_scanning)
5. Model training
6. Evaluation metrics
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from ml_intraday_v3.experiments.comprehensive_grid_search_v2 import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Test single experiment locally")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Experiment config JSON (e.g., batch1_configs/batch1_exp_00001.json)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Data directory",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="ml_intraday_v3/configs",
        help="Config directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file (default: /tmp/test_result.json)",
    )
    args = parser.parse_args()

    # Resolve paths
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = project_root / data_dir

    config_dir = Path(args.config_dir)
    if not config_dir.is_absolute():
        config_dir = project_root / config_dir

    output_path = Path(args.output) if args.output else Path("/tmp/test_result.json")

    # Validate files exist
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        return 1

    if not config_dir.exists():
        print(f"ERROR: Config directory not found: {config_dir}")
        return 1

    print("=" * 80)
    print("TESTING SINGLE EXPERIMENT")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Data dir: {data_dir}")
    print(f"Config dir: {config_dir}")
    print(f"Output: {output_path}")
    print()

    # Load and display experiment config
    with open(config_path, "r") as f:
        exp_config = json.load(f)

    print("Experiment Config:")
    print("-" * 80)
    print(json.dumps(exp_config, indent=2))
    print()

    # Run experiment
    print("=" * 80)
    print("RUNNING EXPERIMENT")
    print("=" * 80)

    try:
        result = run_experiment(exp_config, data_dir, config_dir)
    except Exception as e:
        print(f"\nERROR: Experiment failed with exception:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Save and display results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(json.dumps(result, indent=2))
    print()
    print(f"Results saved to: {output_path}")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    status = result.get("status", "unknown")
    print(f"Status: {status}")

    if status == "success":
        metrics = result.get("metrics", {})
        print(f"Train AUC: {metrics.get('train_auc', 'N/A'):.4f}")
        print(f"Val AUC: {metrics.get('val_auc', 'N/A'):.4f}")
        print(f"Test AUC: {metrics.get('test_auc', 'N/A'):.4f}")
        print(f"CV Splits: {metrics.get('cv_n_splits', 'N/A')}")
        print(f"Train Events: {metrics.get('n_train_events', 'N/A')}")
        print(f"Test Events: {metrics.get('n_test_events', 'N/A')}")
        print()
        print("✅ EXPERIMENT PASSED")
        return 0
    else:
        error = result.get("error", "Unknown error")
        print(f"Error: {error}")
        print()
        print("❌ EXPERIMENT FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
