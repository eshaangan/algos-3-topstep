#!/usr/bin/env python3
"""
Diagnostic Test Runner - Model Simplification
Quickly tests different feature configurations to isolate the failure cause.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_train_with_config(config_name: str, description: str):
    """Run training with a specific configuration."""
    logger.info("=" * 80)
    logger.info(f"TEST: {description}")
    logger.info("=" * 80)

    # Backup current configs
    logger.info("Backing up current configs...")
    for fname in ["features.yaml", "training.yaml"]:
        src = Path(f"ml_intraday_v3/configs/{fname}")
        backup = Path(f"ml_intraday_v3/configs/{fname}.backup")
        if src.exists():
            import shutil
            shutil.copy(src, backup)

    # Apply test config if provided
    if config_name:
        logger.info(f"Applying config: {config_name}")
        # Copy test config to features.yaml
        src = Path(f"ml_intraday_v3/configs/{config_name}.yaml")
        dst = Path("ml_intraday_v3/configs/features.yaml")
        if src.exists():
            import shutil
            shutil.copy(src, dst)

    # Run training
    logger.info("Running training...")
    result = subprocess.run(
        [sys.executable, "-m", "ml_intraday_v3.train_balanced_model"],
        capture_output=True,
        text=True,
        cwd=Path.cwd()
    )

    # Parse AUC from output
    test_auc = None
    for line in result.stdout.split('\n'):
        if 'Test AUC:' in line:
            try:
                test_auc = float(line.split('Test AUC:')[1].strip().split()[0])
            except:
                pass

    # Restore configs
    logger.info("Restoring original configs...")
    for fname in ["features.yaml", "training.yaml"]:
        backup = Path(f"ml_intraday_v3/configs/{fname}.backup")
        dst = Path(f"ml_intraday_v3/configs/{fname}")
        if backup.exists():
            import shutil
            shutil.move(backup, dst)

    return {
        'config': config_name or 'current',
        'description': description,
        'test_auc': test_auc,
        'timestamp': datetime.now().isoformat(),
        'stdout': result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
    }


def main():
    logger.info("="*80)
    logger.info("DIAGNOSTIC TEST SUITE - MODEL SIMPLIFICATION")
    logger.info("="*80)

    results = []

    # Test 1: Current failed model (Phase 1-3)
    logger.info("\n\nTest 1: Current model (normalized + momentum + 6mo window)")
    logger.info("Expected: AUC ~0.49 (failed)")
    result1 = run_train_with_config(None, "Current (Phase 1-3)")
    results.append(result1)
    logger.info(f"  Result: Test AUC = {result1['test_auc']}")

    # Test 2: Revert to baseline (no normalization, no momentum)
    logger.info("\n\nTest 2: Baseline (no normalization, no momentum, original features)")
    logger.info("Expected: If AUC > 0.52, normalization broke it. If AUC ~0.49, baseline had no edge.")
    result2 = run_train_with_config("features_simplified", "Baseline (Pre-Phase-1)")
    results.append(result2)
    logger.info(f"  Result: Test AUC = {result2['test_auc']}")

    # Save results
    output_path = Path("ml_intraday_v3/diagnostics/diagnostic_test_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC SUMMARY")
    logger.info("="*80)
    logger.info(f"Test 1 (Current):  AUC = {result1['test_auc']}")
    logger.info(f"Test 2 (Baseline): AUC = {result2['test_auc']}")

    if result2['test_auc'] and result1['test_auc']:
        delta = result2['test_auc'] - result1['test_auc']
        logger.info(f"\nDelta: {delta:+.4f}")

        if result2['test_auc'] > 0.52:
            logger.info("\n✅ CONCLUSION: Normalization broke the model.")
            logger.info("   RECOMMENDATION: Use baseline features (no normalization) + prune harmful features")
        elif delta > 0.02:
            logger.info("\n⚠️  CONCLUSION: Normalization hurt performance but baseline is weak.")
            logger.info("   RECOMMENDATION: Revert normalization, then run feature selection")
        else:
            logger.info("\n❌ CONCLUSION: Baseline model has no edge (AUC < 0.52).")
            logger.info("   RECOMMENDATION: Fundamental redesign needed - different features, labels, or timeframe")

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
