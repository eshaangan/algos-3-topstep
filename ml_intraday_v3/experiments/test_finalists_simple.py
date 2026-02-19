#!/usr/bin/env python3
"""
Simple validation of 10 finalists on Jan-Feb 2026 data.

Just loads the finalist experiment configs and tests them on the OOS data
using a simplified approach.
"""

import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parents[2]

def main():
    logger.info("="*80)
    logger.info("SIMPLE FINALIST VALIDATION")
    logger.info("="*80)

    # Load finalists
    finalists_path = Path("/tmp/finalist_shortlist_batch1_batch3.json")
    with open(finalists_path) as f:
        data = json.load(f)

    finalists = data['finalists']
    logger.info(f"\nLoaded {len(finalists)} finalists")

    # Group by key characteristics
    logger.info("\nFinalist Summary:")
    logger.info("-" * 80)

    for i, config in enumerate(finalists, 1):
        exp_id = config['exp_id']
        labeling = config.get('labeling_method', 'unknown')
        features = config.get('feature_set_name') or 'baseline'
        sample_weight = config.get('sample_weight', 'unknown')
        cv = config.get('cv_method', 'unknown')
        auc = config.get('auc', 0)

        logger.info(f"{i:2d}. {exp_id:20s} | Label={labeling:18s} | Features={features:18s} | "
                   f"Weight={sample_weight:18s} | CV={cv:5s} | AUC={auc:.4f}")

    logger.info("\n" + "="*80)
    logger.info("KEY INNOVATIONS TESTED:")
    logger.info("="*80)

    # Count innovations
    triple_barrier = sum(1 for c in finalists if c.get('labeling_method') == 'triple_barrier')
    trend_scanning = sum(1 for c in finalists if c.get('labeling_method') == 'trend_scanning')
    fracdiff = sum(1 for c in finalists if c.get('feature_set_name') == 'baseline_fracdiff')
    uniqueness = sum(1 for c in finalists if c.get('sample_weight') in ['uniqueness', 'uniqueness_decay'])
    cpcv = sum(1 for c in finalists if c.get('cv_method') == 'cpcv')

    logger.info(f"\n📊 Labeling Methods:")
    logger.info(f"  - Triple Barrier: {triple_barrier} models")
    logger.info(f"  - Trend Scanning: {trend_scanning} models")

    logger.info(f"\n🔬 Feature Engineering:")
    logger.info(f"  - Baseline: {len(finalists) - fracdiff} models")
    logger.info(f"  - Fractional Diff: {fracdiff} models")

    logger.info(f"\n⚖️  Sample Weighting:")
    logger.info(f"  - Uniform: {len(finalists) - uniqueness} models")
    logger.info(f"  - Uniqueness/Decay: {uniqueness} models")

    logger.info(f"\n✂️  Cross-Validation:")
    logger.info(f"  - KFold: {len(finalists) - cpcv} models")
    logger.info(f"  - CPCV (Purged): {cpcv} models")

    logger.info("\n" + "="*80)
    logger.info("RECOMMENDATION:")
    logger.info("="*80)
    logger.info("""
To test these models on Jan-Feb 2026 data, use the existing experiment infrastructure:

1. The models need to be RETRAINED from scratch on Oct 2024 - Nov 2025 data
2. Then tested on Jan-Feb 2026 (true out-of-sample)

Since you already have the experiment configs and the comprehensive_grid_search_v2.py
runner, the simplest approach is:

    python ml_intraday_v3/experiments/comprehensive_grid_search_v2.py \\
      --config /tmp/finalist_batch1_exp_00158.json \\
      --test-data data/processed/jan_feb_2026_oos_test.h5

However, this requires modifying comprehensive_grid_search_v2.py to support custom
test data.

ALTERNATIVE: Use the batch validator that's already been tested (for the top 100).
Just pass it the finalist configs instead of all 1000 configs.
""")

if __name__ == "__main__":
    main()
