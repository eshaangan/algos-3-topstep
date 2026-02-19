#!/usr/bin/env python3
"""
One-time HMM feature pre-computation script.

Run this ONCE in the background to generate HMM regime features for the entire dataset.
The output file can then be used by all future experiments without re-computing HMM.

Usage:
    python scripts/precompute_hmm_features.py \
        --input data/processed/MES_5min_Oct2024_Dec2025.parquet \
        --output data/processed/MES_5min_Oct2024_Dec2025_with_hmm.parquet

Runtime: ~5-10 hours (one-time cost)
Savings: Eliminates HMM computation from all future experiments (instant feature loading)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from features.hmm_regime import HMMRegimeDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def precompute_hmm_features(input_file: str, output_file: str, n_states: int = 2):
    """
    Pre-compute HMM regime features for the entire dataset.

    Args:
        input_file: Path to input parquet file (OHLCV data)
        output_file: Path to output parquet file (OHLCV + HMM features)
        n_states: Number of HMM states (default: 2 for bull/bear)
    """

    logger.info("=" * 80)
    logger.info("HMM FEATURE PRE-COMPUTATION")
    logger.info("=" * 80)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"HMM states: {n_states}")
    logger.info("")

    # Load data
    logger.info("Loading data...")
    df = pd.read_parquet(input_file)
    logger.info(f"Loaded {len(df):,} bars")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
    logger.info("")

    # Initialize HMM detector
    logger.info("Initializing HMM detector...")
    hmm_detector = HMMRegimeDetector(n_states=n_states)

    # Fit HMM and generate features
    logger.info("Fitting HMM model (this will take several hours)...")
    logger.info("Progress updates every 5,000 bars...")
    start_time = time.time()

    try:
        df_with_hmm = hmm_detector.fit_transform(df)

        elapsed = time.time() - start_time
        logger.info("")
        logger.info(f"✅ HMM computation complete!")
        logger.info(f"Runtime: {elapsed/3600:.2f} hours ({elapsed:.1f} seconds)")
        logger.info("")

        # Feature summary
        logger.info("HMM Features Generated:")
        hmm_cols = [col for col in df_with_hmm.columns if 'regime' in col.lower() or 'hmm' in col.lower()]
        for col in hmm_cols:
            logger.info(f"  - {col}")
        logger.info("")

        # Regime distribution
        if 'regime_state' in df_with_hmm.columns:
            regime_counts = df_with_hmm['regime_state'].value_counts().sort_index()
            logger.info("Regime Distribution:")
            for state, count in regime_counts.items():
                pct = 100 * count / len(df_with_hmm)
                logger.info(f"  State {state}: {count:,} bars ({pct:.1f}%)")
            logger.info("")

        # Save to parquet
        logger.info(f"Saving to {output_file}...")
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_hmm.to_parquet(output_file, compression='snappy')

        file_size = output_path.stat().st_size / (1024**2)  # MB
        logger.info(f"✅ Saved! File size: {file_size:.1f} MB")
        logger.info("")

        # Usage instructions
        logger.info("=" * 80)
        logger.info("✅ SUCCESS! HMM features cached.")
        logger.info("=" * 80)
        logger.info("")
        logger.info("To use these pre-computed features in experiments:")
        logger.info("")
        logger.info("1. Update experiment configs to use this file:")
        logger.info(f"   data_path: {output_file}")
        logger.info("")
        logger.info("2. Or merge HMM features into existing data:")
        logger.info("   df_hmm = pd.read_parquet('" + output_file + "')")
        logger.info("   hmm_features = df_hmm[[col for col in df_hmm.columns if 'regime' in col]]")
        logger.info("   df = df.join(hmm_features)")
        logger.info("")
        logger.info("3. HMM computation will now be INSTANT (just loading from disk)")
        logger.info("")

        return df_with_hmm

    except Exception as e:
        logger.error(f"❌ HMM computation failed: {e}")
        logger.exception(e)
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute HMM regime features for trading data'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input parquet file with OHLCV data'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output parquet file with OHLCV + HMM features'
    )
    parser.add_argument(
        '--n-states',
        type=int,
        default=2,
        help='Number of HMM states (default: 2 for bull/bear regimes)'
    )

    args = parser.parse_args()

    # Run pre-computation
    precompute_hmm_features(
        input_file=args.input,
        output_file=args.output,
        n_states=args.n_states
    )


if __name__ == '__main__':
    main()
