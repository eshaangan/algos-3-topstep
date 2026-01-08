#!/usr/bin/env python3
"""
Run GPU-accelerated HMM regime detection on full dataset.

Usage:
    python run_gpu_hmm.py --run-dir runs/v3_data_20260107_184235 --bar-size 1m
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run GPU HMM on full dataset")
    parser.add_argument("--run-dir", required=True, help="Run directory")
    parser.add_argument("--bar-size", default="1m", help="Bar size (1m or 5m)")
    parser.add_argument("--n-states", type=int, default=2, help="Number of HMM states")
    parser.add_argument("--refit-every", type=int, default=126, help="Refit every N bars")
    parser.add_argument("--rolling-window", type=int, default=252, help="Rolling window size")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, mps, cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    bar_dir = run_dir / f"bar_size={args.bar_size}"

    logger.info("=" * 80)
    logger.info("GPU HMM REGIME DETECTION")
    logger.info("=" * 80)
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Bar size: {args.bar_size}")
    logger.info(f"Device: {args.device}")

    # Load bars
    bars_path = bar_dir / "bars.parquet"
    logger.info(f"Loading bars from {bars_path}")
    bars_df = pd.read_parquet(bars_path)
    logger.info(f"Loaded {len(bars_df):,} bars")

    # Compute returns
    logger.info("Computing log returns...")
    returns = bars_df["close"].pct_change().fillna(0)
    logger.info(f"Returns shape: {returns.shape}")

    # Import GPU HMM
    from ml_intraday_v3.features.hmm_regime_gpu import HMMRegimeDetectorGPU

    # Initialize HMM
    logger.info(f"Initializing GPU HMM with {args.n_states} states...")
    hmm = HMMRegimeDetectorGPU(
        n_states=args.n_states,
        n_iter=50,
        device=args.device,
        min_samples=args.rolling_window,
    )

    # Run expanding window prediction
    logger.info("=" * 80)
    logger.info("Starting GPU HMM expanding window prediction...")
    logger.info(f"Rolling window: {args.rolling_window} bars")
    logger.info(f"Refit every: {args.refit_every} bars")
    logger.info("=" * 80)

    start_time = time.time()

    regime_states, regime_probs = hmm.predict_expanding_batched(
        returns=returns,
        min_train_samples=args.rolling_window,
        refit_every=args.refit_every,
        rolling_window_size=args.rolling_window,
    )

    elapsed = time.time() - start_time
    logger.info(f"GPU HMM completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

    # Save results
    logger.info("=" * 80)
    logger.info("Saving results...")

    # Combine states and probs
    regime_df = regime_probs.copy()
    regime_df["hmm_state"] = regime_states

    output_path = bar_dir / "hmm_regimes_gpu.parquet"
    regime_df.to_parquet(output_path)
    logger.info(f"Saved regime assignments to {output_path}")

    # Print statistics
    logger.info("=" * 80)
    logger.info("REGIME STATISTICS")
    logger.info("=" * 80)

    valid_states = regime_states.dropna()
    state_counts = valid_states.value_counts().sort_index()

    logger.info(f"Total bars with regime assignments: {len(valid_states):,}")
    for state, count in state_counts.items():
        pct = count / len(valid_states) * 100
        logger.info(f"  State {int(state)}: {count:,} bars ({pct:.1f}%)")

    # Transition analysis
    transitions = (valid_states.diff() != 0).sum()
    logger.info(f"Total regime transitions: {transitions:,}")
    logger.info(f"Average regime duration: {len(valid_states) / transitions:.1f} bars")

    # Return statistics by regime
    returns_with_regime = pd.DataFrame({
        "return": returns,
        "regime": regime_states
    }).dropna()

    logger.info("")
    logger.info("RETURNS BY REGIME:")
    for state in sorted(returns_with_regime["regime"].unique()):
        r = returns_with_regime[returns_with_regime["regime"] == state]["return"]
        logger.info(f"  State {int(state)}:")
        logger.info(f"    Mean: {r.mean()*100:.4f}%")
        logger.info(f"    Std:  {r.std()*100:.4f}%")

    logger.info("")
    logger.info("=" * 80)
    logger.info("GPU HMM COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output: {output_path}")
    logger.info(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
