"""
HMM Regime Detection Demo Script.

This script demonstrates how to use the HMM regime detector for market regime
identification. Can be imported and run from notebooks.

Usage:
    from ml_intraday_v3.features.hmm_demo import run_hmm_demo
    run_hmm_demo(bars_df)  # Pass your bars DataFrame

Or run standalone:
    python -m ml_intraday_v3.features.hmm_demo --help
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_hmm_demo(
    bars_df: pd.DataFrame,
    n_states: int = 2,
    min_train_samples: int = 252,
    refit_every: int = 21,
    plot: bool = True,
    figsize: Tuple[int, int] = (14, 10),
) -> dict:
    """
    Run HMM regime detection demo on bar data.

    Parameters
    ----------
    bars_df : pd.DataFrame
        OHLCV bar data with DatetimeIndex and 'close' column.
    n_states : int, default=2
        Number of hidden states (2 = bull/bear).
    min_train_samples : int, default=252
        Minimum samples before HMM fitting starts.
    refit_every : int, default=21
        Refit HMM every N bars.
    plot : bool, default=True
        Whether to create visualizations.
    figsize : tuple, default=(14, 10)
        Figure size for plots.

    Returns
    -------
    results : dict
        Dictionary containing:
        - regime_states: pd.Series of state assignments
        - regime_probs: pd.DataFrame of state probabilities
        - transition_matrix: np.ndarray
        - emission_params: dict with mean/std per state
        - expected_durations: dict with expected regime duration in bars
    """
    try:
        from ml_intraday_v3.features.hmm_regime import (
            HMMRegimeDetector,
            get_regime_spans,
            compare_hmm_models,
        )
    except ImportError as e:
        raise ImportError(
            f"HMM modules not available: {e}\n"
            "Install hmmlearn: pip install hmmlearn>=0.3.0"
        )

    print("=" * 80)
    print("HMM REGIME DETECTION DEMO")
    print("=" * 80)

    # Compute returns
    if "close" not in bars_df.columns:
        raise ValueError("bars_df must have 'close' column")

    returns = bars_df["close"].pct_change().fillna(0)
    print(f"\nData: {len(returns)} bars from {returns.index[0]} to {returns.index[-1]}")
    print(f"Returns: mean={returns.mean()*100:.4f}%, std={returns.std()*100:.4f}%")

    # Fit HMM
    print(f"\n{'='*40}")
    print(f"Fitting {n_states}-state Gaussian HMM")
    print(f"{'='*40}")

    hmm = HMMRegimeDetector(
        n_states=n_states,
        min_samples=min_train_samples,
    )

    regime_states, regime_probs = hmm.predict_expanding(
        returns=returns,
        min_train_samples=min_train_samples,
        refit_every=refit_every,
    )

    # Analyze results
    print(f"\n{'='*40}")
    print("TRANSITION MATRIX")
    print(f"{'='*40}")

    # Fit once on full data for transition matrix
    hmm.fit(returns.dropna())
    trans_mat = hmm.get_transition_matrix()

    state_labels = ["Bear", "Bull"] if n_states == 2 else [f"State_{i}" for i in range(n_states)]

    print("\nP(next_state | current_state):")
    for i, from_label in enumerate(state_labels):
        for j, to_label in enumerate(state_labels):
            print(f"  P({to_label} | {from_label}) = {trans_mat[i, j]:.4f}")

    print(f"\n{'='*40}")
    print("EMISSION PARAMETERS (per-state return distribution)")
    print(f"{'='*40}")

    emission_params = hmm.get_emission_params()
    for label, params in emission_params.items():
        print(f"\n{label.upper()}:")
        print(f"  Mean return: {params['mean']*100:.4f}% per bar")
        print(f"  Std return:  {params['std']*100:.4f}% per bar")

    print(f"\n{'='*40}")
    print("EXPECTED REGIME DURATION")
    print(f"{'='*40}")

    durations = hmm.expected_regime_duration()
    for label, dur in durations.items():
        print(f"  {label}: {dur:.1f} bars")

    print(f"\n{'='*40}")
    print("REGIME DISTRIBUTION")
    print(f"{'='*40}")

    regime_counts = regime_states.dropna().value_counts().sort_index()
    total = regime_counts.sum()
    for state, count in regime_counts.items():
        label = state_labels[int(state)] if int(state) < len(state_labels) else f"State_{int(state)}"
        print(f"  {label}: {count:,} bars ({count/total*100:.1f}%)")

    # Model selection
    print(f"\n{'='*40}")
    print("MODEL SELECTION (AIC/BIC)")
    print(f"{'='*40}")

    comparison = compare_hmm_models(returns, state_range=(2, 5))
    print("\nComparing 2, 3, 4 state models:")
    print(comparison.to_string(index=False))

    # Plotting
    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

            # 1. Price with regime coloring
            ax = axes[0]
            ax.plot(bars_df.index, bars_df["close"], "k-", linewidth=0.5, alpha=0.8)
            ax.set_ylabel("Price")
            ax.set_title("Price with HMM Regime Coloring")

            # Color background by regime
            spans = get_regime_spans(regime_states)
            colors = {0: "red", 1: "green"}
            for start_idx, end_idx, state in spans:
                if pd.isna(state):
                    continue
                start_time = regime_states.index[start_idx]
                end_time = regime_states.index[end_idx]
                color = colors.get(int(state), "gray")
                ax.axvspan(start_time, end_time, alpha=0.2, color=color)

            # 2. Regime probabilities
            ax = axes[1]
            if "prob_bull" in regime_probs.columns:
                ax.plot(regime_probs.index, regime_probs["prob_bull"], "g-", label="P(Bull)", alpha=0.8)
                ax.plot(regime_probs.index, regime_probs["prob_bear"], "r-", label="P(Bear)", alpha=0.8)
            else:
                for col in regime_probs.columns:
                    if col.startswith("prob_state_"):
                        ax.plot(regime_probs.index, regime_probs[col], label=col, alpha=0.8)
            ax.set_ylabel("Probability")
            ax.set_title("HMM State Probabilities")
            ax.legend(loc="upper right")
            ax.set_ylim(0, 1)

            # 3. Regime states
            ax = axes[2]
            valid_states = regime_states.dropna()
            ax.scatter(valid_states.index, valid_states.values, c=valid_states.values,
                      cmap="RdYlGn", s=1, alpha=0.5)
            ax.set_ylabel("State")
            ax.set_title("HMM Regime States (0=Bear, 1=Bull)")
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["Bear", "Bull"])

            # 4. Rolling volatility
            ax = axes[3]
            rolling_vol = returns.rolling(20).std() * np.sqrt(252) * 100
            ax.plot(rolling_vol.index, rolling_vol, "b-", linewidth=0.5, alpha=0.8)
            ax.set_ylabel("Annualized Vol (%)")
            ax.set_title("Rolling 20-bar Volatility")
            ax.set_xlabel("Date")

            plt.tight_layout()
            plt.show()

            print("\n[Plots displayed]")

        except ImportError:
            print("\nMatplotlib not available - skipping plots")

    # Return results
    results = {
        "regime_states": regime_states,
        "regime_probs": regime_probs,
        "transition_matrix": trans_mat,
        "emission_params": emission_params,
        "expected_durations": durations,
        "model_comparison": comparison,
        "hmm_detector": hmm,
    }

    print(f"\n{'='*80}")
    print("Demo complete! Results returned as dictionary.")
    print("Keys: regime_states, regime_probs, transition_matrix, emission_params, ...")
    print("=" * 80)

    return results


def demo_regime_weights(
    events_df: pd.DataFrame,
    regime_probs: pd.DataFrame,
    regime_states: pd.Series,
    target_regime: Optional[int] = None,
    plot: bool = True,
) -> pd.Series:
    """
    Demonstrate regime-based sample weighting.

    Parameters
    ----------
    events_df : pd.DataFrame
        Events with t0 timestamps.
    regime_probs : pd.DataFrame
        State probabilities from HMM.
    regime_states : pd.Series
        State assignments from HMM.
    target_regime : int, optional
        Target regime to weight towards.
    plot : bool, default=True
        Whether to create visualization.

    Returns
    -------
    w_regime : pd.Series
        Regime-based sample weights.
    """
    from ml_intraday_v3.weights.hmm_weights import (
        compute_hmm_regime_weights,
        analyze_regime_weight_distribution,
    )

    print("=" * 80)
    print("REGIME-BASED SAMPLE WEIGHTING DEMO")
    print("=" * 80)

    w_regime = compute_hmm_regime_weights(
        events_df=events_df,
        regime_probs=regime_probs,
        target_regime=target_regime,
        similarity_method="probability",
    )

    print(f"\nWeight Statistics:")
    print(f"  Mean:   {w_regime.mean():.4f}")
    print(f"  Std:    {w_regime.std():.4f}")
    print(f"  Min:    {w_regime.min():.4f}")
    print(f"  Max:    {w_regime.max():.4f}")

    print(f"\n{'='*40}")
    print("WEIGHT DISTRIBUTION BY REGIME")
    print(f"{'='*40}")

    analysis = analyze_regime_weight_distribution(
        w_regime=w_regime,
        regime_states=regime_states,
        events_df=events_df,
    )
    print(analysis.to_string())

    if plot:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(14, 6))

            # Weight distribution
            ax = axes[0]
            ax.hist(w_regime, bins=50, alpha=0.7, edgecolor="black")
            ax.axvline(w_regime.mean(), color="red", linestyle="--", label=f"Mean={w_regime.mean():.3f}")
            ax.set_xlabel("Regime Weight")
            ax.set_ylabel("Count")
            ax.set_title("Distribution of Regime-Based Sample Weights")
            ax.legend()

            # Weights over time
            ax = axes[1]
            event_times = pd.to_datetime(events_df["t0"])
            ax.scatter(event_times, w_regime, c=w_regime, cmap="RdYlGn", s=5, alpha=0.5)
            ax.set_xlabel("Event Time")
            ax.set_ylabel("Regime Weight")
            ax.set_title("Regime Weights Over Time (green=high, red=low)")

            plt.tight_layout()
            plt.show()

        except ImportError:
            print("\nMatplotlib not available - skipping plots")

    return w_regime


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HMM Regime Detection Demo")
    parser.add_argument("--data", type=str, help="Path to bars parquet file")
    parser.add_argument("--n-states", type=int, default=2, help="Number of HMM states")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting")

    args = parser.parse_args()

    if args.data:
        bars_df = pd.read_parquet(args.data)
        results = run_hmm_demo(
            bars_df=bars_df,
            n_states=args.n_states,
            plot=not args.no_plot,
        )
    else:
        print("Usage: python -m ml_intraday_v3.features.hmm_demo --data path/to/bars.parquet")
        print("\nOr import and use in notebook:")
        print("  from ml_intraday_v3.features.hmm_demo import run_hmm_demo")
        print("  results = run_hmm_demo(bars_df)")
