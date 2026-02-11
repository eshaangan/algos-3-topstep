"""
Regime Detection Filter - Quick Win #3

Detects when market regime has shifted from training distribution.
Stops trading until regime stabilizes to prevent losses during regime shifts.

Research Context:
- Jan 2026 live trading failure likely due to regime shift
- Model trained on 2024-2025 data (80.3% accuracy)
- Model failed in Jan 2026 (35.5% win rate)
- No detection = kept trading and lost -$884.73

Methodology:
- Kolmogorov-Smirnov (KS) test on feature distributions
- Compares current market (last 100 bars) vs reference (last 90 days of training)
- Flags regime shift if >30% of features have significantly different distributions (p<0.05)

Expected Impact:
- Would have detected Jan 2026 regime shift
- Would have prevented -$884 loss by stopping trading
- Auto-resumes when regime stabilizes (features return to normal distribution)

Usage:
    # Initialize with training data
    detector = RegimeDetector(feature_cols=['log_return_1', 'vol_20', ...])
    detector.fit(training_features)  # Last 90 days of training data

    # Check before generating signals
    is_safe, shift_pct, shifted_features = detector.detect_shift(current_features)
    if not is_safe:
        logger.warning(f"Regime shift detected: {shift_pct:.1%} features shifted")
        # Don't trade until regime stabilizes
"""

import logging
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Detect market regime shifts using statistical tests on feature distributions.

    Uses Kolmogorov-Smirnov (KS) test to compare current feature distributions
    against reference (training) distributions. Flags regime shift if too many
    features have significantly different distributions.
    """

    def __init__(
        self,
        feature_cols: List[str],
        reference_window_days: int = 90,
        current_window_bars: int = 100,
        significance_level: float = 0.05,
        max_shifted_features_pct: float = 0.30,
        min_samples_per_feature: int = 30
    ):
        """
        Initialize regime detector.

        Args:
            feature_cols: List of feature column names to monitor
            reference_window_days: Days of training data to use as reference (default: 90)
            current_window_bars: Number of recent bars for current regime (default: 100)
            significance_level: P-value threshold for KS test (default: 0.05)
            max_shifted_features_pct: Max % of features that can shift before flagging (default: 0.30)
            min_samples_per_feature: Minimum samples required per feature for valid test (default: 30)
        """
        self.feature_cols = feature_cols
        self.reference_window_days = reference_window_days
        self.current_window_bars = current_window_bars
        self.significance_level = significance_level
        self.max_shifted_features_pct = max_shifted_features_pct
        self.min_samples_per_feature = min_samples_per_feature

        # State
        self.reference_data: pd.DataFrame = None
        self.is_fitted = False
        self.last_check_result: Dict = None

    def fit(self, historical_features: pd.DataFrame) -> None:
        """
        Fit detector on historical training data.

        Args:
            historical_features: DataFrame with feature columns and datetime index
                - Should contain at least reference_window_days of data
                - Will use the MOST RECENT reference_window_days as reference
        """
        if not isinstance(historical_features.index, pd.DatetimeIndex):
            raise ValueError("historical_features must have DatetimeIndex")

        # Use last N days as reference
        cutoff = historical_features.index.max() - pd.Timedelta(days=self.reference_window_days)
        self.reference_data = historical_features[historical_features.index >= cutoff].copy()

        # Validate reference data
        if len(self.reference_data) < self.min_samples_per_feature:
            logger.warning(
                f"Reference data has only {len(self.reference_data)} samples "
                f"(recommended: >{self.min_samples_per_feature})"
            )

        # Log fit statistics
        valid_features = []
        for col in self.feature_cols:
            if col in self.reference_data.columns:
                n_valid = self.reference_data[col].notna().sum()
                if n_valid >= self.min_samples_per_feature:
                    valid_features.append(col)

        logger.info(
            f"Regime detector fitted on {len(self.reference_data)} bars "
            f"({self.reference_data.index.min()} to {self.reference_data.index.max()})"
        )
        logger.info(
            f"Monitoring {len(valid_features)}/{len(self.feature_cols)} features "
            f"with sufficient data"
        )

        self.is_fitted = True

    def detect_shift(
        self,
        current_features: pd.DataFrame
    ) -> Tuple[bool, float, List[Dict]]:
        """
        Detect if current market regime has shifted from reference.

        Args:
            current_features: DataFrame with same features as reference
                - Should contain at least current_window_bars rows
                - Will use the MOST RECENT current_window_bars for testing

        Returns:
            Tuple of:
                - is_safe_to_trade (bool): True if regime is stable, False if shifted
                - shift_percentage (float): Percentage of features that shifted (0.0-1.0)
                - shifted_features (List[Dict]): List of features that shifted, each with:
                    - 'feature': Feature name
                    - 'ks_stat': KS test statistic
                    - 'ks_pvalue': KS test p-value
                    - 'ref_mean': Reference distribution mean
                    - 'curr_mean': Current distribution mean
                    - 'mean_change_pct': Percentage change in mean

        Raises:
            ValueError: If detector not fitted yet
        """
        if not self.is_fitted:
            raise ValueError("Must call fit() before detect_shift()")

        # Use last N bars for current regime
        current_data = current_features.tail(self.current_window_bars)

        if len(current_data) < self.min_samples_per_feature:
            logger.warning(
                f"Current data has only {len(current_data)} samples "
                f"(recommended: >{self.min_samples_per_feature})"
            )

        shifted_features = []
        tested_features = 0

        # Test each feature
        for feature in self.feature_cols:
            # Skip if feature not in data
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                continue

            # Get valid (non-NaN) values
            ref_vals = self.reference_data[feature].dropna()
            curr_vals = current_data[feature].dropna()

            # Skip if insufficient data
            if len(ref_vals) < self.min_samples_per_feature or len(curr_vals) < self.min_samples_per_feature:
                continue

            tested_features += 1

            # Perform KS test
            ks_stat, ks_pvalue = ks_2samp(ref_vals, curr_vals)

            # Check if distributions are significantly different
            if ks_pvalue < self.significance_level:
                ref_mean = ref_vals.mean()
                curr_mean = curr_vals.mean()
                mean_change_pct = (curr_mean - ref_mean) / abs(ref_mean) if ref_mean != 0 else float('inf')

                shifted_features.append({
                    'feature': feature,
                    'ks_stat': ks_stat,
                    'ks_pvalue': ks_pvalue,
                    'ref_mean': ref_mean,
                    'curr_mean': curr_mean,
                    'mean_change_pct': mean_change_pct
                })

        # Calculate shift percentage
        shift_pct = len(shifted_features) / tested_features if tested_features > 0 else 0.0

        # Determine if safe to trade
        is_safe = shift_pct < self.max_shifted_features_pct

        # Sort shifted features by KS statistic (most shifted first)
        shifted_features_sorted = sorted(
            shifted_features,
            key=lambda x: x['ks_stat'],
            reverse=True
        )

        # Store result for later inspection
        self.last_check_result = {
            'timestamp': current_data.index[-1] if len(current_data) > 0 else None,
            'is_safe': is_safe,
            'shift_pct': shift_pct,
            'tested_features': tested_features,
            'shifted_features_count': len(shifted_features),
            'shifted_features': shifted_features_sorted
        }

        # Log results
        if not is_safe:
            logger.warning(
                f"⚠️ REGIME SHIFT DETECTED: {shift_pct:.1%} of features shifted "
                f"({len(shifted_features)}/{tested_features})"
            )
            logger.warning(f"   Top shifted features:")
            for feat in shifted_features_sorted[:5]:
                logger.warning(
                    f"      {feat['feature']}: KS={feat['ks_stat']:.3f}, p={feat['ks_pvalue']:.4f}, "
                    f"mean {feat['ref_mean']:.4f} → {feat['curr_mean']:.4f} "
                    f"({feat['mean_change_pct']:+.1%})"
                )
        else:
            logger.info(
                f"✅ Regime stable: {shift_pct:.1%} of features shifted "
                f"({len(shifted_features)}/{tested_features}, threshold: {self.max_shifted_features_pct:.1%})"
            )

        return is_safe, shift_pct, shifted_features_sorted

    def get_last_result(self) -> Dict:
        """Get results from last detect_shift() call."""
        return self.last_check_result

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"RegimeDetector(status={status}, "
            f"features={len(self.feature_cols)}, "
            f"threshold={self.max_shifted_features_pct:.1%})"
        )


def apply_regime_filter(
    historical_features: pd.DataFrame,
    current_features: pd.DataFrame,
    signals_df: pd.DataFrame,
    feature_cols: List[str],
    reference_window_days: int = 90,
    current_window_bars: int = 100,
    max_shifted_features_pct: float = 0.30
) -> pd.DataFrame:
    """
    Filter signals based on regime detection.

    Convenience function that fits detector and filters signals in one call.

    Args:
        historical_features: Training feature data
        current_features: Current feature data
        signals_df: Trading signals to filter
        feature_cols: Features to monitor
        reference_window_days: Days of history for reference
        current_window_bars: Bars for current regime
        max_shifted_features_pct: Max % features that can shift

    Returns:
        Filtered signals (empty if regime shifted)
    """
    detector = RegimeDetector(
        feature_cols=feature_cols,
        reference_window_days=reference_window_days,
        current_window_bars=current_window_bars,
        max_shifted_features_pct=max_shifted_features_pct
    )

    # Fit on historical data
    detector.fit(historical_features)

    # Check current regime
    is_safe, shift_pct, shifted = detector.detect_shift(current_features)

    # Return signals only if safe
    if is_safe:
        return signals_df
    else:
        logger.warning(
            f"Regime filter: Blocking all signals due to regime shift "
            f"({shift_pct:.1%} features shifted)"
        )
        return pd.DataFrame()  # Return empty DataFrame


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    print("=" * 60)
    print("Regime Detector Test")
    print("=" * 60)

    # Create synthetic data
    np.random.seed(42)

    # Reference data (normal regime)
    dates_ref = pd.date_range('2024-01-01', periods=1000, freq='5min')
    ref_data = pd.DataFrame({
        'log_return_1': np.random.normal(0, 0.001, 1000),
        'vol_20': np.random.normal(0.01, 0.002, 1000),
        'rsi_14': np.random.normal(50, 10, 1000)
    }, index=dates_ref)

    # Current data (shifted regime - higher volatility)
    dates_curr = pd.date_range('2024-06-01', periods=200, freq='5min')
    curr_data = pd.DataFrame({
        'log_return_1': np.random.normal(0, 0.0015, 200),  # Higher std
        'vol_20': np.random.normal(0.015, 0.003, 200),     # Higher mean and std
        'rsi_14': np.random.normal(50, 10, 200)            # Same
    }, index=dates_curr)

    # Test detector
    detector = RegimeDetector(
        feature_cols=['log_return_1', 'vol_20', 'rsi_14'],
        reference_window_days=90,
        current_window_bars=100
    )

    print("\n1. Fitting on reference data...")
    detector.fit(ref_data)

    print("\n2. Detecting regime shift...")
    is_safe, shift_pct, shifted = detector.detect_shift(curr_data)

    print(f"\n3. Results:")
    print(f"   Safe to trade: {is_safe}")
    print(f"   Features shifted: {shift_pct:.1%}")
    print(f"\n4. Shifted features:")
    for feat in shifted:
        print(f"   {feat['feature']}: p={feat['ks_pvalue']:.4f}")
