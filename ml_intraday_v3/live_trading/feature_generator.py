"""
Real-time feature generator for live trading.

Wraps the offline feature builder to guarantee parity with training.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from ml_intraday_v3.features.build import build_features

logger = logging.getLogger(__name__)


class LiveFeatureGenerator:
    """
    Generates features in real-time from streaming bars.

    Uses the same feature calculation logic as the training pipeline
    to ensure consistency.
    """

    def __init__(
        self,
        feature_columns: List[str],
        bar_size: str,
        features_config_path: Optional[Path] = None,
    ):
        """
        Initialize the feature generator.

        Args:
            feature_columns: List of feature column names (from model bundle)
            bar_size: Bar size string (e.g., \"5m\")
            features_config_path: Path to features.yaml (defaults to project config)
        """
        self.feature_columns = feature_columns
        self.bar_size = bar_size

        cfg_path = features_config_path or Path(__file__).resolve().parents[1] / "configs" / "features.yaml"
        with open(cfg_path, "r") as f:
            self.features_config = yaml.safe_load(f)

        logger.info(
            f"LiveFeatureGenerator initialized with {len(feature_columns)} features, bar_size={bar_size}, config={cfg_path}"
        )

    def _prepare_bars(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize bars for offline builder:
        - ensure tz-aware and sorted
        - compute session time features in America/Chicago
        - convert index to UTC for builder
        """
        if bars_df.empty:
            return bars_df

        bars = bars_df.sort_index()
        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("America/Chicago")
        else:
            bars = bars.tz_convert("America/Chicago")

        # Add time columns in Chicago tz to match training intent
        bars = bars.copy()
        bars["minute_of_day"] = bars.index.hour * 60 + bars.index.minute
        bars["day_of_week"] = bars.index.dayofweek

        # Convert to UTC for offline builder expectations
        bars.index = bars.index.tz_convert("UTC")
        return bars

    def generate_features(self, bars_df: pd.DataFrame) -> pd.Series:
        """
        Generate features from a rolling window of bars (adapter to offline builder).
        """
        if bars_df.empty:
            logger.warning("Empty bars DataFrame")
            return pd.Series(dtype=float)

        if len(bars_df) < 30:
            logger.warning(f"Insufficient bars for feature calculation: {len(bars_df)} < 30")
            return pd.Series(dtype=float)

        bars_prepped = self._prepare_bars(bars_df)
        try:
            feats_df = build_features(
                bars_df=bars_prepped,
                bar_size=self.bar_size,
                config=self.features_config,
            )
        except Exception as exc:
            logger.error(f"Failed to build features: {exc}")
            return pd.Series(dtype=float)

        if feats_df.empty:
            logger.warning("Feature builder returned empty DataFrame")
            return pd.Series(dtype=float)

        # Take the last row (latest bar) and align to expected columns
        latest = feats_df.iloc[-1]
        feature_series = latest.reindex(self.feature_columns)

        # Training bundles merged with suffixes (_x/_y) may list e.g. vol_regime_x, vol_regime_y while
        # build_features only emits vol_regime. Fill suffixed names from the unsuffixed column.
        for col in self.feature_columns:
            if col not in feature_series.index:
                continue
            if pd.notna(feature_series[col]):
                continue
            if col.endswith("_x"):
                base = col[:-2]
            elif col.endswith("_y"):
                base = col[:-2]
            else:
                continue
            if base in latest.index and pd.notna(latest[base]):
                feature_series[col] = latest[base]

        # For DualSideModel, 'side' feature needs to be present but will be overwritten by the model
        # Add it as 0.0 (placeholder) if it's expected but not generated
        if 'side' in self.feature_columns and 'side' not in feats_df.columns:
            feature_series['side'] = 0.0

        logger.debug(
            f"Generated {len(feature_series)} features, "
            f"{feature_series.isna().sum()} NaN values"
        )

        return feature_series

    def check_feature_quality(self, features: pd.Series) -> Dict[str, bool]:
        """
        Check quality of generated features.

        Args:
            features: Series of feature values

        Returns:
            Dictionary of quality checks
        """
        checks = {}

        # Exclude 'side' from quality checks - it's not a real feature, just a training-time indicator
        # For DualSideModel, 'side' is expected to be missing/NaN
        features_to_check = features.drop("side", errors="ignore")

        # HMM regime features can be intermittently unavailable in live mode (e.g. after data gaps or
        # before enough history accumulates). The model bundle already includes an imputer, so we
        # treat NaNs in these specific columns as acceptable when HMM features are enabled.
        hmm_cfg = (self.features_config.get("hmm_regime", {}) or {}) if isinstance(self.features_config, dict) else {}
        hmm_enabled = bool(hmm_cfg.get("enabled", False))
        allowed_nan_cols = {
            "hmm_state",
            "prob_state_0",
            "prob_state_1",
            "prob_bull",
            "prob_bear",
        }
        if hmm_enabled:
            present_allowed = [c for c in allowed_nan_cols if c in features_to_check.index]
            if present_allowed:
                features_to_check = features_to_check.copy()
                # Keep values as-is; we only relax the health check for NaNs in these columns.

        # Check for NaN values
        nan_mask = features_to_check.isna()
        if hmm_enabled and allowed_nan_cols:
            # Ignore NaNs in the allowed HMM columns.
            for c in allowed_nan_cols:
                if c in features_to_check.index:
                    nan_mask.loc[c] = False
        nan_count = int(nan_mask.sum())
        checks["has_nan"] = nan_count > 0
        checks["nan_count"] = nan_count
        checks["nan_columns"] = features_to_check.index[nan_mask].tolist() if nan_count else []

        # Check for infinite values (coerce non-numeric to nan first)
        numeric = pd.to_numeric(features_to_check, errors="coerce")
        inf_mask = np.isinf(numeric.to_numpy())
        inf_count = int(inf_mask.sum())
        checks["has_inf"] = inf_count > 0
        checks["inf_count"] = inf_count
        checks["inf_columns"] = features_to_check.index[inf_mask].tolist() if inf_count else []

        # Check for extreme values (> 100 std from mean)
        numeric_features = features_to_check.dropna()
        if len(numeric_features) > 0:
            z_scores = np.abs(
                (numeric_features - numeric_features.mean())
                / (numeric_features.std() + 1e-8)
            )
            extreme_mask = z_scores > 100
            extreme_count = int(extreme_mask.sum())
            checks["has_extreme"] = extreme_count > 0
            checks["extreme_count"] = extreme_count
            checks["extreme_columns"] = z_scores.index[extreme_mask].tolist() if extreme_count else []
        else:
            checks["has_extreme"] = False
            checks["extreme_count"] = 0
            checks["extreme_columns"] = []

        checks["healthy"] = not (
            checks["has_nan"] or checks["has_inf"] or checks["has_extreme"]
        )

        if not checks["healthy"]:
            logger.warning(f"Feature quality issues: {checks}")

        return checks
