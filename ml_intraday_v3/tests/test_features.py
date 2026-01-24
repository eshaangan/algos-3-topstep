"""
Tests for feature engineering module.

Critical tests:
1. Deterministic column ordering
2. Causal computation (no lookahead)
3. Future perturbation (leakage test)
4. Multi-bar size support
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from features import (
    build_features,
    get_feature_registry,
    filter_registry_for_bar_size,
    write_feature_schema,
    compute_schema_hash,
)


@pytest.fixture
def minimal_features_config():
    """Minimal features config for testing."""
    return {
        "computation": {
            "nan_policy": "keep_with_mask",
            "eps": 1e-8,
        },
        "returns": {
            "single_bar": True,
            "lookback_bars": {
                "1m": [3, 6],
                "5m": [2, 4],
            },
            "enable_multi_horizon": False,  # Disable to test bar-size specific features only
        },
        "volatility": {
            "atr_period": 14,
        },
        "trend": {
            "ema_fast_period": 13,
            "ema_slow_period": 34,
        },
        "structure": {
            "enabled": True,
        },
        "time": {
            "enabled": True,
            "minutes_per_session": 1440,
        },
        "output": {
            "column_order": "registry",
            "include_session_features": True,
            "include_synthetic_flag": True,
        },
    }


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame for testing."""
    # Create 100 bars of synthetic data
    np.random.seed(42)
    n = 100

    # Start from a known datetime
    start_time = pd.Timestamp("2025-01-15 09:30:00", tz="UTC")
    timestamps = [start_time + timedelta(minutes=i) for i in range(n)]

    # Generate realistic-looking OHLCV data
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.1)

    data = {
        "open": close_prices + np.random.randn(n) * 0.05,
        "high": close_prices + np.abs(np.random.randn(n)) * 0.1,
        "low": close_prices - np.abs(np.random.randn(n)) * 0.1,
        "close": close_prices,
        "volume": np.random.randint(100, 1000, n),
        "is_synthetic": np.zeros(n, dtype=bool),
        "minute_of_day": [(t.hour * 60 + t.minute) for t in timestamps],
        "day_of_week": [t.dayofweek for t in timestamps],
    }

    # Ensure OHLC validity
    for i in range(n):
        data["high"][i] = max(
            data["high"][i], data["open"][i], data["close"][i]
        )
        data["low"][i] = min(data["low"][i], data["open"][i], data["close"][i])

    df = pd.DataFrame(data, index=timestamps)
    return df


class TestFeatureRegistry:
    """Test feature registry and deterministic ordering."""

    def test_registry_has_expected_features(self, minimal_features_config):
        """Test that registry contains expected features."""
        registry = get_feature_registry(minimal_features_config)

        feature_names = [spec.name for spec in registry]

        # Check key features are present
        assert "log_return_1" in feature_names
        assert "true_range" in feature_names
        assert "atr_14" in feature_names
        assert "ema_13" in feature_names
        assert "ema_34" in feature_names
        assert "ema_spread" in feature_names
        assert "ema_ratio" in feature_names
        assert "candle_body" in feature_names
        assert "candle_range" in feature_names
        assert "minute_of_day_sin" in feature_names
        assert "minute_of_day_cos" in feature_names
        assert "day_of_week" in feature_names
        assert "is_synthetic" in feature_names
        assert "usable_for_training" in feature_names

    def test_registry_bar_size_filtering(self, minimal_features_config):
        """Test that registry filters by bar size correctly."""
        full_registry = get_feature_registry(minimal_features_config)

        # Filter for 1m
        registry_1m = filter_registry_for_bar_size(full_registry, "1m")
        names_1m = [spec.name for spec in registry_1m]

        # Should have 1m-specific returns
        assert "log_return_3" in names_1m  # 1m lookback
        assert "log_return_6" in names_1m  # 1m lookback

        # Filter for 5m
        registry_5m = filter_registry_for_bar_size(full_registry, "5m")
        names_5m = [spec.name for spec in registry_5m]

        # Should have 5m-specific returns
        assert "log_return_2" in names_5m  # 5m lookback
        assert "log_return_4" in names_5m  # 5m lookback

        # Should not have 1m-only returns in 5m
        assert "log_return_3" not in names_5m
        assert "log_return_6" not in names_5m

    def test_feature_columns_deterministic_order(self, minimal_features_config):
        """Test that feature columns are always in same deterministic order."""
        registry_1 = get_feature_registry(minimal_features_config)
        registry_2 = get_feature_registry(minimal_features_config)

        names_1 = [spec.name for spec in registry_1]
        names_2 = [spec.name for spec in registry_2]

        # Order must be identical
        assert names_1 == names_2

        # Verify ordering convention: returns, volatility, trend, structure, time, meta
        # Returns should come first
        assert names_1[0].startswith("log_return")

        # true_range and atr should be in volatility section
        true_range_idx = names_1.index("true_range")
        atr_idx = names_1.index("atr_14")
        assert true_range_idx < atr_idx  # true_range before ATR

        # EMAs should be in trend section
        ema_fast_idx = names_1.index("ema_13")
        ema_slow_idx = names_1.index("ema_34")
        assert atr_idx < ema_fast_idx  # trend after volatility

        # Candle features in structure section
        candle_body_idx = names_1.index("candle_body")
        assert ema_slow_idx < candle_body_idx  # structure after trend

        # Time features
        minute_sin_idx = names_1.index("minute_of_day_sin")
        assert candle_body_idx < minute_sin_idx  # time after structure

        # Meta features last
        is_synthetic_idx = names_1.index("is_synthetic")
        usable_idx = names_1.index("usable_for_training")
        assert minute_sin_idx < is_synthetic_idx  # meta after time
        assert is_synthetic_idx < usable_idx  # usable_for_training last


class TestFeatureBuild:
    """Test feature computation."""

    def test_build_features_basic(self, sample_ohlcv_df, minimal_features_config):
        """Test that build_features runs without error."""
        features_df = build_features(
            bars_df=sample_ohlcv_df,
            bar_size="1m",
            config=minimal_features_config,
        )

        assert len(features_df) == len(sample_ohlcv_df)
        assert features_df.index.equals(sample_ohlcv_df.index)

        # Check some features were computed
        assert "log_return_1" in features_df.columns
        assert "true_range" in features_df.columns
        assert "ema_13" in features_df.columns

    def test_features_have_nans_for_lookback(
        self, sample_ohlcv_df, minimal_features_config
    ):
        """Test that features with lookback have NaNs at start."""
        features_df = build_features(
            bars_df=sample_ohlcv_df,
            bar_size="1m",
            config=minimal_features_config,
        )

        # log_return_1 should have 1 NaN at start
        assert features_df["log_return_1"].isna().sum() >= 1
        assert features_df["log_return_1"].iloc[0] is pd.NA or pd.isna(
            features_df["log_return_1"].iloc[0]
        )

        # log_return_6 should have 6 NaNs at start
        assert features_df["log_return_6"].isna().sum() >= 6

        # Note: ATR (EWM) doesn't produce NaNs with adjust=False
        # It starts computing from bar 1 using the first true_range value
        # This is acceptable behavior for exponential moving averages

    def test_usable_for_training_mask(self, sample_ohlcv_df, minimal_features_config):
        """Test that usable_for_training mask is computed correctly."""
        features_df = build_features(
            bars_df=sample_ohlcv_df,
            bar_size="1m",
            config=minimal_features_config,
        )

        assert "usable_for_training" in features_df.columns

        # Early rows should not be usable (due to NaN features)
        assert not features_df["usable_for_training"].iloc[0]

        # Later rows (after warmup) should be usable
        # ATR has 14-bar warmup, so after ~40 bars we should have usable rows
        assert features_df["usable_for_training"].iloc[50:].any()

    def test_features_causal_no_lookahead_on_known_series(self):
        """
        Test features are causal on a known series.

        Create a step function in prices at time T.
        Features at time T should not reflect the step.
        Features at time T+1 should reflect the step.
        """
        # Create simple price series with step
        n = 50
        timestamps = pd.date_range("2025-01-15 09:30", periods=n, freq="1min", tz="UTC")

        prices = np.ones(n) * 100.0
        # Step up at index 25
        prices[25:] = 110.0

        df = pd.DataFrame(
            {
                "open": prices,
                "high": prices + 0.1,
                "low": prices - 0.1,
                "close": prices,
                "volume": [100] * n,
                "is_synthetic": [False] * n,
            },
            index=timestamps,
        )

        config = {
            "computation": {"nan_policy": "keep_with_mask", "eps": 1e-8},
            "returns": {"single_bar": True, "lookback_bars": {"1m": [3], "5m": []}},
            "volatility": {"atr_period": 5},
            "trend": {"ema_fast_period": 5, "ema_slow_period": 10},
            "structure": {"enabled": True},
            "time": {"enabled": False},
            "output": {
                "column_order": "registry",
                "include_synthetic_flag": False,
            },
        }

        features_df = build_features(df, "1m", config)

        # At time 25 (step time), log_return_1 should still be ~0
        # (because it uses close[25] / close[24])
        # At time 26, log_return_1 should be positive (110/110 vs previous 100)

        # Actually at time 25, log_return_1 = log(close[25] / close[24]) = log(110/100)
        # So it WILL show the step at time 25, which is correct - it's using data <= 25

        # The key test: features at time 24 should not change if we modify time 25+

        # Get features at time 24
        feat_at_24_before = features_df.loc[timestamps[24]].copy()

        # Now modify prices after time 24
        df_modified = df.copy()
        df_modified.loc[timestamps[25]:, "close"] = 150.0  # Different step

        features_df_modified = build_features(df_modified, "1m", config)

        # Features at time 24 should be identical
        feat_at_24_after = features_df_modified.loc[timestamps[24]]

        # Compare (allowing for floating point tolerance)
        for col in feat_at_24_before.index:
            if pd.isna(feat_at_24_before[col]) and pd.isna(feat_at_24_after[col]):
                continue  # Both NaN, OK
            elif pd.isna(feat_at_24_before[col]) or pd.isna(feat_at_24_after[col]):
                pytest.fail(
                    f"NaN mismatch at time 24 for {col}: before={feat_at_24_before[col]}, after={feat_at_24_after[col]}"
                )
            else:
                assert np.isclose(
                    feat_at_24_before[col], feat_at_24_after[col], atol=1e-10
                ), f"Feature {col} changed at time 24 when modifying future prices!"


class TestFuturePerturbation:
    """Critical leakage test: future perturbation must not change past features."""

    def test_future_perturbation_does_not_change_past_features(
        self, sample_ohlcv_df, minimal_features_config
    ):
        """
        LEAKAGE TEST: Modifying future prices must NOT change past features.

        This is the critical test for causal feature computation.

        Procedure:
        1. Compute features on original data
        2. Select a cutoff timestamp
        3. Modify OHLC AFTER cutoff
        4. Recompute features
        5. Assert features at/before cutoff are IDENTICAL
        """
        # Use original data
        df_original = sample_ohlcv_df.copy()

        # Compute features on original
        features_original = build_features(
            bars_df=df_original,
            bar_size="1m",
            config=minimal_features_config,
        )

        # Select cutoff at 50% of data
        cutoff_idx = len(df_original) // 2
        cutoff_timestamp = df_original.index[cutoff_idx]

        # Create modified version: perturb all OHLC AFTER cutoff (EXCLUSIVE)
        # We modify FROM cutoff onwards, so we'll compare STRICTLY BEFORE cutoff
        df_modified = df_original.copy()
        perturbation = 10.0  # Large perturbation
        df_modified.loc[cutoff_timestamp:, "open"] += perturbation
        df_modified.loc[cutoff_timestamp:, "high"] += perturbation
        df_modified.loc[cutoff_timestamp:, "low"] += perturbation
        df_modified.loc[cutoff_timestamp:, "close"] += perturbation

        # Recompute features on modified data
        features_modified = build_features(
            bars_df=df_modified,
            bar_size="1m",
            config=minimal_features_config,
        )

        # Extract features STRICTLY BEFORE cutoff from both runs
        # Use iloc to get rows before cutoff_idx (exclusive of cutoff)
        features_before_original = features_original.iloc[:cutoff_idx]
        features_before_modified = features_modified.iloc[:cutoff_idx]

        # Compare each feature column
        for col in features_before_original.columns:
            # Get values
            vals_original = features_before_original[col]
            vals_modified = features_before_modified[col]

            # Compare (handling NaNs)
            for idx in vals_original.index:
                orig_val = vals_original.loc[idx]
                mod_val = vals_modified.loc[idx]

                if pd.isna(orig_val) and pd.isna(mod_val):
                    continue  # Both NaN, OK

                if pd.isna(orig_val) or pd.isna(mod_val):
                    pytest.fail(
                        f"NaN mismatch for feature '{col}' at {idx}: "
                        f"original={orig_val}, modified={mod_val}"
                    )

                # For non-NaN values, must be identical
                assert np.isclose(orig_val, mod_val, atol=1e-10), (
                    f"LEAKAGE DETECTED! Feature '{col}' changed at {idx} "
                    f"(before cutoff {cutoff_timestamp}) when future prices were perturbed. "
                    f"Original: {orig_val}, Modified: {mod_val}"
                )

        print(
            f"✓ Future perturbation test PASSED: {len(features_before_original.columns)} features "
            f"at {len(features_before_original)} timestamps before cutoff are identical."
        )


class TestFeatureSchema:
    """Test feature schema artifacts and hashing."""

    def test_schema_hash_deterministic(self, minimal_features_config):
        """Test that schema hash is deterministic."""
        registry = get_feature_registry(minimal_features_config)
        registry_1m = filter_registry_for_bar_size(registry, "1m")

        feature_columns = [spec.name for spec in registry_1m]

        hash1 = compute_schema_hash(feature_columns, registry_1m)
        hash2 = compute_schema_hash(feature_columns, registry_1m)

        assert hash1 == hash2

    def test_schema_hash_changes_with_columns(self, minimal_features_config):
        """Test that schema hash changes if columns change."""
        registry = get_feature_registry(minimal_features_config)
        registry_1m = filter_registry_for_bar_size(registry, "1m")

        feature_columns = [spec.name for spec in registry_1m]

        hash_full = compute_schema_hash(feature_columns, registry_1m)

        # Remove last feature
        feature_columns_subset = feature_columns[:-1]
        registry_subset = registry_1m[:-1]

        hash_subset = compute_schema_hash(feature_columns_subset, registry_subset)

        assert hash_full != hash_subset

    def test_write_feature_schema(self, tmp_path, minimal_features_config):
        """Test writing feature schema JSON."""
        registry = get_feature_registry(minimal_features_config)
        registry_1m = filter_registry_for_bar_size(registry, "1m")

        feature_columns = [spec.name for spec in registry_1m]

        schema_path = tmp_path / "feature_schema.json"

        schema_hash = write_feature_schema(
            output_path=schema_path,
            feature_columns=feature_columns,
            registry=registry_1m,
            bar_size="1m",
            config=minimal_features_config,
            code_version="1.0.0",
            config_hash="test_hash_123",
        )

        # Check file was written
        assert schema_path.exists()

        # Load and verify contents
        with open(schema_path, "r") as f:
            schema = json.load(f)

        assert schema["schema_hash"] == schema_hash
        assert schema["bar_size"] == "1m"
        assert schema["n_features"] == len(feature_columns)
        assert schema["feature_columns"] == feature_columns
        assert len(schema["feature_specs"]) == len(registry_1m)


class TestMultiBarSize:
    """Test multi-bar size support."""

    def test_features_written_for_both_bar_sizes(
        self, tmp_path, sample_ohlcv_df, minimal_features_config
    ):
        """
        Test that features can be built for both 1m and 5m.

        This simulates the CLI build-features command.
        """
        # Create fake run directory structure
        run_dir = tmp_path / "test_run"
        run_dir.mkdir()

        # Create bar_size directories
        bar_dir_1m = run_dir / "bar_size=1m"
        bar_dir_5m = run_dir / "bar_size=5m"
        bar_dir_1m.mkdir()
        bar_dir_5m.mkdir()

        # Write sample bars to both
        sample_ohlcv_df.to_parquet(bar_dir_1m / "bars.parquet")
        sample_ohlcv_df.to_parquet(bar_dir_5m / "bars.parquet")

        # Build features for both bar sizes
        for bar_size in ["1m", "5m"]:
            bar_dir = run_dir / f"bar_size={bar_size}"
            bars_path = bar_dir / "bars.parquet"

            # Load bars
            bars_df = pd.read_parquet(bars_path)

            # Build features
            features_df = build_features(
                bars_df=bars_df,
                bar_size=bar_size,
                config=minimal_features_config,
            )

            # Write features
            features_path = bar_dir / "features.parquet"
            features_df.to_parquet(features_path)

            # Write schema
            registry = get_feature_registry(minimal_features_config)
            bar_registry = filter_registry_for_bar_size(registry, bar_size)
            feature_columns = list(features_df.columns)

            schema_path = bar_dir / "feature_schema.json"
            write_feature_schema(
                output_path=schema_path,
                feature_columns=feature_columns,
                registry=bar_registry,
                bar_size=bar_size,
                config=minimal_features_config,
            )

        # Verify artifacts exist for both bar sizes
        assert (bar_dir_1m / "features.parquet").exists()
        assert (bar_dir_1m / "feature_schema.json").exists()
        assert (bar_dir_5m / "features.parquet").exists()
        assert (bar_dir_5m / "feature_schema.json").exists()

        # Load and verify schemas are different (due to bar-size specific features)
        with open(bar_dir_1m / "feature_schema.json", "r") as f:
            schema_1m = json.load(f)

        with open(bar_dir_5m / "feature_schema.json", "r") as f:
            schema_5m = json.load(f)

        # Schemas should have different feature columns (due to bar-size specific returns)
        assert schema_1m["feature_columns"] != schema_5m["feature_columns"]

        # 1m should have log_return_3, log_return_6
        assert "log_return_3" in schema_1m["feature_columns"]
        assert "log_return_6" in schema_1m["feature_columns"]

        # 5m should have log_return_2, log_return_4
        assert "log_return_2" in schema_5m["feature_columns"]
        assert "log_return_4" in schema_5m["feature_columns"]

        print(
            f"✓ Multi-bar test PASSED: features and schemas written for both 1m and 5m"
        )
