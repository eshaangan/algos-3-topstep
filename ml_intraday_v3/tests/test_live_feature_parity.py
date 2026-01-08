import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from ml_intraday_v3.features.build import build_features
from ml_intraday_v3.live_trading.feature_generator import LiveFeatureGenerator


def _make_synthetic_bars(n: int = 120):
    idx = pd.date_range(
        start="2024-01-02 08:30",
        periods=n,
        freq="5min",
        tz="America/Chicago",
    )
    base = np.linspace(100, 110, n)
    df = pd.DataFrame(
        {
            "open": base,
            "high": base + 0.5,
            "low": base - 0.5,
            "close": base + 0.1,
            "volume": np.linspace(1000, 2000, n),
        },
        index=idx,
    )
    return df


def test_live_features_match_offline_no_nans():
    bars_chi = _make_synthetic_bars()

    # Offline features
    with open(Path("ml_intraday_v3/configs/features.yaml"), "r") as f:
        feat_cfg = yaml.safe_load(f)
    offline = build_features(
        bars_df=bars_chi.tz_convert("UTC"),
        bar_size="5m",
        config=feat_cfg,
    )
    feature_cols = list(offline.columns)

    # Live features via adapter
    gen = LiveFeatureGenerator(
        feature_columns=feature_cols,
        bar_size="5m",
        features_config_path=Path("ml_intraday_v3/configs/features.yaml"),
    )
    live_series = gen.generate_features(bars_chi)

    assert live_series.isna().sum() == 0, "Live features contain NaNs"

    # Compare a subset of key columns for parity
    subset = [
        "log_return_1",
        "log_return_2",
        "log_return_4",
        "true_range",
        "atr_14",
        "vol_20",
    ]
    for col in subset:
        np.testing.assert_allclose(
            live_series[col],
            offline.iloc[-1][col],
            rtol=1e-8,
            atol=1e-8,
            err_msg=f"Mismatch in {col}",
        )


def test_check_feature_quality_lists_columns():
    gen = LiveFeatureGenerator(
        feature_columns=["a", "b"],
        bar_size="5m",
        features_config_path=Path("ml_intraday_v3/configs/features.yaml"),
    )
    series = pd.Series({"a": np.nan, "b": np.inf})
    checks = gen.check_feature_quality(series)
    assert checks["has_nan"] and checks["nan_columns"] == ["a"]
    assert checks["has_inf"] and checks["inf_columns"] == ["b"]

