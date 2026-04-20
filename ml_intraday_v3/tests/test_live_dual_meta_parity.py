from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_intraday_v3.live_trading.decision_gate import (
    compute_live_regime_context,
    evaluate_live_trade_decision,
)
from ml_intraday_v3.live_trading.model_predictor import LiveModelPredictor


class SideAwarePrimaryModel:
    def predict_proba(self, X):
        if hasattr(X, "iloc"):
            side = float(X.iloc[0, 1])
        else:
            side = float(X[0, 1])
        prob = 0.70 if side > 0 else 0.20
        return np.array([[1.0 - prob, prob]])


class FixedMetaModel:
    def __init__(self, prob: float):
        self.prob = float(prob)

    def predict_proba(self, X):
        return np.array([[1.0 - self.prob, self.prob]])


def _standard_state(n_features: int) -> dict:
    return {
        "impute": "median",
        "scaler": "standard",
        "medians": [0.0] * n_features,
        "means": [0.0] * n_features,
        "stds": [1.0] * n_features,
    }


def _make_bars(n: int = 160) -> pd.DataFrame:
    idx = pd.date_range(
        start="2026-01-05 09:30",
        periods=n,
        freq="5min",
        tz="UTC",
    )
    base = np.linspace(100.0, 101.5, n)
    wave = np.sin(np.linspace(0, 6, n)) * 0.15
    close = base + wave
    return pd.DataFrame(
        {
            "open": close - 0.05,
            "high": close + 0.20,
            "low": close - 0.20,
            "close": close,
            "volume": np.linspace(1000, 2000, n),
        },
        index=idx,
    )


def test_predictor_uses_matching_routed_meta_model(tmp_path: Path):
    bundle_path = tmp_path / "bundle.pkl"
    bundle = {
        "primary_model": SideAwarePrimaryModel(),
        "primary_preprocessor": _standard_state(2),
        "primary_feature_columns": ["feature_a", "side"],
        "thresholds": {"primary_threshold": 0.38, "meta_threshold": 0.45},
        "meta_model": None,
        "meta_preprocessor": None,
        "meta_feature_columns": None,
        "meta_routes": [
            {
                "name": "long_route",
                "side": "long",
                "regimes": [4],
                "threshold_primary": 0.60,
                "threshold_meta": 0.45,
                "model": FixedMetaModel(0.90),
                "preprocessor": _standard_state(4),
                "feature_columns": ["feature_a", "side", "p_primary", "p_primary_logit"],
            },
            {
                "name": "short_route",
                "side": "short",
                "regimes": [4],
                "threshold_primary": 0.60,
                "threshold_meta": 0.45,
                "model": FixedMetaModel(0.05),
                "preprocessor": _standard_state(4),
                "feature_columns": ["feature_a", "side", "p_primary", "p_primary_logit"],
            },
        ],
    }
    joblib.dump(bundle, bundle_path)

    predictor = LiveModelPredictor(bundle_path)
    prediction = predictor.predict(
        pd.Series({"feature_a": 1.0, "side": 0.0}),
        use_meta=True,
        combined_regime=4,
    )

    assert prediction["side"] == 1
    assert prediction["meta_route"] == "long_route"
    assert prediction["meta_prob"] == 0.90
    assert prediction["meta_threshold"] == 0.45


def test_evaluate_live_trade_decision_applies_threshold_schedule():
    bars = _make_bars()
    timestamp = bars.index[-1]
    base_config = {
        "decision": {
            "use_meta": False,
            "primary_threshold": 0.38,
            "primary_threshold_by_side": {"long": 0.38, "short": 0.48},
            "regime_filter": {
                "enabled": True,
                "vol_window": 20,
                "trend_window": 50,
                "threshold_schedule": [],
                "overlay_rules": [],
            },
        }
    }
    regime_context = compute_live_regime_context(
        timestamp=timestamp,
        bars_df=bars,
        decision_config=base_config,
    )
    combined_regime = int(regime_context["combined_regime"])
    config = {
        "decision": {
            "use_meta": False,
            "primary_threshold": 0.38,
            "primary_threshold_by_side": {"long": 0.38, "short": 0.48},
            "regime_filter": {
                "enabled": True,
                "vol_window": 20,
                "trend_window": 50,
                "threshold_schedule": [
                    {
                        "name": "tighten_here",
                        "regimes": [combined_regime],
                        "side": "long",
                        "threshold": 0.50,
                        "reason": "tighten_here",
                    }
                ],
                "overlay_rules": [],
            },
        }
    }

    decision = evaluate_live_trade_decision(
        timestamp=timestamp,
        prediction={"side": 1, "y_prob": 0.45, "score_ev": 0.45},
        bars_df=bars,
        decision_config=config,
    )

    assert decision["accept"] is False
    assert decision["decision_reason"] == "tighten_here"


def test_evaluate_live_trade_decision_uses_route_specific_meta_threshold():
    bars = _make_bars(80)
    decision = evaluate_live_trade_decision(
        timestamp=bars.index[-1],
        prediction={
            "side": 1,
            "y_prob": 0.50,
            "score_ev": 0.50,
            "meta_prob": 0.42,
            "meta_threshold": 0.45,
            "meta_route": "long_route",
        },
        bars_df=bars,
        decision_config={
            "decision": {
                "use_meta": True,
                "primary_threshold": 0.38,
                "primary_threshold_by_side": {"long": 0.38, "short": 0.48},
                "meta_threshold": 0.40,
                "require_meta_for_trade": False,
                "regime_filter": {"enabled": False},
            }
        },
    )

    assert decision["accept"] is False
    assert decision["decision_reason"] == "threshold_meta"
