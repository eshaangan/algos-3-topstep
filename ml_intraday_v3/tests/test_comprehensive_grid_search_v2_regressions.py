from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_intraday_v3.experiments.comprehensive_grid_search_v2 import (
    create_sample_weights,
    fit_with_sample_weights,
    run_single_fold,
)


def _make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=500)),
        ]
    )


def test_create_sample_weights_class_balanced_requires_labels():
    events = pd.DataFrame(
        {
            "event_id": [0, 1, 2, 3],
            "t0": pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC"),
            "t1": pd.date_range("2025-01-01 00:05", periods=4, freq="5min", tz="UTC"),
        }
    )

    try:
        create_sample_weights(events, method="class_balanced")
    except ValueError as exc:
        assert "requires labeled events_df" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing y in class_balanced weights")


def test_fit_with_sample_weights_supports_pipeline_and_calibrated_pipeline():
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(size=(120, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(([0, 1] * 60), dtype=int)
    w = np.linspace(0.5, 1.5, len(y))

    model = _make_pipeline()
    fit_with_sample_weights(model, X, y, w)
    assert hasattr(model.named_steps["model"], "coef_")

    calibrated = CalibratedClassifierCV(_make_pipeline(), method="sigmoid", cv=3)
    fit_with_sample_weights(calibrated, X, y, w)
    proba = calibrated.predict_proba(X[:5])
    assert proba.shape == (5, 2)


def test_run_single_fold_class_balanced_with_logreg_pipeline():
    n = 80
    idx = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    event_ids = np.arange(n, dtype=int)

    all_df = pd.DataFrame(index=idx)
    events_df = pd.DataFrame(
        {
            "event_id": event_ids,
            "t0": idx,
            "t1": idx + pd.Timedelta(minutes=5),
            "side": np.where(event_ids % 2 == 0, 1, -1),
        }
    )
    features_df = pd.DataFrame(
        {
            "f1": np.sin(np.linspace(0, 6.0, n)),
            "f2": np.cos(np.linspace(0, 4.0, n)),
        },
        index=idx,
    )
    labels_df = pd.DataFrame(
        {
            "event_id": event_ids,
            "y": np.where(event_ids % 2 == 0, 1, -1),
        }
    ).set_index("event_id")

    fold = {
        "train_event_ids": event_ids[:64].tolist(),
        "test_event_ids": event_ids[64:].tolist(),
    }
    exp_config = {
        "model_kind": "logreg",
        "model_params": {"C": 1.0},
        "sample_weight": "class_balanced",
        "balance_method": "none",
    }

    result = run_single_fold(
        all_df=all_df,
        events_df=events_df,
        features_df=features_df,
        labels_df=labels_df,
        exp_config=exp_config,
        fold=fold,
        fold_num=1,
    )

    assert "error" not in result
    assert 0.0 <= result["test_auc"] <= 1.0
