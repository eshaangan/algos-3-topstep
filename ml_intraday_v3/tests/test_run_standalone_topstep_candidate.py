import pandas as pd
import pytest

import ml_intraday_v3.experiments.run_standalone_topstep_candidate as runner


def test_fit_candidate_model_trains_side_models_when_enabled(monkeypatch):
    calls = []

    def fake_fit_binary_candidate_model(train_df, training_cfg):
        calls.append(train_df.copy())
        return {
            "model": object(),
            "preprocessor": object(),
            "feature_columns": ["side", "feature_a"],
            "positive_label": 1,
        }

    monkeypatch.setattr(runner, "_fit_binary_candidate_model", fake_fit_binary_candidate_model)

    train_df = pd.DataFrame(
        {
            "event_id": range(8),
            "t0": pd.date_range("2025-01-01", periods=8, freq="5min", tz="UTC"),
            "side": [1, 1, 1, 1, -1, -1, -1, -1],
            "y": [1, -1, 1, -1, 1, -1, 1, -1],
            "w_final": [1.0] * 8,
            "feature_a": [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4],
        }
    )
    training_cfg = {
        "target": {"positive_label": 1},
        "side_specialization": {
            "enabled": True,
            "min_side_train_samples": 2,
            "min_side_positive_samples": 1,
            "min_side_negative_samples": 1,
            "fallback_to_global": True,
        },
    }

    candidate = runner._fit_candidate_model(train_df, training_cfg)

    assert candidate["mode"] == "side_specialized"
    assert candidate["trained_sides"] == [-1, 1]
    assert set(candidate["side_models"].keys()) == {-1, 1}
    assert candidate["global_model"] is not None
    assert len(calls) == 3
    assert set(calls[1]["side"].unique()) == {1}
    assert set(calls[2]["side"].unique()) == {-1}


def test_score_candidate_side_specialized_uses_matching_side_models(monkeypatch):
    def fake_predict_binary_candidate_proba(test_df, candidate):
        marker = candidate["marker"]
        if marker == "long":
            return pd.Series([0.8] * len(test_df), dtype=float).to_numpy()
        if marker == "short":
            return pd.Series([0.3] * len(test_df), dtype=float).to_numpy()
        return pd.Series([0.5] * len(test_df), dtype=float).to_numpy()

    monkeypatch.setattr(runner, "_predict_binary_candidate_proba", fake_predict_binary_candidate_proba)

    test_df = pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13],
            "side": [1, -1, 1, -1],
            "y": [1, -1, 1, -1],
            "w_final": [1.0, 1.0, 1.0, 1.0],
        }
    )
    candidate = {
        "mode": "side_specialized",
        "positive_label": 1,
        "side_models": {
            1: {"marker": "long"},
            -1: {"marker": "short"},
        },
        "global_model": {"marker": "global"},
    }
    training_cfg = {"eval": {"threshold": 0.5}}

    classification, primary_preds = runner._score_candidate(test_df, candidate, training_cfg)

    assert primary_preds["y_prob"].tolist() == [0.8, 0.3, 0.8, 0.3]
    assert primary_preds["model_source"].tolist() == [
        "side_model",
        "side_model",
        "side_model",
        "side_model",
    ]
    assert classification["test_accuracy"] == 1.0
    assert classification["positive_prediction_rate"] == 0.5


def test_fit_candidate_model_trains_regime_route_models_when_enabled(monkeypatch):
    calls = []

    def fake_fit_binary_candidate_model(train_df, training_cfg):
        calls.append(train_df.copy())
        return {
            "model": object(),
            "preprocessor": object(),
            "feature_columns": ["side", "feature_a"],
            "positive_label": 1,
        }

    monkeypatch.setattr(runner, "_fit_binary_candidate_model", fake_fit_binary_candidate_model)

    train_df = pd.DataFrame(
        {
            "event_id": range(10),
            "t0": pd.date_range("2025-01-01", periods=10, freq="5min", tz="UTC"),
            "side": [1, 1, -1, -1, -1, -1, 1, -1, 1, -1],
            "combined_regime": [1, 2, 1, 1, 5, 5, 4, 3, 2, 5],
            "y": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            "w_final": [1.0] * 10,
            "feature_a": [0.1] * 10,
        }
    )
    training_cfg = {
        "target": {"positive_label": 1},
        "regime_routing": {
            "enabled": True,
            "routes": [
                {
                    "name": "jan_short_regimes",
                    "side": "short",
                    "regimes": [1, 5],
                    "min_train_samples": 2,
                    "min_positive_samples": 1,
                    "min_negative_samples": 1,
                }
            ],
        },
    }

    candidate = runner._fit_candidate_model(train_df, training_cfg)

    assert candidate["mode"] == "regime_routed"
    assert len(candidate["route_models"]) == 1
    assert candidate["route_models"][0]["name"] == "jan_short_regimes"
    assert candidate["route_models"][0]["regimes"] == [1, 5]
    assert candidate["route_models"][0]["side"] == "short"
    assert len(calls) == 2
    assert set(calls[1]["side"].unique()) == {-1}
    assert set(calls[1]["combined_regime"].unique()) == {1, 5}


def test_score_candidate_regime_routed_uses_matching_route_models(monkeypatch):
    def fake_predict_binary_candidate_proba(test_df, candidate):
        marker = candidate["marker"]
        if marker == "route":
            return pd.Series([0.2] * len(test_df), dtype=float).to_numpy()
        return pd.Series([0.8] * len(test_df), dtype=float).to_numpy()

    monkeypatch.setattr(runner, "_predict_binary_candidate_proba", fake_predict_binary_candidate_proba)

    test_df = pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13],
            "side": [1, -1, -1, -1],
            "combined_regime": [1, 1, 5, 4],
            "y": [1, -1, -1, -1],
            "w_final": [1.0, 1.0, 1.0, 1.0],
        }
    )
    candidate = {
        "mode": "regime_routed",
        "positive_label": 1,
        "global_model": {"marker": "global"},
        "route_models": [
            {
                "name": "jan_short_regimes",
                "regimes": [1, 5],
                "side": "short",
                "candidate": {"marker": "route"},
            }
        ],
    }
    training_cfg = {"eval": {"threshold": 0.5}}

    classification, primary_preds = runner._score_candidate(test_df, candidate, training_cfg)

    assert primary_preds["y_prob"].tolist() == [0.8, 0.2, 0.2, 0.8]
    assert primary_preds["model_source"].tolist() == [
        "global_model",
        "route:jan_short_regimes",
        "route:jan_short_regimes",
        "global_model",
    ]
    assert classification["positive_prediction_rate"] == 0.5


def test_build_targeted_meta_dataset_filters_route_and_proposed_trades():
    primary_preds = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "y_prob": [0.55, 0.65, 0.45, 0.80],
        }
    )
    base_df = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "t0": pd.date_range("2025-01-01", periods=4, freq="5min", tz="UTC"),
            "side": [1, -1, -1, -1],
            "combined_regime": [6, 6, 6, 5],
            "y": [1, 1, -1, -1],
            "w_final": [1.0, 1.0, 1.0, 1.0],
            "feature_a": [0.1, 0.2, 0.3, 0.4],
        }
    )
    events_df = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "y": [1, 1, -1, -1],
            "ret_net": [0.5, 1.0, -0.5, -1.0],
        }
    )
    training_cfg = {
        "eval": {"threshold": 0.5},
        "meta": {
            "enabled": True,
            "threshold_primary": 0.5,
            "target": {"kind": "ret_net_positive"},
            "features": {
                "include_original_features": False,
                "include_primary_prob": True,
                "include_primary_logit": False,
            },
            "route": {
                "name": "jan_short_veto",
                "side": "short",
                "regimes": [6],
            },
        },
    }

    meta_df, feature_cols = runner._build_targeted_meta_dataset(
        primary_preds,
        base_df,
        events_df,
        training_cfg,
    )

    assert meta_df["event_id"].tolist() == [2]
    assert meta_df["y"].tolist() == [1]
    assert feature_cols == ["p_primary"]
    assert "feature_a" not in meta_df.columns


def test_train_targeted_meta_model_returns_subset_predictions():
    train_df = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5, 6],
            "t0": pd.date_range("2025-01-01", periods=6, freq="5min", tz="UTC"),
            "side": [-1, -1, -1, -1, -1, 1],
            "combined_regime": [6, 6, 6, 6, 6, 6],
            "y": [1, -1, 1, -1, 1, -1],
            "w_final": [1.0] * 6,
            "feature_a": [2.0, -2.0, 1.5, -1.5, 1.0, 0.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "event_id": [10, 11, 12],
            "t0": pd.date_range("2025-01-02", periods=3, freq="5min", tz="UTC"),
            "side": [-1, 1, -1],
            "combined_regime": [6, 6, 5],
            "y": [1, -1, -1],
            "w_final": [1.0, 1.0, 1.0],
            "feature_a": [1.8, 0.2, -1.8],
        }
    )
    train_events = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5, 6],
            "y": [1, -1, 1, -1, 1, -1],
            "ret_net": [1.0, -1.0, 0.8, -0.8, 0.5, -0.2],
        }
    )
    test_events = pd.DataFrame(
        {
            "event_id": [10, 11, 12],
            "y": [1, -1, -1],
            "ret_net": [0.7, -0.1, -0.6],
        }
    )
    primary_train_preds = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5, 6],
            "y_prob": [0.80, 0.75, 0.70, 0.68, 0.72, 0.65],
        }
    )
    primary_test_preds = pd.DataFrame(
        {
            "event_id": [10, 11, 12],
            "y_prob": [0.78, 0.62, 0.81],
        }
    )
    training_cfg = {
        "seed": 42,
        "sample_weight": {"enabled": False},
        "preprocessing": {"scaler": "standard", "impute": "median"},
        "meta": {
            "enabled": True,
            "threshold_primary": 0.5,
            "threshold_meta": 0.5,
            "target": {"kind": "ret_net_positive"},
            "features": {
                "include_original_features": True,
                "include_primary_prob": True,
                "include_primary_logit": True,
            },
            "route": {
                "name": "jan_short_veto",
                "side": "short",
                "regimes": [6],
                "min_train_samples": 2,
                "min_positive_samples": 1,
                "min_negative_samples": 1,
            },
            "model": {
                "params": {
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 200,
                }
            },
        },
    }

    result = runner._train_targeted_meta_model(
        train_df=train_df,
        test_df=test_df,
        train_events=train_events,
        test_events=test_events,
        primary_train_preds=primary_train_preds,
        primary_test_preds=primary_test_preds,
        training_cfg=training_cfg,
    )

    assert result["enabled"] is True
    assert result["skipped"] is False
    assert result["meta_preds"]["event_id"].tolist() == [10]
    assert result["meta_preds"]["meta_source"].tolist() == ["jan_short_veto"]
    assert len(result["p_meta"]) == 1


def test_train_targeted_meta_model_combines_multiple_routes():
    train_df = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "t0": pd.date_range("2025-01-01", periods=8, freq="5min", tz="UTC"),
            "side": [-1, -1, -1, -1, 1, 1, 1, 1],
            "combined_regime": [6, 6, 6, 6, 0, 0, 0, 0],
            "y": [1, -1, 1, -1, 1, -1, 1, -1],
            "w_final": [1.0] * 8,
            "feature_a": [2.0, -2.0, 1.5, -1.5, 2.2, -2.2, 1.8, -1.8],
        }
    )
    test_df = pd.DataFrame(
        {
            "event_id": [10, 11, 12],
            "t0": pd.date_range("2025-01-02", periods=3, freq="5min", tz="UTC"),
            "side": [-1, 1, 1],
            "combined_regime": [6, 0, 5],
            "y": [1, -1, 1],
            "w_final": [1.0, 1.0, 1.0],
            "feature_a": [1.9, -1.9, 0.1],
        }
    )
    train_events = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "y": [1, -1, 1, -1, 1, -1, 1, -1],
            "ret_net": [1.0, -1.0, 0.8, -0.8, 1.2, -1.2, 0.6, -0.6],
        }
    )
    test_events = pd.DataFrame(
        {
            "event_id": [10, 11, 12],
            "y": [1, -1, 1],
            "ret_net": [0.7, -0.7, 0.2],
        }
    )
    primary_train_preds = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "y_prob": [0.80, 0.75, 0.70, 0.68, 0.82, 0.77, 0.74, 0.69],
        }
    )
    primary_test_preds = pd.DataFrame(
        {
            "event_id": [10, 11, 12],
            "y_prob": [0.78, 0.79, 0.81],
        }
    )
    training_cfg = {
        "seed": 42,
        "sample_weight": {"enabled": False},
        "preprocessing": {"scaler": "standard", "impute": "median"},
        "meta": {
            "enabled": True,
            "target": {"kind": "ret_net_positive"},
            "features": {
                "include_original_features": True,
                "include_primary_prob": True,
                "include_primary_logit": True,
            },
            "routes": [
                {
                    "name": "jan_short_veto",
                    "side": "short",
                    "regimes": [6],
                    "threshold_primary": 0.5,
                    "threshold_meta": 0.5,
                    "min_train_samples": 2,
                    "min_positive_samples": 1,
                    "min_negative_samples": 1,
                },
                {
                    "name": "mar_long_veto",
                    "side": "long",
                    "regimes": [0],
                    "threshold_primary": 0.5,
                    "threshold_meta": 0.5,
                    "min_train_samples": 2,
                    "min_positive_samples": 1,
                    "min_negative_samples": 1,
                },
            ],
            "model": {
                "params": {
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 200,
                }
            },
        },
    }

    result = runner._train_targeted_meta_model(
        train_df=train_df,
        test_df=test_df,
        train_events=train_events,
        test_events=test_events,
        primary_train_preds=primary_train_preds,
        primary_test_preds=primary_test_preds,
        training_cfg=training_cfg,
    )

    assert result["enabled"] is True
    assert result["skipped"] is False
    assert result["meta_preds"]["event_id"].tolist() == [10, 11]
    assert result["meta_preds"]["meta_source"].tolist() == ["jan_short_veto", "mar_long_veto"]
    assert len(result["routes"]) == 2
    assert all(not route.get("skipped", False) for route in result["routes"])
    assert result["metrics"]["mode"] == "multi_route"


def test_build_primary_score_diagnostics_reports_month_side_and_regime_groups():
    test_df = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "t0": pd.to_datetime(
                [
                    "2026-01-02 15:35:00+00:00",
                    "2026-01-03 15:40:00+00:00",
                    "2026-01-04 15:45:00+00:00",
                    "2026-01-05 15:50:00+00:00",
                    "2026-01-06 15:55:00+00:00",
                    "2026-02-02 15:35:00+00:00",
                    "2026-02-03 15:40:00+00:00",
                    "2026-02-04 15:45:00+00:00",
                    "2026-02-05 15:50:00+00:00",
                    "2026-02-06 15:55:00+00:00",
                ],
                utc=True,
            ),
            "side": [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1],
            "combined_regime": [2, 2, 5, 5, 2, 2, 2, 5, 5, 5],
            "regime_label": [
                "low_vol_sideways",
                "low_vol_sideways",
                "med_vol_uptrend",
                "med_vol_uptrend",
                "low_vol_sideways",
                "low_vol_sideways",
                "low_vol_sideways",
                "med_vol_uptrend",
                "med_vol_uptrend",
                "med_vol_uptrend",
            ],
            "y": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            "w_final": [1.0] * 10,
        }
    )
    test_events = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "ret_net": [1.0, -0.8, 0.7, -0.4, 0.9, -1.1, -0.6, 0.6, -0.5, 0.4],
        }
    )
    primary_preds = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y_prob": [0.80, 0.25, 0.75, 0.30, 0.78, 0.20, 0.18, 0.72, 0.35, 0.68],
            "model_source": ["single_model"] * 10,
        }
    )

    diagnostics = runner._build_primary_score_diagnostics(
        test_df=test_df,
        test_events=test_events,
        primary_preds=primary_preds,
        positive_label=1,
    )

    assert diagnostics["overall"]["rows"] == 10
    assert diagnostics["overall"]["rank_ic_ret"] is not None
    assert diagnostics["overall"]["ece"] is not None
    assert {row["month"] for row in diagnostics["by_month"]} == {"2026-01", "2026-02"}
    assert {row["side_label"] for row in diagnostics["by_side"]} == {"long", "short"}
    assert any(row["regime_label"] == "low_vol_sideways" for row in diagnostics["by_regime"])
    assert diagnostics["problem_groups"]


def test_build_risk_tearsheet_computes_daily_and_streak_metrics():
    trades_df = pd.DataFrame(
        {
            "executed": [True, True, True, True, True],
            "pnl_usd": [100.0, -50.0, -25.0, 200.0, -75.0],
            "exit_ts": pd.to_datetime(
                [
                    "2026-01-02 16:00:00+00:00",
                    "2026-01-02 18:00:00+00:00",
                    "2026-01-03 16:00:00+00:00",
                    "2026-01-04 16:00:00+00:00",
                    "2026-01-04 18:00:00+00:00",
                ],
                utc=True,
            ),
        }
    )
    equity_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-02 16:00:00+00:00",
                    "2026-01-02 18:00:00+00:00",
                    "2026-01-03 16:00:00+00:00",
                    "2026-01-04 16:00:00+00:00",
                    "2026-01-04 18:00:00+00:00",
                ],
                utc=True,
            ),
            "equity": [50100.0, 50050.0, 50025.0, 50225.0, 50150.0],
            "pnl_usd": [100.0, -50.0, -25.0, 200.0, -75.0],
        }
    )

    risk = runner._build_risk_tearsheet(trades_df, equity_df)

    assert risk["executed_trades"] == 5
    assert risk["streaks"]["longest_loss_streak"] == 2
    assert risk["daily"]["days"] == 3
    assert risk["daily"]["worst_day_usd"] == -25.0
    assert risk["tail"]["worst_trade_usd"] == -75.0
    assert risk["equity"]["max_drawdown_usd"] == 75.0


def test_build_meta_score_diagnostics_coalesces_side_and_regime_context():
    meta_result = {
        "enabled": True,
        "skipped": False,
        "meta_test_df": pd.DataFrame(
            {
                "event_id": [10, 11, 12, 13],
                "y": [1, 0, 1, 0],
                "side": [-1, -1, -1, -1],
                "combined_regime": [2, 2, 2, 2],
            }
        ),
        "meta_preds": pd.DataFrame(
            {
                "event_id": [10, 11, 12, 13],
                "p_meta": [0.70, 0.20, 0.65, 0.30],
                "meta_source": ["route"] * 4,
            }
        ),
    }
    test_df = pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13],
            "t0": pd.to_datetime(
                [
                    "2026-01-02 15:35:00+00:00",
                    "2026-01-03 15:40:00+00:00",
                    "2026-01-04 15:45:00+00:00",
                    "2026-01-05 15:50:00+00:00",
                ],
                utc=True,
            ),
            "side": [-1, -1, -1, -1],
            "combined_regime": [2, 2, 2, 2],
            "regime_label": ["low_vol_sideways"] * 4,
        }
    )
    test_events = pd.DataFrame(
        {
            "event_id": [10, 11, 12, 13],
            "ret_net": [0.8, -0.7, 0.6, -0.5],
        }
    )

    diagnostics = runner._build_meta_score_diagnostics(
        meta_result=meta_result,
        test_df=test_df,
        test_events=test_events,
    )

    assert diagnostics["enabled"] is True
    assert diagnostics["skipped"] is False
    assert diagnostics["overall"]["rows"] == 4
    assert diagnostics["by_side"]
    assert any(row["regime_label"] == "low_vol_sideways" for row in diagnostics["by_regime"])


def test_apply_target_routes_relabels_matching_short_regimes_from_ret_net():
    model_df = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "side": [-1, -1, 1, -1],
            "combined_regime": [2, 8, 2, 1],
            "y": [1, 1, 1, -1],
            "w_final": [1.0, 1.0, 1.0, 1.0],
            "feature_a": [0.1, 0.2, 0.3, 0.4],
        }
    )
    events_df = pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4],
            "ret_net": [-0.5, 0.7, 0.4, 0.2],
        }
    )
    training_cfg = {
        "target": {"classes": [-1, 1], "positive_label": 1},
        "target_routes": {
            "enabled": True,
            "routes": [
                {
                    "name": "short_uptrend_profit_route",
                    "side": "short",
                    "regimes": [2, 8],
                    "target_kind": "ret_net_positive",
                }
            ],
        },
    }

    relabeled = runner._apply_target_routes(model_df, events_df, training_cfg)

    assert relabeled["y_original"].tolist() == [1, 1, 1, -1]
    assert relabeled["y"].tolist() == [-1, 1, 1, -1]
    assert relabeled["target_route_name"].tolist() == [
        "short_uptrend_profit_route",
        "short_uptrend_profit_route",
        "",
        "",
    ]
    assert "ret_net" not in relabeled.columns


def test_summarize_target_routes_reports_rows_and_regimes():
    model_df = pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "y": [-1, 1, -1],
            "target_route_name": ["short_uptrend_profit_route", "short_uptrend_profit_route", ""],
            "combined_regime": [2, 8, 1],
        }
    )
    events_df = pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "ret_net": [-0.5, 0.7, 0.1],
        }
    )

    summary = runner._summarize_target_routes(model_df, events_df)

    assert summary["enabled"] is True
    assert summary["rows_relabeled"] == 2
    assert summary["routes"][0]["name"] == "short_uptrend_profit_route"
    assert summary["routes"][0]["rows"] == 2
    assert summary["routes"][0]["regimes"] == [2, 8]
    assert summary["routes"][0]["avg_ret_net"] == pytest.approx(0.1)
