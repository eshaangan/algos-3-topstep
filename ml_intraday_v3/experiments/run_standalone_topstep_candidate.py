#!/usr/bin/env python3
"""
Train and evaluate a standalone ML candidate against Topstep-style promotion gates.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_intraday_v3.analysis.standalone_promotion import (  # noqa: E402
    evaluate_promotion_gates,
    summarize_trade_directions,
    summarize_trades_df,
)
from ml_intraday_v3.backtesting_v3 import run_backtest  # noqa: E402
from ml_intraday_v3.experiments.risk_scaling import scale_risk_config_for_contracts  # noqa: E402
from ml_intraday_v3.backtesting_v3.decisions import compute_regime_at_events  # noqa: E402
from ml_intraday_v3.core.instrument import load_instrument_from_execution_spec  # noqa: E402
from ml_intraday_v3.features.build import build_features  # noqa: E402
from ml_intraday_v3.labels.events import balance_events, generate_events  # noqa: E402
from ml_intraday_v3.labels.triple_barrier import apply_triplebarrier  # noqa: E402
from ml_intraday_v3.training.metrics import compute_metrics  # noqa: E402
from ml_intraday_v3.training.preprocess import FoldPreprocessor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = {
    "event_id",
    "t0",
    "y",
    "y_original",
    "w_final",
    "vol_regime",
    "trend_regime",
    "combined_regime",
    "regime_label",
    "target_route_name",
}
META_EXCLUDED_COLUMNS = {"event_id", "t0", "y", "w_final", "regime_label"}


class SigmoidCalibratorWrapper:
    """Expose a `.predict()` interface matching IsotonicRegression."""

    def __init__(self, logistic_model: LogisticRegression):
        self.logistic_model = logistic_model

    def predict(self, raw_proba: np.ndarray) -> np.ndarray:
        return self.logistic_model.predict_proba(raw_proba.reshape(-1, 1))[:, 1]


class CalibratedBinaryModel:
    """Wrap a fitted classifier with a 1D probability calibrator."""

    def __init__(self, base_model: LGBMClassifier, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator
        self.classes_ = base_model.classes_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_proba = self.base_model.predict_proba(X)[:, 1]
        calibrated_p1 = self.calibrator.predict(raw_proba)
        calibrated_p0 = 1.0 - calibrated_p1
        return np.column_stack([calibrated_p0, calibrated_p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


def _to_utc_timestamp(value, *, end_of_day: bool = False) -> pd.Timestamp:
    if isinstance(value, str) and end_of_day and len(value) == 10:
        value = f"{value} 23:59:59"
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _load_single_bars(data_path: Path, hdf_key: str) -> pd.DataFrame:
    bars = pd.read_hdf(data_path, key=hdf_key)
    if "timestamp" in bars.columns:
        bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
        bars = bars.set_index("timestamp")
    elif not isinstance(bars.index, pd.DatetimeIndex):
        raise ValueError("Bars must have a timestamp column or DatetimeIndex")
    elif bars.index.tz is None:
        bars.index = bars.index.tz_localize("UTC")
    else:
        bars.index = bars.index.tz_convert("UTC")
    return bars.sort_index()


def _load_bars(data_paths: list[Path], hdf_key: str) -> pd.DataFrame:
    frames = []
    for data_path in data_paths:
        frames.append(_load_single_bars(data_path, hdf_key))

    if not frames:
        raise ValueError("No data paths provided")

    bars = pd.concat(frames).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    return bars


def _derive_cost_mode(labeling_cfg: dict) -> str:
    tb_cfg = (labeling_cfg.get("primary_labeling") or {}).get("triple_barrier", {})
    return "net_in_events" if tb_cfg.get("account_for_costs", False) else "gross_in_events"


def _ensure_weights(events_df: pd.DataFrame) -> pd.DataFrame:
    out = events_df.copy()
    if "w_final" not in out.columns:
        out["w_final"] = 1.0
    return out


def _apply_sample_decay(events_df: pd.DataFrame, training_cfg: dict) -> pd.DataFrame:
    decay_cfg = training_cfg.get("sample_decay", {}) or {}
    if not decay_cfg.get("enabled", False) or "t0" not in events_df.columns:
        return events_df

    out = events_df.copy()
    decay_lambda = float(decay_cfg.get("lambda", 0.0) or 0.0)
    if decay_lambda <= 0.0:
        return out

    ref_date = decay_cfg.get("reference_date")
    if ref_date is None:
        ref_ts = _to_utc_timestamp(out["t0"].max())
    else:
        ref_ts = _to_utc_timestamp(ref_date)

    age_days = (ref_ts - out["t0"]).dt.total_seconds() / 86400.0
    weights = np.exp(-decay_lambda * age_days)
    out["w_final"] = out["w_final"].astype(float) * weights
    return out


def _prepare_events_and_dataset(
    bars_df: pd.DataFrame,
    bar_size: str,
    labeling_cfg: dict,
    execution_spec: dict,
    instrument_spec,
    feature_cfg: dict,
    training_cfg: dict,
    balance_train: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = generate_events(
        bars_df=bars_df,
        bar_size=bar_size,
        labeling_config=labeling_cfg,
        execution_spec=execution_spec,
    )
    events = apply_triplebarrier(
        bars_df=bars_df,
        events_df=events,
        bar_size=bar_size,
        labeling_config=labeling_cfg,
        execution_spec=execution_spec,
        instrument_spec=instrument_spec,
    )

    tb_cfg = (labeling_cfg.get("primary_labeling") or {}).get("triple_barrier", {})
    if tb_cfg.get("drop_vertical_barrier", False):
        events = events[events["y"] != 0].reset_index(drop=True)

    if balance_train:
        event_cfg = training_cfg.get("event_generation", {}) or {}
        if event_cfg.get("balance_events", True):
            events = balance_events(
                events=events,
                target_long_ratio=float(event_cfg.get("target_long_ratio", 0.50)),
                method=event_cfg.get("balance_method", "undersample"),
            )

    events = _ensure_weights(events)
    events = _apply_sample_decay(events, training_cfg)

    features = build_features(bars_df, bar_size, feature_cfg)
    feature_rows = features.reindex(events["t0"]).reset_index(drop=True)

    model_df = pd.concat(
        [
            events[["event_id", "t0", "side", "y", "w_final"]].reset_index(drop=True),
            feature_rows,
        ],
        axis=1,
    )
    valid = ~model_df.isna().any(axis=1)
    model_df = model_df.loc[valid].reset_index(drop=True)
    events = events.loc[valid].reset_index(drop=True)

    regime_routing_cfg = training_cfg.get("regime_routing", {}) or {}
    meta_cfg = training_cfg.get("meta", {}) or {}
    target_route_cfg = training_cfg.get("target_routes", {}) or {}
    need_regime_columns = bool(regime_routing_cfg.get("enabled", False)) or bool(
        meta_cfg.get("enabled", False)
    ) or bool(target_route_cfg.get("enabled", False))
    if need_regime_columns:
        regime_events = compute_regime_at_events(
            events_df=events[["event_id", "t0"]].copy(),
            bars_df=bars_df,
            vol_window=int(
                regime_routing_cfg.get(
                    "vol_window",
                    (meta_cfg.get("route", {}) or {}).get("vol_window", 20),
                )
            ),
            trend_window=int(
                regime_routing_cfg.get(
                    "trend_window",
                    (meta_cfg.get("route", {}) or {}).get("trend_window", 50),
                )
            ),
        )
        regime_cols = [
            "event_id",
            "vol_regime",
            "trend_regime",
            "combined_regime",
            "regime_label",
        ]
        model_df = model_df.merge(regime_events[regime_cols], on="event_id", how="left")
    model_df = _apply_target_routes(model_df=model_df, events_df=events, training_cfg=training_cfg)
    return events, model_df


def _prepare_events_and_dataset_session_day(
    bars_df: pd.DataFrame,
    session_date: date,
    bar_size: str,
    labeling_cfg: dict,
    execution_spec: dict,
    instrument_spec,
    feature_cfg: dict,
    training_cfg: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Like _prepare_events_and_dataset with balance_train=False, but only keeps events whose
    t0 falls on ``session_date`` in America/Chicago (after generate_events).

    ``bars_df`` should include enough history before that session for indicators / event rules
    (e.g. concatenation of train bars and that session's bars).
    """
    events = generate_events(
        bars_df=bars_df,
        bar_size=bar_size,
        labeling_config=labeling_cfg,
        execution_spec=execution_spec,
    )
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    ts = events["t0"]
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    mask = ts.dt.tz_convert("America/Chicago").dt.date == session_date
    events = events.loc[mask].reset_index(drop=True)
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()

    events = apply_triplebarrier(
        bars_df=bars_df,
        events_df=events,
        bar_size=bar_size,
        labeling_config=labeling_cfg,
        execution_spec=execution_spec,
        instrument_spec=instrument_spec,
    )

    tb_cfg = (labeling_cfg.get("primary_labeling") or {}).get("triple_barrier", {})
    if tb_cfg.get("drop_vertical_barrier", False):
        events = events[events["y"] != 0].reset_index(drop=True)

    events = _ensure_weights(events)
    events = _apply_sample_decay(events, training_cfg)

    features = build_features(bars_df, bar_size, feature_cfg)
    feature_rows = features.reindex(events["t0"]).reset_index(drop=True)

    model_df = pd.concat(
        [
            events[["event_id", "t0", "side", "y", "w_final"]].reset_index(drop=True),
            feature_rows,
        ],
        axis=1,
    )
    valid = ~model_df.isna().any(axis=1)
    model_df = model_df.loc[valid].reset_index(drop=True)
    events = events.loc[valid].reset_index(drop=True)

    regime_routing_cfg = training_cfg.get("regime_routing", {}) or {}
    meta_cfg = training_cfg.get("meta", {}) or {}
    target_route_cfg = training_cfg.get("target_routes", {}) or {}
    need_regime_columns = bool(regime_routing_cfg.get("enabled", False)) or bool(
        meta_cfg.get("enabled", False)
    ) or bool(target_route_cfg.get("enabled", False))
    if need_regime_columns:
        regime_events = compute_regime_at_events(
            events_df=events[["event_id", "t0"]].copy(),
            bars_df=bars_df,
            vol_window=int(
                regime_routing_cfg.get(
                    "vol_window",
                    (meta_cfg.get("route", {}) or {}).get("vol_window", 20),
                )
            ),
            trend_window=int(
                regime_routing_cfg.get(
                    "trend_window",
                    (meta_cfg.get("route", {}) or {}).get("trend_window", 50),
                )
            ),
        )
        regime_cols = [
            "event_id",
            "vol_regime",
            "trend_regime",
            "combined_regime",
            "regime_label",
        ]
        model_df = model_df.merge(regime_events[regime_cols], on="event_id", how="left")
    model_df = _apply_target_routes(model_df=model_df, events_df=events, training_cfg=training_cfg)
    return events, model_df


def _binary_target(y_values: np.ndarray, positive_label: int) -> np.ndarray:
    return (np.asarray(y_values) == positive_label).astype(int)


def _negative_label(training_cfg: dict) -> int:
    target_cfg = training_cfg.get("target", {}) or {}
    positive_label = int(target_cfg.get("positive_label", 1))
    classes = [int(v) for v in (target_cfg.get("classes", []) or [])]
    for value in classes:
        if value != positive_label:
            return value
    return 0 if positive_label != 0 else -1


def _normalize_target_routes(training_cfg: dict) -> list[dict]:
    target_route_cfg = training_cfg.get("target_routes", {}) or {}
    routes = []
    for idx, route in enumerate(target_route_cfg.get("routes", []) or []):
        if not route:
            continue
        normalized = dict(route)
        normalized["name"] = normalized.get("name", f"target_route_{idx}")
        normalized["side"] = str(normalized.get("side", "both")).lower()
        normalized["regimes"] = [int(r) for r in (normalized.get("regimes", []) or [])]
        normalized["target_kind"] = str(normalized.get("target_kind", "ret_net_positive")).lower()
        routes.append(normalized)
    return routes


def _route_binary_label(
    route_df: pd.DataFrame,
    route: dict,
    training_cfg: dict,
) -> pd.Series:
    positive_label = int((training_cfg.get("target", {}) or {}).get("positive_label", 1))
    negative_label = _negative_label(training_cfg)
    target_kind = route["target_kind"]

    if target_kind == "y_positive":
        positive_mask = route_df["y_original"] == positive_label
    elif target_kind == "ret_net_positive":
        if "ret_net" not in route_df.columns:
            raise ValueError("ret_net not available for target route relabeling")
        positive_mask = route_df["ret_net"] > 0
    elif target_kind == "y_and_ret_net":
        if "ret_net" not in route_df.columns:
            raise ValueError("ret_net not available for target route relabeling")
        positive_mask = (route_df["y_original"] == positive_label) & (route_df["ret_net"] > 0)
    else:
        raise ValueError(f"Unsupported target route kind: {target_kind}")

    relabeled = pd.Series(negative_label, index=route_df.index, dtype=int)
    relabeled.loc[positive_mask] = positive_label
    return relabeled


def _apply_target_routes(
    model_df: pd.DataFrame,
    events_df: pd.DataFrame,
    training_cfg: dict,
) -> pd.DataFrame:
    target_route_cfg = training_cfg.get("target_routes", {}) or {}
    if not target_route_cfg.get("enabled", False):
        return model_df

    out = model_df.copy()
    out["y_original"] = out["y"].astype(int)
    out["target_route_name"] = ""

    if "combined_regime" not in out.columns:
        raise ValueError("combined_regime not available for target route relabeling")

    ret_cols = ["event_id"]
    if "ret_net" in events_df.columns:
        ret_cols.append("ret_net")
    if len(ret_cols) > 1:
        out = out.merge(events_df[ret_cols], on="event_id", how="left")
    elif "ret_net" not in out.columns:
        out["ret_net"] = np.nan

    for route in _normalize_target_routes(training_cfg):
        mask = _candidate_side_mask(out["side"], route["side"])
        if route["regimes"]:
            mask &= out["combined_regime"].isin(route["regimes"])
        if not mask.any():
            continue
        out.loc[mask, "y"] = _route_binary_label(out.loc[mask], route, training_cfg)
        out.loc[mask, "target_route_name"] = route["name"]

    if "ret_net" in out.columns and "ret_net" not in model_df.columns:
        out = out.drop(columns=["ret_net"])
    return out


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    unique = np.unique(y_true)
    if len(unique) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out):
        return None
    return out


def _safe_spearman(x, y) -> float | None:
    x_ser = pd.Series(x, dtype=float)
    y_ser = pd.Series(y, dtype=float)
    valid = x_ser.notna() & y_ser.notna()
    if valid.sum() < 3:
        return None
    x_valid = x_ser.loc[valid]
    y_valid = y_ser.loc[valid]
    if x_valid.nunique() < 2 or y_valid.nunique() < 2:
        return None
    corr = x_valid.corr(y_valid, method="spearman")
    return None if pd.isna(corr) else float(corr)


def _longest_streak(mask) -> int:
    longest = 0
    current = 0
    for value in pd.Series(mask).fillna(False).astype(bool):
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _normalize_group_value(value):
    if pd.isna(value):
        return "unknown"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        as_float = float(value)
        if as_float.is_integer():
            return int(as_float)
        return as_float
    return str(value)


def _expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float | None, list[dict]]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    if valid.sum() == 0:
        return None, []

    y_true = y_true[valid]
    y_prob = y_prob[valid]
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(y_true)
    ece = 0.0
    rows = []
    for idx in range(n_bins):
        lower = bins[idx]
        upper = bins[idx + 1]
        if idx == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        if not mask.any():
            continue
        avg_conf = float(y_prob[mask].mean())
        avg_target = float(y_true[mask].mean())
        weight = float(mask.mean())
        gap = abs(avg_conf - avg_target)
        ece += gap * weight
        rows.append(
            {
                "bin": idx + 1,
                "rows": int(mask.sum()),
                "prob_min": float(lower),
                "prob_max": float(upper),
                "avg_confidence": avg_conf,
                "target_rate": avg_target,
                "gap": float(gap),
            }
        )
    return float(ece) if total else None, rows


def _score_quantile_profile(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    return_col: str = "ret_net",
    n_quantiles: int = 10,
) -> list[dict]:
    if score_col not in df.columns or target_col not in df.columns:
        return []

    cols = [score_col, target_col]
    if return_col in df.columns:
        cols.append(return_col)
    work = df[cols].copy()
    work = work[work[score_col].notna() & work[target_col].notna()].reset_index(drop=True)
    if len(work) < 2:
        return []

    q = min(int(n_quantiles), len(work))
    try:
        work["quantile"] = pd.qcut(
            work[score_col].rank(method="first"),
            q=q,
            labels=False,
            duplicates="drop",
        )
    except ValueError:
        return []
    if work["quantile"].isna().all():
        return []

    rows = []
    for quantile, group in work.groupby("quantile", sort=True):
        rows.append(
            {
                "quantile": int(quantile) + 1,
                "rows": int(len(group)),
                "score_mean": float(group[score_col].mean()),
                "score_min": float(group[score_col].min()),
                "score_max": float(group[score_col].max()),
                "target_rate": float(group[target_col].mean()),
                "avg_ret_net": (
                    float(group[return_col].mean())
                    if return_col in group.columns and group[return_col].notna().any()
                    else None
                ),
            }
        )
    return rows


def _summarize_score_frame(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    return_col: str = "ret_net",
    *,
    include_profiles: bool,
) -> dict:
    if score_col not in df.columns or target_col not in df.columns:
        return {}

    work = df.copy()
    work = work[work[score_col].notna() & work[target_col].notna()].reset_index(drop=True)
    if work.empty:
        return {}

    y_true = work[target_col].to_numpy(dtype=float)
    y_prob = work[score_col].to_numpy(dtype=float)
    ece, calibration_bins = _expected_calibration_error(y_true, y_prob)
    quantile_profile = _score_quantile_profile(
        work,
        score_col=score_col,
        target_col=target_col,
        return_col=return_col,
    )
    top_quantile = quantile_profile[-1] if quantile_profile else {}
    bottom_quantile = quantile_profile[0] if quantile_profile else {}

    result = {
        "rows": int(len(work)),
        "score_mean": float(np.mean(y_prob)),
        "score_std": float(np.std(y_prob)) if len(work) > 1 else 0.0,
        "target_rate": float(np.mean(y_true)),
        "avg_ret_net": (
            float(work[return_col].mean())
            if return_col in work.columns and work[return_col].notna().any()
            else None
        ),
        "brier_score": float(brier_score_loss(y_true.astype(int), y_prob)),
        "ece": ece,
        "rank_ic_ret": (
            _safe_spearman(work[score_col], work[return_col])
            if return_col in work.columns
            else None
        ),
        "top_quantile_avg_ret_net": top_quantile.get("avg_ret_net"),
        "bottom_quantile_avg_ret_net": bottom_quantile.get("avg_ret_net"),
        "decile_spread_ret_net": (
            None
            if top_quantile.get("avg_ret_net") is None or bottom_quantile.get("avg_ret_net") is None
            else float(top_quantile["avg_ret_net"] - bottom_quantile["avg_ret_net"])
        ),
    }
    if include_profiles:
        result["calibration_bins"] = calibration_bins
        result["quantile_profile"] = quantile_profile
    return result


def _group_score_diagnostics(
    df: pd.DataFrame,
    group_cols: list[str],
    score_col: str,
    target_col: str,
    return_col: str = "ret_net",
    min_rows: int = 5,
) -> list[dict]:
    if df.empty:
        return []

    results = []
    for group_key, group_df in df.groupby(group_cols, dropna=False):
        if len(group_df) < min_rows:
            continue
        metrics = _summarize_score_frame(
            group_df,
            score_col=score_col,
            target_col=target_col,
            return_col=return_col,
            include_profiles=False,
        )
        if not metrics:
            continue
        row = {}
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        for col_name, value in zip(group_cols, group_key):
            row[col_name] = _normalize_group_value(value)
        row.update(metrics)
        results.append(row)

    results.sort(
        key=lambda item: (
            item.get("avg_ret_net") is None,
            item.get("avg_ret_net") if item.get("avg_ret_net") is not None else 0.0,
        )
    )
    return results


def _build_score_diagnostics(
    base_df: pd.DataFrame,
    score_col: str,
    target_col: str,
    *,
    return_col: str = "ret_net",
) -> dict:
    if base_df.empty:
        return {}

    work = base_df.copy()
    work["month"] = pd.to_datetime(work["t0"], utc=True).dt.strftime("%Y-%m")
    work["side_label"] = np.where(work["side"].fillna(1).astype(float) >= 0, "long", "short")

    regime_cols = [col for col in ["combined_regime", "regime_label"] if col in work.columns]
    diagnostics = {
        "overall": _summarize_score_frame(
            work,
            score_col=score_col,
            target_col=target_col,
            return_col=return_col,
            include_profiles=True,
        ),
        "by_month": _group_score_diagnostics(
            work,
            ["month"],
            score_col=score_col,
            target_col=target_col,
            return_col=return_col,
            min_rows=4,
        ),
        "by_side": _group_score_diagnostics(
            work,
            ["side_label"],
            score_col=score_col,
            target_col=target_col,
            return_col=return_col,
            min_rows=4,
        ),
        "by_month_side": _group_score_diagnostics(
            work,
            ["month", "side_label"],
            score_col=score_col,
            target_col=target_col,
            return_col=return_col,
            min_rows=4,
        ),
    }
    if regime_cols:
        diagnostics["by_regime"] = _group_score_diagnostics(
            work,
            regime_cols,
            score_col=score_col,
            target_col=target_col,
            return_col=return_col,
            min_rows=4,
        )
        diagnostics["problem_groups"] = _group_score_diagnostics(
            work,
            ["month", "side_label", *regime_cols],
            score_col=score_col,
            target_col=target_col,
            return_col=return_col,
            min_rows=3,
        )[:10]
    else:
        diagnostics["by_regime"] = []
        diagnostics["problem_groups"] = []
    return diagnostics


def _build_primary_score_diagnostics(
    test_df: pd.DataFrame,
    test_events: pd.DataFrame,
    primary_preds: pd.DataFrame,
    positive_label: int,
) -> dict:
    cols = ["event_id", "t0", "side", "y"]
    cols += [c for c in ["combined_regime", "regime_label"] if c in test_df.columns]
    merged = test_df[cols].merge(
        primary_preds[["event_id", "y_prob", "model_source"]],
        on="event_id",
        how="inner",
    )
    if "ret_net" in test_events.columns:
        merged = merged.merge(
            test_events[["event_id", "ret_net"]],
            on="event_id",
            how="left",
        )
    merged["target"] = _binary_target(merged["y"].to_numpy(), positive_label)
    return _build_score_diagnostics(merged, "y_prob", "target")


def _build_meta_score_diagnostics(
    meta_result: dict,
    test_df: pd.DataFrame,
    test_events: pd.DataFrame,
) -> dict:
    if not meta_result.get("enabled", False):
        return {"enabled": False}
    if meta_result.get("skipped", False):
        return {
            "enabled": True,
            "skipped": True,
            "reason": meta_result.get("reason"),
            "test_rows": int(len(meta_result.get("meta_test_df", []))),
        }

    meta_test_df = meta_result.get("meta_test_df", pd.DataFrame()).copy()
    meta_preds = meta_result.get("meta_preds", pd.DataFrame()).copy()
    if meta_test_df.empty or meta_preds.empty:
        return {
            "enabled": True,
            "skipped": True,
            "reason": "missing_meta_predictions",
            "test_rows": int(len(meta_test_df)),
        }

    cols = ["event_id", "t0", "side"]
    cols += [c for c in ["combined_regime", "regime_label"] if c in test_df.columns]
    merged = meta_test_df.merge(meta_preds[["event_id", "p_meta", "meta_source"]], on="event_id", how="inner")
    merged = merged.merge(test_df[cols], on="event_id", how="left", suffixes=("", "_ctx"))
    for col in ["t0", "side", "combined_regime", "regime_label"]:
        ctx_col = f"{col}_ctx"
        if ctx_col not in merged.columns:
            continue
        if col not in merged.columns:
            merged[col] = merged[ctx_col]
        else:
            merged[col] = merged[col].where(merged[col].notna(), merged[ctx_col])
    if "ret_net" in test_events.columns:
        merged = merged.merge(test_events[["event_id", "ret_net"]], on="event_id", how="left")

    diagnostics = _build_score_diagnostics(merged, "p_meta", "y")
    diagnostics["enabled"] = True
    diagnostics["skipped"] = False
    diagnostics["test_rows"] = int(len(meta_test_df))
    return diagnostics


def _build_risk_tearsheet(trades_df: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    if trades_df is None or trades_df.empty:
        return {
            "executed_trades": 0,
            "daily": {},
            "streaks": {"longest_win_streak": 0, "longest_loss_streak": 0},
            "tail": {},
            "equity": {},
        }

    executed = trades_df.copy()
    if "executed" in executed.columns:
        executed = executed[executed["executed"]].copy()
    if executed.empty:
        return {
            "executed_trades": 0,
            "daily": {},
            "streaks": {"longest_win_streak": 0, "longest_loss_streak": 0},
            "tail": {},
            "equity": {},
        }

    executed["pnl_usd"] = pd.to_numeric(executed["pnl_usd"], errors="coerce")
    executed["exit_ts"] = pd.to_datetime(executed["exit_ts"], utc=True, errors="coerce")
    valid_exits = executed["exit_ts"].notna()
    daily_summary = {}
    if valid_exits.any():
        session_day = executed.loc[valid_exits, "exit_ts"].dt.tz_convert("America/Chicago").dt.strftime("%Y-%m-%d")
        daily_pnl = executed.loc[valid_exits].groupby(session_day)["pnl_usd"].sum().sort_index()
        daily_values = daily_pnl.to_numpy(dtype=float)
        downside = daily_values[daily_values < 0]
        daily_std = float(np.std(daily_values, ddof=1)) if len(daily_values) > 1 else None
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else None
        daily_summary = {
            "days": int(len(daily_pnl)),
            "avg_day_usd": float(daily_pnl.mean()),
            "best_day_usd": float(daily_pnl.max()),
            "worst_day_usd": float(daily_pnl.min()),
            "daily_win_rate": float((daily_pnl > 0).mean()),
            "daily_pnl_std": daily_std,
            "daily_sharpe": (
                float(daily_pnl.mean() / daily_std * np.sqrt(252))
                if daily_std and daily_std > 0
                else None
            ),
            "daily_sortino": (
                float(daily_pnl.mean() / downside_std * np.sqrt(252))
                if downside_std and downside_std > 0
                else None
            ),
            "daily_pnl_by_day": {
                str(day): float(value) for day, value in daily_pnl.items()
            },
        }

    pnl_series = executed["pnl_usd"].fillna(0.0)
    win_mask = pnl_series > 0
    loss_mask = pnl_series < 0
    streaks = {
        "longest_win_streak": _longest_streak(win_mask),
        "longest_loss_streak": _longest_streak(loss_mask),
    }

    tail = {
        "best_trade_usd": float(pnl_series.max()),
        "worst_trade_usd": float(pnl_series.min()),
        "trade_pnl_p05": float(np.percentile(pnl_series, 5)),
        "trade_pnl_p95": float(np.percentile(pnl_series, 95)),
        "trade_pnl_std": (
            float(np.std(pnl_series.to_numpy(dtype=float), ddof=1))
            if len(pnl_series) > 1
            else 0.0
        ),
    }

    equity_summary = {}
    if equity_df is not None and not equity_df.empty:
        eq = equity_df.copy()
        eq["timestamp"] = pd.to_datetime(eq["timestamp"], utc=True, errors="coerce")
        eq = eq.sort_values("timestamp").reset_index(drop=True)
        equity = pd.to_numeric(eq["equity"], errors="coerce").dropna()
        if not equity.empty:
            peak = equity.cummax()
            drawdown = peak - equity
            underwater = drawdown > 0
            total_pnl = float(pnl_series.sum())
            max_dd = float(drawdown.max()) if len(drawdown) else 0.0
            equity_summary = {
                "final_equity": float(equity.iloc[-1]),
                "max_drawdown_usd": max_dd,
                "recovery_factor": (float(total_pnl / max_dd) if max_dd > 0 else None),
                "longest_underwater_bars": _longest_streak(underwater),
            }

    return {
        "executed_trades": int(len(executed)),
        "daily": daily_summary,
        "streaks": streaks,
        "tail": tail,
        "equity": equity_summary,
    }


def _summarize_target_routes(model_df: pd.DataFrame, events_df: pd.DataFrame | None = None) -> dict:
    if "target_route_name" not in model_df.columns:
        return {"enabled": False, "rows_relabeled": 0, "routes": []}

    relabeled = model_df[model_df["target_route_name"].fillna("") != ""].copy()
    if events_df is not None and "ret_net" in events_df.columns and "ret_net" not in relabeled.columns:
        relabeled = relabeled.merge(events_df[["event_id", "ret_net"]], on="event_id", how="left")
    if relabeled.empty:
        return {"enabled": True, "rows_relabeled": 0, "routes": []}

    routes = []
    for route_name, group in relabeled.groupby("target_route_name", sort=True):
        routes.append(
            {
                "name": str(route_name),
                "rows": int(len(group)),
                "positive_rate": float((group["y"] == group["y"].max()).mean()),
                "avg_ret_net": float(group["ret_net"].mean()) if "ret_net" in group.columns and group["ret_net"].notna().any() else None,
                "regimes": (
                    sorted(group["combined_regime"].dropna().astype(int).unique().tolist())
                    if "combined_regime" in group.columns
                    else []
                ),
            }
        )

    return {
        "enabled": True,
        "rows_relabeled": int(len(relabeled)),
        "routes": routes,
    }


def _fit_binary_candidate_model(train_df: pd.DataFrame, training_cfg: dict) -> dict:
    feature_cols = [c for c in train_df.columns if c not in NON_FEATURE_COLUMNS]
    pre = FoldPreprocessor(feature_cols, training_cfg).fit(train_df)
    X_train, y_train_raw, w_train = pre.transform(train_df)

    target_cfg = training_cfg.get("target", {}) or {}
    positive_label = int(target_cfg.get("positive_label", 1))
    y_train = _binary_target(y_train_raw, positive_label)
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training window has single-class labels after filtering")

    model_params = dict((training_cfg.get("model", {}) or {}).get("params", {}))
    model_params.pop("objective", None)
    model = LGBMClassifier(
        objective="binary",
        random_state=int(training_cfg.get("seed", 42)),
        verbose=-1,
        **model_params,
    )

    cal_cfg = training_cfg.get("calibration", {}) or {}
    calibration_enabled = bool(cal_cfg.get("enabled", False))
    calibrator = None
    train_X = X_train
    train_y = y_train
    train_w = w_train

    if calibration_enabled and len(X_train) >= 50 and y_train.sum() > 1 and (len(y_train) - y_train.sum()) > 1:
        cal_fraction = float(cal_cfg.get("calibration_fraction", 0.20))
        cal_method = cal_cfg.get("method", "isotonic")
        cal_seed = int(cal_cfg.get("calibration_seed", 42))

        if train_w is not None:
            (
                train_X,
                cal_X,
                train_y,
                cal_y,
                train_w,
                _cal_w,
            ) = train_test_split(
                X_train,
                y_train,
                train_w,
                test_size=cal_fraction,
                random_state=cal_seed,
                stratify=y_train,
            )
        else:
            train_X, cal_X, train_y, cal_y = train_test_split(
                X_train,
                y_train,
                test_size=cal_fraction,
                random_state=cal_seed,
                stratify=y_train,
            )

        model.fit(train_X, train_y, sample_weight=train_w)
        cal_proba = model.predict_proba(cal_X)[:, 1]
        if cal_method == "sigmoid":
            logistic_cal = LogisticRegression(solver="lbfgs", max_iter=1000)
            logistic_cal.fit(cal_proba.reshape(-1, 1), cal_y)
            calibrator = SigmoidCalibratorWrapper(logistic_cal)
        else:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(cal_proba, cal_y)
        fitted_model = CalibratedBinaryModel(model, calibrator)
    else:
        model.fit(X_train, y_train, sample_weight=w_train)
        fitted_model = model

    return {
        "model": fitted_model,
        "preprocessor": pre,
        "feature_columns": feature_cols,
        "positive_label": positive_label,
    }


def _candidate_side_mask(side_series: pd.Series, side_name: str) -> pd.Series:
    if side_name == "long":
        return side_series.fillna(1).astype(float) >= 0
    if side_name == "short":
        return side_series.fillna(1).astype(float) < 0
    return pd.Series(True, index=side_series.index)


def _normalize_regime_routes(training_cfg: dict) -> list[dict]:
    routing_cfg = training_cfg.get("regime_routing", {}) or {}
    routes = []
    for idx, route in enumerate(routing_cfg.get("routes", []) or []):
        if not route:
            continue
        normalized = dict(route)
        normalized["name"] = normalized.get("name", f"route_{idx}")
        normalized["regimes"] = [int(r) for r in (normalized.get("regimes", []) or [])]
        normalized["side"] = str(normalized.get("side", "both")).lower()
        normalized["min_train_samples"] = int(normalized.get("min_train_samples", 250))
        normalized["min_positive_samples"] = int(normalized.get("min_positive_samples", 25))
        normalized["min_negative_samples"] = int(normalized.get("min_negative_samples", 25))
        routes.append(normalized)
    return routes


def _normalize_meta_routes(training_cfg: dict) -> list[dict]:
    meta_cfg = training_cfg.get("meta", {}) or {}
    default_threshold_primary = float(
        meta_cfg.get(
            "threshold_primary",
            ((training_cfg.get("eval", {}) or {}).get("threshold", 0.5)),
        )
    )
    default_threshold_meta = float(meta_cfg.get("threshold_meta", 0.5))
    default_target_kind = str(
        ((meta_cfg.get("target", {}) or {}).get("kind", "ret_net_positive"))
    ).lower()
    raw_routes = meta_cfg.get("routes", []) or []
    if not raw_routes:
        raw_routes = [meta_cfg.get("route", {}) or {}]

    routes = []
    for idx, route_cfg in enumerate(raw_routes):
        if route_cfg is None:
            continue
        routes.append(
            {
                "name": str(route_cfg.get("name", f"meta_route_{idx}")),
                "side": str(route_cfg.get("side", "both")).lower(),
                "regimes": [int(r) for r in (route_cfg.get("regimes", []) or [])],
                "min_train_samples": int(route_cfg.get("min_train_samples", 120)),
                "min_positive_samples": int(route_cfg.get("min_positive_samples", 15)),
                "min_negative_samples": int(route_cfg.get("min_negative_samples", 15)),
                "threshold_primary": float(
                    route_cfg.get("threshold_primary", default_threshold_primary)
                ),
                "threshold_meta": float(route_cfg.get("threshold_meta", default_threshold_meta)),
                "target_kind": str(route_cfg.get("target_kind", default_target_kind)).lower(),
            }
        )
    return routes


def _normalize_meta_route(training_cfg: dict) -> dict:
    routes = _normalize_meta_routes(training_cfg)
    if routes:
        return routes[0]
    return {
        "name": "meta_route",
        "side": "both",
        "regimes": [],
        "min_train_samples": 120,
        "min_positive_samples": 15,
        "min_negative_samples": 15,
        "threshold_primary": float(
            ((training_cfg.get("meta", {}) or {}).get(
                "threshold_primary",
                ((training_cfg.get("eval", {}) or {}).get("threshold", 0.5)),
            ))
        ),
        "threshold_meta": float(((training_cfg.get("meta", {}) or {}).get("threshold_meta", 0.5))),
        "target_kind": str(
            (((training_cfg.get("meta", {}) or {}).get("target", {}) or {}).get(
                "kind", "ret_net_positive"
            ))
        ).lower(),
    }


def _build_targeted_meta_dataset(
    primary_preds_df: pd.DataFrame,
    base_event_dataset_df: pd.DataFrame,
    events_df: pd.DataFrame,
    training_cfg: dict,
    route: dict | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    meta_cfg = training_cfg.get("meta", {}) or {}
    if not meta_cfg.get("enabled", False):
        return pd.DataFrame(), []

    route = route or _normalize_meta_route(training_cfg)
    threshold_primary = float(route["threshold_primary"])
    preds = primary_preds_df[["event_id", "y_prob"]].copy()
    preds = preds[preds["y_prob"] >= threshold_primary].reset_index(drop=True)
    if preds.empty:
        return pd.DataFrame(), []

    base_df = base_event_dataset_df.merge(preds, on="event_id", how="inner")
    if route["regimes"]:
        if "combined_regime" not in base_df.columns:
            raise ValueError("combined_regime not available for meta route filter")
        route_mask = base_df["combined_regime"].isin(route["regimes"])
        route_mask &= _candidate_side_mask(base_df["side"], route["side"])
        base_df = base_df.loc[route_mask].reset_index(drop=True)
        if base_df.empty:
            return pd.DataFrame(), []

    events_cols = ["event_id", "y"]
    if "ret_net" in events_df.columns:
        events_cols.append("ret_net")
    merged = base_df.merge(events_df[events_cols], on="event_id", how="left", suffixes=("", "_evt"))

    target_kind = route["target_kind"]
    if target_kind == "y_positive":
        meta_y = (merged["y_evt"] == 1).astype(int)
    elif target_kind == "ret_net_positive":
        if "ret_net" not in merged.columns:
            raise ValueError("ret_net not available for meta target")
        meta_y = (merged["ret_net"] > 0).astype(int)
    elif target_kind == "y_and_ret_net":
        if "ret_net" not in merged.columns:
            raise ValueError("ret_net not available for meta target")
        meta_y = ((merged["y_evt"] == 1) & (merged["ret_net"] > 0)).astype(int)
    else:
        raise ValueError(f"Unsupported meta target kind: {target_kind}")

    features_cfg = meta_cfg.get("features", {}) or {}
    include_primary_prob = bool(features_cfg.get("include_primary_prob", True))
    include_primary_logit = bool(features_cfg.get("include_primary_logit", True))
    include_original = bool(features_cfg.get("include_original_features", True))

    base_feature_cols: list[str] = []
    if include_original:
        for col in base_event_dataset_df.columns:
            if col in META_EXCLUDED_COLUMNS:
                continue
            if col not in merged.columns:
                continue
            if not pd.api.types.is_numeric_dtype(merged[col]):
                continue
            base_feature_cols.append(col)

    feature_cols: list[str] = []
    if include_original:
        feature_cols.extend(base_feature_cols)
    if include_primary_prob:
        feature_cols.append("p_primary")
    if include_primary_logit:
        feature_cols.append("p_primary_logit")

    merged["p_primary"] = merged["y_prob"].astype(float)
    eps = 1e-6
    p_clip = merged["p_primary"].clip(eps, 1.0 - eps)
    merged["p_primary_logit"] = np.log(p_clip / (1.0 - p_clip))

    meta_df = pd.DataFrame(
        {
            "event_id": merged["event_id"].to_numpy(),
            "y": meta_y.to_numpy(),
            "w_final": merged["w_final"].to_numpy(),
            "p_primary": merged["p_primary"].to_numpy(),
            "p_primary_logit": merged["p_primary_logit"].to_numpy(),
        }
    )
    if include_original and base_feature_cols:
        meta_df = pd.concat([meta_df, merged[base_feature_cols].reset_index(drop=True)], axis=1)

    base_cols = ["event_id", "y", "w_final", "p_primary", "p_primary_logit"]
    ordered_cols = base_cols + [c for c in feature_cols if c not in base_cols]
    meta_df = meta_df[ordered_cols]
    return meta_df, feature_cols


def _train_single_targeted_meta_route(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    primary_train_preds: pd.DataFrame,
    primary_test_preds: pd.DataFrame,
    training_cfg: dict,
    route: dict,
) -> dict:
    meta_cfg = training_cfg.get("meta", {}) or {}
    meta_train_df, meta_feature_cols = _build_targeted_meta_dataset(
        primary_preds_df=primary_train_preds,
        base_event_dataset_df=train_df,
        events_df=train_events,
        training_cfg=training_cfg,
        route=route,
    )
    meta_test_df, _ = _build_targeted_meta_dataset(
        primary_preds_df=primary_test_preds,
        base_event_dataset_df=test_df,
        events_df=test_events,
        training_cfg=training_cfg,
        route=route,
    )

    route_result = {
        "name": route["name"],
        "route": route,
        "meta_test_df": meta_test_df,
        "feature_cols": meta_feature_cols,
    }
    if meta_train_df.empty or meta_test_df.empty:
        return {
            **route_result,
            "skipped": True,
            "reason": "no_targeted_meta_samples",
        }

    if len(meta_train_df) < route["min_train_samples"]:
        return {
            **route_result,
            "skipped": True,
            "reason": "insufficient_targeted_meta_train_samples",
        }

    y_train_counts = meta_train_df["y"].value_counts()
    positive_count = int(y_train_counts.get(1, 0))
    negative_count = int(y_train_counts.get(0, 0))
    if (
        positive_count < route["min_positive_samples"]
        or negative_count < route["min_negative_samples"]
    ):
        return {
            **route_result,
            "skipped": True,
            "reason": "insufficient_targeted_meta_class_balance",
        }

    pre = FoldPreprocessor(meta_feature_cols, training_cfg).fit(meta_train_df)
    X_train, y_train, w_train = pre.transform(meta_train_df)
    X_test, y_test, w_test = pre.transform(meta_test_df)
    if len(np.unique(y_train)) < 2:
        return {
            **route_result,
            "skipped": True,
            "reason": "single_class_targeted_meta_train",
        }

    params = (meta_cfg.get("model", {}) or {}).get("params", {})
    model = LogisticRegression(
        C=params.get("C", 1.0),
        penalty=params.get("penalty", "l2"),
        solver=params.get("solver", "lbfgs"),
        max_iter=params.get("max_iter", 200),
        class_weight=params.get("class_weight"),
        random_state=int(training_cfg.get("seed", 42)),
    )
    use_weight = bool((training_cfg.get("sample_weight", {}) or {}).get("enabled", False))
    model.fit(X_train, y_train, sample_weight=w_train if use_weight else None)

    threshold = float(route["threshold_meta"])
    p_meta = model.predict_proba(X_test)[:, 1]
    m_pred = (p_meta >= threshold).astype(int)
    metrics = compute_metrics(
        y_test,
        p_meta,
        threshold=threshold,
        sample_weight=w_test if use_weight else None,
    )
    meta_preds = pd.DataFrame(
        {
            "event_id": meta_test_df["event_id"].to_numpy(),
            "p_meta": p_meta,
            "meta_source": np.full(len(meta_test_df), route["name"], dtype=object),
        }
    )
    return {
        **route_result,
        "skipped": False,
        "model": model,
        "preprocessor": pre,
        "meta_preds": meta_preds,
        "p_meta": p_meta,
        "m_pred": m_pred,
        "metrics": metrics,
        "w_test": w_test if use_weight else np.ones(len(meta_test_df), dtype=float),
    }


def _train_targeted_meta_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_events: pd.DataFrame,
    test_events: pd.DataFrame,
    primary_train_preds: pd.DataFrame,
    primary_test_preds: pd.DataFrame,
    training_cfg: dict,
) -> dict:
    meta_cfg = training_cfg.get("meta", {}) or {}
    if not meta_cfg.get("enabled", False):
        return {"enabled": False}

    route_results = [
        _train_single_targeted_meta_route(
            train_df=train_df,
            test_df=test_df,
            train_events=train_events,
            test_events=test_events,
            primary_train_preds=primary_train_preds,
            primary_test_preds=primary_test_preds,
            training_cfg=training_cfg,
            route=route,
        )
        for route in _normalize_meta_routes(training_cfg)
    ]
    successful_routes = [route for route in route_results if not route.get("skipped", False)]
    _skipped_frames = [
        route["meta_test_df"] for route in route_results if not route["meta_test_df"].empty
    ]
    skipped_meta_test_df = (
        pd.concat(_skipped_frames, ignore_index=True) if _skipped_frames else pd.DataFrame()
    )
    if not successful_routes:
        reasons = [route.get("reason") for route in route_results if route.get("reason")]
        reason = reasons[0] if len(set(reasons)) == 1 and reasons else "all_targeted_meta_routes_skipped"
        return {
            "enabled": True,
            "skipped": True,
            "reason": reason,
            "meta_test_df": skipped_meta_test_df,
            "routes": route_results,
        }

    meta_test_df = pd.concat(
        [route["meta_test_df"] for route in successful_routes if not route["meta_test_df"].empty],
        ignore_index=True,
    )
    meta_preds = pd.concat(
        [route["meta_preds"] for route in successful_routes if not route["meta_preds"].empty],
        ignore_index=True,
    )
    duplicated_event_ids = meta_preds["event_id"].duplicated(keep=False)
    if duplicated_event_ids.any():
        duplicate_sources = meta_preds.loc[duplicated_event_ids, ["event_id", "meta_source"]]
        raise ValueError(
            "Overlapping meta routes produced duplicate meta predictions: "
            f"{duplicate_sources.to_dict(orient='records')}"
        )

    p_meta = np.concatenate([route["p_meta"] for route in successful_routes], dtype=float)
    m_pred = np.concatenate([route["m_pred"] for route in successful_routes], dtype=int)
    w_test = np.concatenate([route["w_test"] for route in successful_routes], dtype=float)
    metrics = (
        successful_routes[0]["metrics"]
        if len(successful_routes) == 1
        else {
            "mode": "multi_route",
            "routes": [
                {
                    "name": route["name"],
                    "threshold_primary": route["route"]["threshold_primary"],
                    "threshold_meta": route["route"]["threshold_meta"],
                    "test_rows": int(len(route["meta_test_df"])),
                    "metrics": route["metrics"],
                }
                for route in successful_routes
            ],
        }
    )
    return {
        "enabled": True,
        "skipped": False,
        "feature_cols": successful_routes[0].get("feature_cols", []),
        "meta_test_df": meta_test_df,
        "meta_preds": meta_preds,
        "p_meta": p_meta,
        "m_pred": m_pred,
        "metrics": metrics,
        "w_test": w_test,
        "routes": route_results,
    }


def _fit_regime_routed_candidate(train_df: pd.DataFrame, training_cfg: dict) -> dict:
    target_cfg = training_cfg.get("target", {}) or {}
    positive_label = int(target_cfg.get("positive_label", 1))
    global_candidate = _fit_binary_candidate_model(train_df, training_cfg)
    route_models = []

    for route in _normalize_regime_routes(training_cfg):
        if "combined_regime" not in train_df.columns:
            logger.warning(
                "Skipping regime route %s because combined_regime is unavailable",
                route["name"],
            )
            continue

        route_mask = train_df["combined_regime"].isin(route["regimes"])
        route_mask &= _candidate_side_mask(train_df["side"], route["side"])
        route_df = train_df.loc[route_mask].reset_index(drop=True)
        if len(route_df) < route["min_train_samples"]:
            logger.info(
                "Skipping regime route %s: only %d rows (need %d)",
                route["name"],
                len(route_df),
                route["min_train_samples"],
            )
            continue

        y_route = _binary_target(route_df["y"].to_numpy(), positive_label)
        positive_count = int(y_route.sum())
        negative_count = int(len(y_route) - positive_count)
        if (
            positive_count < route["min_positive_samples"]
            or negative_count < route["min_negative_samples"]
        ):
            logger.info(
                "Skipping regime route %s: class counts pos=%d neg=%d",
                route["name"],
                positive_count,
                negative_count,
            )
            continue

        route_models.append(
            {
                "name": route["name"],
                "regimes": route["regimes"],
                "side": route["side"],
                "candidate": _fit_binary_candidate_model(route_df, training_cfg),
            }
        )

    if not route_models:
        candidate = dict(global_candidate)
        candidate["mode"] = "single"
        candidate["trained_sides"] = []
        return candidate

    return {
        "mode": "regime_routed",
        "positive_label": positive_label,
        "global_model": global_candidate,
        "route_models": route_models,
    }


def _fit_candidate_model(train_df: pd.DataFrame, training_cfg: dict) -> dict:
    regime_routing_cfg = training_cfg.get("regime_routing", {}) or {}
    if regime_routing_cfg.get("enabled", False):
        return _fit_regime_routed_candidate(train_df, training_cfg)

    side_cfg = training_cfg.get("side_specialization", {}) or {}
    if not side_cfg.get("enabled", False):
        candidate = _fit_binary_candidate_model(train_df, training_cfg)
        candidate["mode"] = "single"
        candidate["trained_sides"] = []
        return candidate

    min_side_train_samples = int(side_cfg.get("min_side_train_samples", 250))
    min_side_positive = int(side_cfg.get("min_side_positive_samples", 25))
    min_side_negative = int(side_cfg.get("min_side_negative_samples", 25))
    fallback_to_global = bool(side_cfg.get("fallback_to_global", True))

    target_cfg = training_cfg.get("target", {}) or {}
    positive_label = int(target_cfg.get("positive_label", 1))
    side_models: dict[int, dict] = {}
    trained_sides: list[int] = []

    global_candidate = None
    if fallback_to_global:
        global_candidate = _fit_binary_candidate_model(train_df, training_cfg)

    for side in [1, -1]:
        side_df = train_df.loc[train_df["side"] == side].reset_index(drop=True)
        if len(side_df) < min_side_train_samples:
            logger.info(
                "Skipping side-specialized model for side=%s: only %d rows (need %d)",
                side,
                len(side_df),
                min_side_train_samples,
            )
            continue

        y_side = _binary_target(side_df["y"].to_numpy(), positive_label)
        positive_count = int(y_side.sum())
        negative_count = int(len(y_side) - positive_count)
        if positive_count < min_side_positive or negative_count < min_side_negative:
            logger.info(
                "Skipping side-specialized model for side=%s: class counts pos=%d neg=%d",
                side,
                positive_count,
                negative_count,
            )
            continue

        side_models[side] = _fit_binary_candidate_model(side_df, training_cfg)
        trained_sides.append(side)

    if not side_models:
        logger.warning(
            "Side specialization enabled but no side-specific models were trainable; using single model"
        )
        candidate = global_candidate or _fit_binary_candidate_model(train_df, training_cfg)
        candidate["mode"] = "single"
        candidate["trained_sides"] = []
        return candidate

    return {
        "mode": "side_specialized",
        "positive_label": positive_label,
        "side_models": side_models,
        "trained_sides": sorted(trained_sides),
        "global_model": global_candidate,
    }


def _predict_binary_candidate_proba(test_df: pd.DataFrame, candidate: dict) -> np.ndarray:
    if test_df is None or test_df.empty:
        return np.array([], dtype=float)
    pre = candidate["preprocessor"]
    X_test, _y_test_raw, _w_test = pre.transform(test_df)
    if X_test.shape[0] == 0:
        return np.array([], dtype=float)
    return candidate["model"].predict_proba(X_test)[:, 1]


def _score_candidate(test_df: pd.DataFrame, candidate: dict, training_cfg: dict) -> tuple[dict, pd.DataFrame]:
    y_test = _binary_target(test_df["y"].to_numpy(), candidate["positive_label"])

    if candidate.get("mode") == "regime_routed":
        proba = _predict_binary_candidate_proba(test_df, candidate["global_model"])
        model_source = np.full(len(test_df), "global_model", dtype=object)

        for route in candidate.get("route_models", []):
            route_mask = test_df["combined_regime"].isin(route["regimes"])
            route_mask &= _candidate_side_mask(test_df["side"], route["side"])
            route_mask = route_mask.to_numpy()
            if not route_mask.any():
                continue
            proba[route_mask] = _predict_binary_candidate_proba(
                test_df.loc[route_mask],
                route["candidate"],
            )
            model_source[route_mask] = f"route:{route['name']}"
    elif candidate.get("mode") == "side_specialized":
        proba = np.full(len(test_df), np.nan, dtype=float)
        model_source = np.full(len(test_df), "", dtype=object)

        for side in [1, -1]:
            side_mask = (test_df["side"] == side).to_numpy()
            if not side_mask.any():
                continue
            side_candidate = candidate["side_models"].get(side) or candidate.get("global_model")
            if side_candidate is None:
                raise ValueError(f"No candidate model available for side {side}")
            proba[side_mask] = _predict_binary_candidate_proba(test_df.loc[side_mask], side_candidate)
            model_source[side_mask] = "side_model" if side in candidate["side_models"] else "global_fallback"

        if np.isnan(proba).any():
            raise ValueError("Side-specialized scorer produced NaN probabilities")
    else:
        proba = _predict_binary_candidate_proba(test_df, candidate)
        model_source = np.full(len(test_df), "single_model", dtype=object)

    threshold = float((training_cfg.get("eval", {}) or {}).get("threshold", 0.5))
    pred = (proba >= threshold).astype(int)

    classification = {
        "test_auc": _safe_auc(y_test, proba),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "positive_prediction_rate": float(pred.mean()) if len(pred) else None,
        "target_rate": float(y_test.mean()) if len(y_test) else None,
    }
    primary_preds = pd.DataFrame(
        {
            "event_id": test_df["event_id"].to_numpy(),
            "y_prob": proba,
            "score_ev": (2.0 * proba) - 1.0,
            "model_source": model_source,
        }
    )
    return classification, primary_preds


def _period_bounds(window_cfg: dict, default_lookback_days: int, default_gap_days: int) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    test_start = _to_utc_timestamp(window_cfg["test_start"])
    test_end = _to_utc_timestamp(window_cfg["test_end"], end_of_day=True)

    if window_cfg.get("train_start") and window_cfg.get("train_end"):
        train_start = _to_utc_timestamp(window_cfg["train_start"])
        train_end = _to_utc_timestamp(window_cfg["train_end"], end_of_day=True)
    else:
        lookback_days = int(window_cfg.get("train_lookback_days", default_lookback_days))
        gap_days = int(window_cfg.get("gap_days", default_gap_days))
        train_end = test_start - pd.Timedelta(days=gap_days) - pd.Timedelta(seconds=1)
        train_start = train_end - pd.Timedelta(days=lookback_days)

    return train_start, train_end, test_start, test_end


def _fit_promotion_window_artifacts(
    *,
    name: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    bars_train: pd.DataFrame,
    bars_test: pd.DataFrame,
    bar_size: str,
    labeling_cfg: dict,
    execution_spec: dict,
    instrument_spec,
    feature_cfg: dict,
    training_cfg: dict,
    backtest_cfg: dict,
    bars_test_label_context: pd.DataFrame | None = None,
    session_day_chicago: date | None = None,
) -> dict:
    """
    Train primary + meta for one promotion window; return everything needed for run_backtest.
    """
    train_events, train_df = _prepare_events_and_dataset(
        bars_df=bars_train,
        bar_size=bar_size,
        labeling_cfg=labeling_cfg,
        execution_spec=execution_spec,
        instrument_spec=instrument_spec,
        feature_cfg=feature_cfg,
        training_cfg=training_cfg,
        balance_train=True,
    )
    if bars_test_label_context is not None and session_day_chicago is not None:
        test_events, test_df = _prepare_events_and_dataset_session_day(
            bars_df=bars_test_label_context,
            session_date=session_day_chicago,
            bar_size=bar_size,
            labeling_cfg=labeling_cfg,
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            feature_cfg=feature_cfg,
            training_cfg=training_cfg,
        )
        bars_for_backtest = bars_test_label_context
    else:
        test_events, test_df = _prepare_events_and_dataset(
            bars_df=bars_test,
            bar_size=bar_size,
            labeling_cfg=labeling_cfg,
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            feature_cfg=feature_cfg,
            training_cfg=training_cfg,
            balance_train=False,
        )
        bars_for_backtest = bars_test

    candidate = _fit_candidate_model(train_df, training_cfg)
    _train_classification, primary_train_preds = _score_candidate(train_df, candidate, training_cfg)
    classification, primary_preds = _score_candidate(test_df, candidate, training_cfg)
    meta_result = _train_targeted_meta_model(
        train_df=train_df,
        test_df=test_df,
        train_events=train_events,
        test_events=test_events,
        primary_train_preds=primary_train_preds,
        primary_test_preds=primary_preds,
        training_cfg=training_cfg,
    )
    local_backtest_cfg = deepcopy(backtest_cfg)
    local_backtest_cfg.setdefault("decision", {})
    meta_preds = None
    if meta_result.get("enabled", False) and not meta_result.get("skipped", False):
        local_backtest_cfg["decision"]["use_meta"] = True
        local_backtest_cfg["decision"]["meta_threshold"] = float(
            (training_cfg.get("meta", {}) or {}).get("threshold_meta", 0.5)
        )
        local_backtest_cfg["decision"]["require_meta_for_trade"] = bool(
            (training_cfg.get("meta", {}) or {}).get("require_meta_for_trade", False)
        )
        meta_preds = meta_result["meta_preds"]
    else:
        local_backtest_cfg["decision"]["use_meta"] = False

    primary_score_diagnostics = _build_primary_score_diagnostics(
        test_df=test_df,
        test_events=test_events,
        primary_preds=primary_preds,
        positive_label=candidate["positive_label"],
    )
    meta_score_diagnostics = _build_meta_score_diagnostics(
        meta_result=meta_result,
        test_df=test_df,
        test_events=test_events,
    )
    meta_routes_summary = [
        {
            "name": route.get("name"),
            "skipped": bool(route.get("skipped", False)),
            "reason": route.get("reason"),
            "threshold_primary": (route.get("route", {}) or {}).get("threshold_primary"),
            "threshold_meta": (route.get("route", {}) or {}).get("threshold_meta"),
            "test_rows": int(len(route.get("meta_test_df", []))),
            "metrics": route.get("metrics"),
        }
        for route in (meta_result.get("routes", []) or [])
    ]
    target_route_summary = {
        "train": _summarize_target_routes(train_df, train_events),
        "test": _summarize_target_routes(test_df, test_events),
    }
    primary_overall = primary_score_diagnostics.get("overall", {}) or {}
    classification["rank_ic_ret"] = primary_overall.get("rank_ic_ret")
    classification["ece"] = primary_overall.get("ece")
    classification["decile_spread_ret_net"] = primary_overall.get("decile_spread_ret_net")

    event_direction = summarize_trade_directions(
        int((test_events["side"] > 0).sum()),
        int((test_events["side"] < 0).sum()),
    )

    return {
        "name": name,
        "train_start": train_start,
        "train_end": train_end,
        "test_start": test_start,
        "test_end": test_end,
        "train_events": int(len(train_df)),
        "test_events": int(len(test_df)),
        "test_events_df": test_events,
        "bars_test": bars_for_backtest,
        "primary_preds": primary_preds,
        "meta_preds": meta_preds,
        "local_backtest_cfg": local_backtest_cfg,
        "classification": classification,
        "meta_result": meta_result,
        "meta_routes_summary": meta_routes_summary,
        "primary_score_diagnostics": primary_score_diagnostics,
        "meta_score_diagnostics": meta_score_diagnostics,
        "target_route_summary": target_route_summary,
        "test_event_direction_summary": event_direction,
    }


def _build_window_result_from_backtest(art: dict, trades_df: pd.DataFrame, equity_df: pd.DataFrame, backtest_metrics: dict) -> dict:
    direction_summary = summarize_trades_df(trades_df)
    risk_tearsheet = _build_risk_tearsheet(trades_df, equity_df)
    meta_result = art["meta_result"]
    return {
        "name": art["name"],
        "train_period": {"start": str(art["train_start"]), "end": str(art["train_end"])},
        "test_period": {"start": str(art["test_start"]), "end": str(art["test_end"])},
        "train_events": art["train_events"],
        "test_events": art["test_events"],
        "classification": art["classification"],
        "meta": {
            "enabled": bool(meta_result.get("enabled", False)),
            "skipped": bool(meta_result.get("skipped", False)),
            "reason": meta_result.get("reason"),
            "metrics": meta_result.get("metrics"),
            "routes": art["meta_routes_summary"],
            "test_rows": int(len(meta_result.get("meta_test_df", [])))
            if meta_result.get("enabled", False)
            else 0,
        },
        "score_diagnostics": {
            "primary": art["primary_score_diagnostics"],
            "meta": art["meta_score_diagnostics"],
        },
        "target_routes": art["target_route_summary"],
        "backtest_metrics": backtest_metrics,
        "risk_tearsheet": risk_tearsheet,
        "direction_summary": direction_summary,
        "test_event_direction_summary": art["test_event_direction_summary"],
    }


def _persist_window_outputs(
    window_dir: Path,
    window_result: dict,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    art: dict,
) -> None:
    window_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_parquet(window_dir / "trades.parquet")
    equity_df.to_parquet(window_dir / "equity.parquet")
    with open(window_dir / "risk_tearsheet.json", "w") as f:
        json.dump(window_result["risk_tearsheet"], f, indent=2)
    with open(window_dir / "primary_score_diagnostics.json", "w") as f:
        json.dump(art["primary_score_diagnostics"], f, indent=2)
    with open(window_dir / "meta_score_diagnostics.json", "w") as f:
        json.dump(art["meta_score_diagnostics"], f, indent=2)
    meta_preds = art["meta_preds"]
    if meta_preds is not None:
        meta_preds.to_parquet(window_dir / "meta_preds.parquet")
        with open(window_dir / "meta_metrics.json", "w") as f:
            json.dump(art["meta_result"].get("metrics", {}), f, indent=2)
    with open(window_dir / "window_summary.json", "w") as f:
        json.dump(window_result, f, indent=2)


def _render_report(summary: dict, output_dir: Path) -> None:
    def _fmt(value, kind: str = "plain") -> str:
        if value is None:
            return "None"
        if kind == "usd":
            return f"${float(value):.2f}"
        if kind == "ratio":
            return f"{float(value):.4f}"
        return str(value)

    lines = [
        "# Standalone ML Candidate Report",
        "",
        f"- Generated: `{datetime.utcnow().isoformat()}Z`",
        f"- Overall pass: `{summary['passed']}`",
        f"- Passing windows: `{summary['passed_windows']}/{summary['total_windows']}`",
        f"- Pass ratio: `{summary['pass_ratio']:.2f}`",
        f"- Overall total PnL: `{_fmt(summary['overall_total_pnl_usd'], 'usd')}`",
        "",
        "## Overall Direction",
        "",
        f"- Long trades: `{summary['overall_direction_summary']['long_trades']}`",
        f"- Short trades: `{summary['overall_direction_summary']['short_trades']}`",
        f"- Long share: `{summary['overall_direction_summary']['long_share']}`",
        f"- Short share: `{summary['overall_direction_summary']['short_share']}`",
        "",
        "## Overall Risk",
        "",
    ]

    overall_risk = summary.get("risk_tearsheet", {}) or {}
    overall_daily = overall_risk.get("daily", {}) or {}
    overall_streaks = overall_risk.get("streaks", {}) or {}
    overall_equity = overall_risk.get("equity", {}) or {}
    lines.extend(
        [
            f"- Worst day: `{_fmt(overall_daily.get('worst_day_usd'), 'usd')}`",
            f"- Best day: `{_fmt(overall_daily.get('best_day_usd'), 'usd')}`",
            f"- Daily Sharpe: `{_fmt(overall_daily.get('daily_sharpe'), 'ratio')}`",
            f"- Longest loss streak: `{overall_streaks.get('longest_loss_streak')}`",
            f"- Recovery factor: `{_fmt(overall_equity.get('recovery_factor'), 'ratio')}`",
            "",
        "## Overall Failures",
        "",
        ]
    )

    if summary["overall_failures"]:
        for failure in summary["overall_failures"]:
            lines.append(
                f"- `{failure['metric']}` actual=`{failure['actual']}` expected=`{failure['expected']}`"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Windows", ""])
    for window in summary["windows"]:
        backtest = window.get("backtest_metrics", {}) or {}
        classification = window.get("classification", {}) or {}
        direction = window.get("direction_summary", {}) or {}
        score_diag = (window.get("score_diagnostics", {}) or {}).get("primary", {}) or {}
        risk_tearsheet = window.get("risk_tearsheet", {}) or {}
        daily_risk = risk_tearsheet.get("daily", {}) or {}
        streak_risk = risk_tearsheet.get("streaks", {}) or {}
        problem_groups = score_diag.get("problem_groups", []) or []
        worst_short_group = next(
            (group for group in problem_groups if group.get("side_label") == "short"),
            None,
        )
        lines.extend(
            [
                f"### {window['name']}",
                "",
                f"- Passed: `{window['gate_result']['passed']}`",
                f"- Train events: `{window['train_events']}`",
                f"- Test events: `{window['test_events']}`",
                f"- Test AUC: `{_fmt(classification.get('test_auc'), 'ratio')}`",
                f"- Test accuracy: `{_fmt(classification.get('test_accuracy'), 'ratio')}`",
                f"- Total PnL: `{_fmt(backtest.get('total_pnl_usd'), 'usd')}`",
                f"- Profit factor: `{_fmt(backtest.get('profit_factor'), 'ratio')}`",
                f"- Max drawdown: `{_fmt(backtest.get('max_drawdown_usd'), 'usd')}`",
                f"- Rank IC vs ret_net: `{_fmt(score_diag.get('overall', {}).get('rank_ic_ret'), 'ratio')}`",
                f"- ECE: `{_fmt(score_diag.get('overall', {}).get('ece'), 'ratio')}`",
                f"- Decile spread: `{_fmt(score_diag.get('overall', {}).get('decile_spread_ret_net'), 'ratio')}`",
                f"- Long trades: `{direction.get('long_trades', 0)}`",
                f"- Short trades: `{direction.get('short_trades', 0)}`",
                f"- Worst day: `{_fmt(daily_risk.get('worst_day_usd'), 'usd')}`",
                f"- Longest loss streak: `{streak_risk.get('longest_loss_streak')}`",
            ]
        )
        if worst_short_group:
            lines.append(
                "- Worst short bucket: "
                f"`{worst_short_group.get('month')}` / "
                f"`{worst_short_group.get('regime_label')}` "
                f"avg_ret_net=`{_fmt(worst_short_group.get('avg_ret_net'), 'ratio')}` "
                f"rank_ic=`{_fmt(worst_short_group.get('rank_ic_ret'), 'ratio')}`"
            )
        if window["gate_result"]["failures"]:
            lines.append("- Failures:")
            for failure in window["gate_result"]["failures"]:
                lines.append(
                    f"  - `{failure['metric']}` actual=`{failure['actual']}` expected=`{failure['expected']}`"
                )
        lines.append("")

    report_path = output_dir / "promotion_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_candidate(
    data_path: Path,
    hdf_key: str,
    training_cfg_path: Path,
    labeling_cfg_path: Path,
    feature_cfg_path: Path,
    execution_spec_path: Path,
    backtest_cfg_path: Path,
    risk_cfg_path: Path,
    acceptance_cfg_path: Path,
    output_dir: Path,
) -> dict:
    training_cfg = _load_yaml(training_cfg_path)
    labeling_cfg = _load_yaml(labeling_cfg_path)
    feature_cfg = _load_yaml(feature_cfg_path)
    execution_spec = _load_yaml(execution_spec_path)
    backtest_cfg = _load_yaml(backtest_cfg_path)
    risk_cfg = _load_yaml(risk_cfg_path)
    acceptance_cfg = _load_yaml(acceptance_cfg_path)
    from ml_intraday_v3.analysis.decision_parity import assert_decision_parity
    assert_decision_parity(backtest_cfg, execution_spec)
    data_cfg = acceptance_cfg.get("data", {}) or {}
    additional_paths = [PROJECT_ROOT / path for path in data_cfg.get("additional_paths", [])]
    bars = _load_bars([data_path, *additional_paths], hdf_key)

    instrument_spec = load_instrument_from_execution_spec(execution_spec_path)
    label_schema = {"schema_version": "1.0.0", "cost_mode": _derive_cost_mode(labeling_cfg)}

    output_dir.mkdir(parents=True, exist_ok=True)

    bar_size = data_cfg.get("bar_size", "5m")
    schedule_cfg = acceptance_cfg.get("training_window", {}) or {}
    windows = acceptance_cfg.get("windows", []) or []
    if not windows:
        raise ValueError("acceptance config must define at least one evaluation window")

    window_results = []
    all_trades = []
    all_equity = []
    for window_cfg in windows:
        name = window_cfg["name"]
        train_start, train_end, test_start, test_end = _period_bounds(
            window_cfg=window_cfg,
            default_lookback_days=int(schedule_cfg.get("lookback_days", 180)),
            default_gap_days=int(schedule_cfg.get("gap_days", 0)),
        )
        bars_train = bars[(bars.index >= train_start) & (bars.index <= train_end)].copy()
        bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)].copy()
        if bars_train.empty or bars_test.empty:
            raise ValueError(f"Window {name} has empty train/test bars")

        logger.info(
            "Window %s | train=%s..%s (%d bars) | test=%s..%s (%d bars)",
            name,
            train_start.date(),
            train_end.date(),
            len(bars_train),
            test_start.date(),
            test_end.date(),
            len(bars_test),
        )

        art = _fit_promotion_window_artifacts(
            name=name,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            bars_train=bars_train,
            bars_test=bars_test,
            bar_size=bar_size,
            labeling_cfg=labeling_cfg,
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            feature_cfg=feature_cfg,
            training_cfg=training_cfg,
            backtest_cfg=backtest_cfg,
        )

        trades_df, equity_df, backtest_metrics = run_backtest(
            events_df=art["test_events_df"],
            bars_df=art["bars_test"],
            primary_preds_df=art["primary_preds"],
            meta_preds_df=art["meta_preds"],
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            label_schema=label_schema,
            risk_cfg=risk_cfg,
            backtest_cfg=art["local_backtest_cfg"],
            bar_size=bar_size,
        )

        window_result = _build_window_result_from_backtest(art, trades_df, equity_df, backtest_metrics)
        window_results.append(window_result)
        all_trades.append(trades_df.assign(window_name=name))
        all_equity.append(equity_df.assign(window_name=name))

        window_dir = output_dir / name
        _persist_window_outputs(window_dir, window_result, trades_df, equity_df, art)

    summary = evaluate_promotion_gates(window_results, acceptance_cfg.get("gates", {}) or {})
    overall_risk_tearsheet = _build_risk_tearsheet(
        pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame(),
    )
    summary["risk_tearsheet"] = overall_risk_tearsheet
    with open(output_dir / "promotion_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "overall_risk_tearsheet.json", "w") as f:
        json.dump(overall_risk_tearsheet, f, indent=2)
    _render_report(summary, output_dir)
    return summary


def run_candidate_contract_variants(
    data_path: Path,
    hdf_key: str,
    training_cfg_path: Path,
    labeling_cfg_path: Path,
    feature_cfg_path: Path,
    execution_spec_path: Path,
    backtest_cfg_path: Path,
    risk_cfg_path: Path,
    acceptance_cfg_path: Path,
    output_dir: Path,
    contract_counts: tuple[int, ...] = (1, 2, 3),
    scale_risk_with_contracts: bool = True,
) -> dict[int, dict]:
    """
    Train each promotion window once, then run backtests for each contract count.

    When ``scale_risk_with_contracts`` is True, dollar daily/DD limits scale with
    ``contracts`` so multi-lot paths are comparable to the 1-lot risk budget.

    Writes under ``output_dir / f"contracts_{n}" / <window_name>/`` mirroring
    ``run_candidate`` layout per variant.
    """
    training_cfg = _load_yaml(training_cfg_path)
    labeling_cfg = _load_yaml(labeling_cfg_path)
    feature_cfg = _load_yaml(feature_cfg_path)
    execution_spec = _load_yaml(execution_spec_path)
    backtest_cfg = _load_yaml(backtest_cfg_path)
    base_risk_cfg = _load_yaml(risk_cfg_path)
    acceptance_cfg = _load_yaml(acceptance_cfg_path)
    data_cfg = acceptance_cfg.get("data", {}) or {}
    additional_paths = [PROJECT_ROOT / path for path in data_cfg.get("additional_paths", [])]
    bars = _load_bars([data_path, *additional_paths], hdf_key)

    instrument_spec = load_instrument_from_execution_spec(execution_spec_path)
    label_schema = {"schema_version": "1.0.0", "cost_mode": _derive_cost_mode(labeling_cfg)}

    output_dir.mkdir(parents=True, exist_ok=True)

    bar_size = data_cfg.get("bar_size", "5m")
    schedule_cfg = acceptance_cfg.get("training_window", {}) or {}
    windows = acceptance_cfg.get("windows", []) or []
    if not windows:
        raise ValueError("acceptance config must define at least one evaluation window")

    window_arts: list[dict] = []
    for window_cfg in windows:
        name = window_cfg["name"]
        train_start, train_end, test_start, test_end = _period_bounds(
            window_cfg=window_cfg,
            default_lookback_days=int(schedule_cfg.get("lookback_days", 180)),
            default_gap_days=int(schedule_cfg.get("gap_days", 0)),
        )
        bars_train = bars[(bars.index >= train_start) & (bars.index <= train_end)].copy()
        bars_test = bars[(bars.index >= test_start) & (bars.index <= test_end)].copy()
        if bars_train.empty or bars_test.empty:
            raise ValueError(f"Window {name} has empty train/test bars")

        logger.info(
            "Train-once window %s | train=%s..%s (%d bars) | test=%s..%s (%d bars)",
            name,
            train_start.date(),
            train_end.date(),
            len(bars_train),
            test_start.date(),
            test_end.date(),
            len(bars_test),
        )

        art = _fit_promotion_window_artifacts(
            name=name,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            bars_train=bars_train,
            bars_test=bars_test,
            bar_size=bar_size,
            labeling_cfg=labeling_cfg,
            execution_spec=execution_spec,
            instrument_spec=instrument_spec,
            feature_cfg=feature_cfg,
            training_cfg=training_cfg,
            backtest_cfg=backtest_cfg,
        )
        window_arts.append(art)

    summaries: dict[int, dict] = {}
    for c in contract_counts:
        c = int(c)
        variant_dir = output_dir / f"contracts_{c}"
        window_results: list[dict] = []
        all_trades: list[pd.DataFrame] = []
        all_equity: list[pd.DataFrame] = []
        risk_eff = (
            scale_risk_config_for_contracts(base_risk_cfg, c, base_contracts=1)
            if scale_risk_with_contracts
            else deepcopy(base_risk_cfg)
        )

        for art in window_arts:
            local_bt = deepcopy(art["local_backtest_cfg"])
            local_bt.setdefault("sizing", {})["contracts"] = c

            trades_df, equity_df, backtest_metrics = run_backtest(
                events_df=art["test_events_df"],
                bars_df=art["bars_test"],
                primary_preds_df=art["primary_preds"],
                meta_preds_df=art["meta_preds"],
                execution_spec=execution_spec,
                instrument_spec=instrument_spec,
                label_schema=label_schema,
                risk_cfg=risk_eff,
                backtest_cfg=local_bt,
                bar_size=bar_size,
            )

            window_result = _build_window_result_from_backtest(art, trades_df, equity_df, backtest_metrics)
            window_results.append(window_result)
            all_trades.append(trades_df.assign(window_name=art["name"]))
            all_equity.append(equity_df.assign(window_name=art["name"]))

            window_dir = variant_dir / art["name"]
            _persist_window_outputs(window_dir, window_result, trades_df, equity_df, art)

        summary = evaluate_promotion_gates(window_results, acceptance_cfg.get("gates", {}) or {})
        overall_risk_tearsheet = _build_risk_tearsheet(
            pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
            pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame(),
        )
        summary["risk_tearsheet"] = overall_risk_tearsheet
        meta = {
            "contract_count": c,
            "scale_risk_with_contracts": scale_risk_with_contracts,
        }
        summary["contract_sweep_meta"] = meta
        with open(variant_dir / "promotion_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(variant_dir / "overall_risk_tearsheet.json", "w") as f:
            json.dump(overall_risk_tearsheet, f, indent=2)
        _render_report(summary, variant_dir)
        summaries[c] = summary

    with open(output_dir / "contract_variants_aggregate.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "contract_counts": list(contract_counts),
                "scale_risk_with_contracts": scale_risk_with_contracts,
                "summaries": {str(k): v for k, v in summaries.items()},
            },
            f,
            indent=2,
            default=str,
        )

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone ML candidate promotion check")
    parser.add_argument(
        "--data-path",
        default="data/processed/mes_bars_databento_rth.h5",
        help="Path to HDF bars file",
    )
    parser.add_argument("--hdf-key", default="bars_5min", help="HDF key for bars")
    parser.add_argument(
        "--training-config",
        default="ml_intraday_v3/configs/training_standalone_topstep.yaml",
        help="Training config path",
    )
    parser.add_argument(
        "--labeling-config",
        default="ml_intraday_v3/configs/labeling.yaml",
        help="Labeling config path",
    )
    parser.add_argument(
        "--features-config",
        default="ml_intraday_v3/configs/features.yaml",
        help="Features config path",
    )
    parser.add_argument(
        "--execution-spec",
        default="ml_intraday_v3/configs/execution_spec.yaml",
        help="Execution spec path",
    )
    parser.add_argument(
        "--backtest-config",
        default="ml_intraday_v3/configs/backtest_standalone_topstep.yaml",
        help="Backtest config path",
    )
    parser.add_argument(
        "--risk-config",
        default="ml_intraday_v3/configs/risk_topstep_50k_strict_nolock.yaml",
        help="Risk config path",
    )
    parser.add_argument(
        "--acceptance-config",
        default="ml_intraday_v3/configs/standalone_viability.yaml",
        help="Promotion-gate config path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (defaults to timestamped folder under experiments/results)",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "ml_intraday_v3" / "experiments" / "results" / f"standalone_topstep_{ts}"

    summary = run_candidate(
        data_path=PROJECT_ROOT / args.data_path,
        hdf_key=args.hdf_key,
        training_cfg_path=PROJECT_ROOT / args.training_config,
        labeling_cfg_path=PROJECT_ROOT / args.labeling_config,
        feature_cfg_path=PROJECT_ROOT / args.features_config,
        execution_spec_path=PROJECT_ROOT / args.execution_spec,
        backtest_cfg_path=PROJECT_ROOT / args.backtest_config,
        risk_cfg_path=PROJECT_ROOT / args.risk_config,
        acceptance_cfg_path=PROJECT_ROOT / args.acceptance_config,
        output_dir=output_dir,
    )
    logger.info("Standalone candidate pass=%s | windows=%d/%d", summary["passed"], summary["passed_windows"], summary["total_windows"])


if __name__ == "__main__":
    main()
